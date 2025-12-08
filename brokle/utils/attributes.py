"""
Utility functions for working with OpenTelemetry span attributes.

Provides helpers for setting attributes, serializing messages, and
handling structured content.
"""

import json
from typing import Any, Dict, List, Optional, Union

from opentelemetry.sdk.trace import Span


def set_span_attributes(span: Span, attributes: Dict[str, Any]) -> None:
    """
    Set multiple attributes on a span with type coercion.

    OpenTelemetry attributes must be primitives (str, int, float, bool) or
    lists/tuples of primitives. This function handles serialization of
    complex types.

    Args:
        span: OpenTelemetry span
        attributes: Dictionary of attributes to set

    Example:
        >>> set_span_attributes(span, {
        ...     "key1": "value",
        ...     "key2": {"nested": "object"},  # Serialized to JSON
        ...     "key3": [1, 2, 3],
        ... })
    """
    for key, value in attributes.items():
        if value is None:
            continue

        # Handle primitives directly
        if isinstance(value, (str, int, float, bool)):
            span.set_attribute(key, value)

        # Handle lists/tuples of primitives
        elif isinstance(value, (list, tuple)):
            # Check if all elements are primitives
            if all(isinstance(x, (str, int, float, bool)) for x in value):
                span.set_attribute(key, value)
            else:
                # Serialize complex list to JSON
                span.set_attribute(key, json.dumps(value))

        # Handle dicts and other complex types
        else:
            # Serialize to JSON string
            span.set_attribute(key, json.dumps(value, default=str))


def serialize_messages(
    messages: List[Dict[str, Any]],
    exclude_keys: Optional[List[str]] = None,
) -> str:
    """
    Serialize messages to OTEL GenAI format (JSON).

    Handles OpenAI/Anthropic message formats and converts them to
    OTEL GenAI 1.28+ compliant JSON format.

    Args:
        messages: List of message dictionaries
        exclude_keys: Optional keys to exclude from serialization

    Returns:
        JSON string of messages

    Example:
        >>> messages = [
        ...     {"role": "user", "content": "Hello"},
        ...     {"role": "assistant", "content": "Hi there!"},
        ... ]
        >>> json_str = serialize_messages(messages)
        >>> # Returns: '[{"role":"user","content":"Hello"},...]'
    """
    exclude_keys = exclude_keys or []

    serialized = []
    for msg in messages:
        # Filter out excluded keys
        filtered_msg = {k: v for k, v in msg.items() if k not in exclude_keys}

        # Handle Pydantic models
        if hasattr(msg, "model_dump"):
            filtered_msg = msg.model_dump(exclude_none=True)

        serialized.append(filtered_msg)

    return json.dumps(serialized, default=str)


def extract_system_messages(
    messages: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract system messages from message list.

    OTEL GenAI 1.28+ recommends separating system messages into
    gen_ai.system_instructions for privacy/filtering.

    Args:
        messages: List of message dictionaries

    Returns:
        Tuple of (non_system_messages, system_messages)

    Example:
        >>> messages = [
        ...     {"role": "system", "content": "You are helpful"},
        ...     {"role": "user", "content": "Hello"},
        ... ]
        >>> regular, system = extract_system_messages(messages)
        >>> len(regular) == 1
        True
        >>> len(system) == 1
        True
    """
    system_messages = []
    non_system_messages = []

    for msg in messages:
        role = msg.get("role", "")
        if role == "system":
            system_messages.append(msg)
        else:
            non_system_messages.append(msg)

    return non_system_messages, system_messages


def format_tool_calls(tool_calls: List[Any]) -> List[Dict[str, Any]]:
    """
    Format tool calls for OTEL GenAI format.

    Handles OpenAI tool call format and converts to structured format.

    Args:
        tool_calls: List of tool call objects

    Returns:
        List of formatted tool call dictionaries

    Example:
        >>> # OpenAI format
        >>> tool_calls = [
        ...     {
        ...         "id": "call_123",
        ...         "type": "function",
        ...         "function": {
        ...             "name": "get_weather",
        ...             "arguments": '{"location": "SF"}'
        ...         }
        ...     }
        ... ]
        >>> formatted = format_tool_calls(tool_calls)
    """
    formatted = []

    for call in tool_calls:
        # Handle Pydantic models
        if hasattr(call, "model_dump"):
            call_dict = call.model_dump(exclude_none=True)
        elif isinstance(call, dict):
            call_dict = call
        else:
            call_dict = {"raw": str(call)}

        formatted.append(call_dict)

    return formatted


def mask_sensitive_content(
    content: str,
    mask_fn: Optional[callable] = None,
) -> str:
    """
    Mask sensitive content using custom masking function.

    Args:
        content: Content to mask
        mask_fn: Optional custom masking function

    Returns:
        Masked content

    Example:
        >>> def simple_mask(text):
        ...     return text.replace("secret", "[REDACTED]")
        >>> masked = mask_sensitive_content("my secret key", simple_mask)
        >>> masked
        'my [REDACTED] key'
    """
    if mask_fn is None:
        return content

    try:
        return mask_fn(content)
    except Exception:
        # If masking fails, return original content
        # This prevents telemetry failures due to masking errors
        return content


def calculate_total_tokens(
    input_tokens: Optional[int],
    output_tokens: Optional[int],
) -> Optional[int]:
    """
    Calculate total tokens from input and output tokens.

    Note: total_tokens is NOT in OTEL GenAI spec, so we store it
    under brokle.usage.total_tokens.

    Args:
        input_tokens: Input token count
        output_tokens: Output token count

    Returns:
        Total tokens or None if either input is None

    Example:
        >>> total = calculate_total_tokens(100, 50)
        >>> total
        150
        >>> calculate_total_tokens(None, 50) is None
        True
    """
    if input_tokens is None or output_tokens is None:
        return None

    return input_tokens + output_tokens


def extract_model_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract standard model parameters from provider-specific format.

    Maps provider parameters to OTEL GenAI standard attributes.

    Args:
        params: Provider-specific parameters

    Returns:
        Dictionary of OTEL-compliant parameter attributes

    Example:
        >>> params = {
        ...     "temperature": 0.7,
        ...     "max_tokens": 100,
        ...     "custom_param": "value",  # Ignored
        ... }
        >>> extracted = extract_model_parameters(params)
        >>> "gen_ai.request.temperature" in extracted
        True
    """
    from ..types import Attrs

    extracted = {}

    # Standard OTEL GenAI parameters
    if "temperature" in params:
        extracted[Attrs.GEN_AI_REQUEST_TEMPERATURE] = params["temperature"]
    if "max_tokens" in params:
        extracted[Attrs.GEN_AI_REQUEST_MAX_TOKENS] = params["max_tokens"]
    if "top_p" in params:
        extracted[Attrs.GEN_AI_REQUEST_TOP_P] = params["top_p"]
    if "top_k" in params:
        extracted[Attrs.GEN_AI_REQUEST_TOP_K] = params["top_k"]
    if "frequency_penalty" in params:
        extracted[Attrs.GEN_AI_REQUEST_FREQUENCY_PENALTY] = params["frequency_penalty"]
    if "presence_penalty" in params:
        extracted[Attrs.GEN_AI_REQUEST_PRESENCE_PENALTY] = params["presence_penalty"]
    if "stop" in params:
        extracted[Attrs.GEN_AI_REQUEST_STOP_SEQUENCES] = params["stop"]

    return extracted
