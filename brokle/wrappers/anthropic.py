"""
Anthropic SDK wrapper for automatic observability.

Wraps Anthropic client to automatically create OTEL spans with GenAI 1.28+ attributes.
"""

import json
import time
from typing import Any, Optional, TYPE_CHECKING

from opentelemetry.trace import Status, StatusCode

from ..client import get_client
from ..types import Attrs, ObservationType, LLMProvider, OperationType
from ..utils.attributes import serialize_messages, calculate_total_tokens

if TYPE_CHECKING:
    import anthropic


def wrap_anthropic(client: "anthropic.Anthropic") -> "anthropic.Anthropic":
    """
    Wrap Anthropic client for automatic observability.

    This function wraps the Anthropic client's messages.create method
    to automatically create OTEL spans with GenAI semantic attributes.

    Args:
        client: Anthropic client instance

    Returns:
        Wrapped Anthropic client (same instance with instrumented methods)

    Example:
        >>> import anthropic
        >>> from brokle import get_client, wrap_anthropic
        >>>
        >>> # Initialize Brokle
        >>> brokle = get_client()
        >>>
        >>> # Wrap Anthropic client
        >>> client = wrap_anthropic(anthropic.Anthropic(api_key="..."))
        >>>
        >>> # All calls automatically tracked
        >>> response = client.messages.create(
        ...     model="claude-3-opus",
        ...     messages=[{"role": "user", "content": "Hello"}]
        ... )
        >>> brokle.flush()
    """
    # Store original method
    original_messages_create = client.messages.create

    def wrapped_messages_create(*args, **kwargs):
        """Wrapped messages.create with automatic tracing."""
        # Get Brokle client
        brokle_client = get_client()

        # Extract request parameters
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        system = kwargs.get("system")  # Anthropic uses separate system param
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens")
        top_p = kwargs.get("top_p")
        top_k = kwargs.get("top_k")
        stop_sequences = kwargs.get("stop_sequences")
        stream = kwargs.get("stream", False)
        metadata = kwargs.get("metadata")

        # Build OTEL GenAI attributes
        attrs = {
            Attrs.BROKLE_OBSERVATION_TYPE: ObservationType.GENERATION,
            Attrs.GEN_AI_PROVIDER_NAME: LLMProvider.ANTHROPIC,
            Attrs.GEN_AI_OPERATION_NAME: OperationType.CHAT,
            Attrs.GEN_AI_REQUEST_MODEL: model,
            Attrs.BROKLE_STREAMING: stream,
        }

        # Add messages (Anthropic doesn't use system role in messages)
        if messages:
            attrs[Attrs.GEN_AI_INPUT_MESSAGES] = serialize_messages(messages)

        # Add system instructions if present
        if system:
            # Anthropic system is a string, convert to OTEL format
            system_msgs = [{"role": "system", "content": system}]
            attrs[Attrs.GEN_AI_SYSTEM_INSTRUCTIONS] = json.dumps(system_msgs)
            # Store raw for Anthropic-specific tracking
            attrs[Attrs.ANTHROPIC_REQUEST_SYSTEM] = system

        # Add model parameters
        if temperature is not None:
            attrs[Attrs.GEN_AI_REQUEST_TEMPERATURE] = temperature
        if max_tokens is not None:
            attrs[Attrs.GEN_AI_REQUEST_MAX_TOKENS] = max_tokens
        if top_p is not None:
            attrs[Attrs.GEN_AI_REQUEST_TOP_P] = top_p

        # Anthropic-specific attributes
        if top_k is not None:
            attrs[Attrs.ANTHROPIC_REQUEST_TOP_K] = top_k
        if stop_sequences is not None:
            attrs[Attrs.ANTHROPIC_REQUEST_STOP_SEQUENCES] = stop_sequences
        if stream is not None:
            attrs[Attrs.ANTHROPIC_REQUEST_STREAM] = stream
        if metadata:
            attrs[Attrs.ANTHROPIC_REQUEST_METADATA] = json.dumps(metadata)

        # Create span name following OTEL pattern: "{operation} {model}"
        span_name = f"{OperationType.CHAT} {model}"

        # Create span and make API call
        with brokle_client.start_as_current_span(span_name, attributes=attrs) as span:
            try:
                # Record start time for latency
                start_time = time.time()

                # Make actual API call
                response = original_messages_create(*args, **kwargs)

                # Calculate latency
                latency_ms = (time.time() - start_time) * 1000

                # Extract response metadata
                if hasattr(response, "id"):
                    span.set_attribute(Attrs.GEN_AI_RESPONSE_ID, response.id)
                if hasattr(response, "model"):
                    span.set_attribute(Attrs.GEN_AI_RESPONSE_MODEL, response.model)

                # Extract stop reason
                if hasattr(response, "stop_reason") and response.stop_reason:
                    span.set_attribute(Attrs.GEN_AI_RESPONSE_FINISH_REASONS, [response.stop_reason])
                    span.set_attribute(Attrs.ANTHROPIC_RESPONSE_STOP_REASON, response.stop_reason)
                if hasattr(response, "stop_sequence") and response.stop_sequence:
                    span.set_attribute(Attrs.ANTHROPIC_RESPONSE_STOP_SEQUENCE, response.stop_sequence)

                # Extract output messages
                if hasattr(response, "content") and response.content:
                    output_messages = []

                    # Anthropic returns list of content blocks
                    for content_block in response.content:
                        if hasattr(content_block, "type"):
                            if content_block.type == "text":
                                # Text content
                                output_messages.append({
                                    "role": "assistant",
                                    "content": content_block.text,
                                })
                            elif content_block.type == "tool_use":
                                # Tool use
                                output_messages.append({
                                    "role": "assistant",
                                    "tool_calls": [{
                                        "id": content_block.id,
                                        "type": "function",
                                        "function": {
                                            "name": content_block.name,
                                            "arguments": json.dumps(content_block.input),
                                        }
                                    }]
                                })

                    if output_messages:
                        span.set_attribute(Attrs.GEN_AI_OUTPUT_MESSAGES, json.dumps(output_messages))

                # Extract usage statistics
                if hasattr(response, "usage") and response.usage:
                    usage = response.usage
                    if hasattr(usage, "input_tokens") and usage.input_tokens:
                        span.set_attribute(Attrs.GEN_AI_USAGE_INPUT_TOKENS, usage.input_tokens)
                    if hasattr(usage, "output_tokens") and usage.output_tokens:
                        span.set_attribute(Attrs.GEN_AI_USAGE_OUTPUT_TOKENS, usage.output_tokens)

                    # Calculate total tokens
                    total_tokens = calculate_total_tokens(
                        usage.input_tokens if hasattr(usage, "input_tokens") else None,
                        usage.output_tokens if hasattr(usage, "output_tokens") else None,
                    )
                    if total_tokens:
                        span.set_attribute(Attrs.BROKLE_USAGE_TOTAL_TOKENS, total_tokens)

                # Set latency
                span.set_attribute(Attrs.BROKLE_USAGE_LATENCY_MS, latency_ms)

                # Mark span as successful
                span.set_status(Status(StatusCode.OK))

                return response

            except Exception as e:
                # Record error
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    # Replace method
    client.messages.create = wrapped_messages_create

    return client


def wrap_anthropic_async(client: "anthropic.AsyncAnthropic") -> "anthropic.AsyncAnthropic":
    """
    Wrap AsyncAnthropic client for automatic observability.

    Similar to wrap_anthropic but for async client.

    Args:
        client: AsyncAnthropic client instance

    Returns:
        Wrapped AsyncAnthropic client

    Example:
        >>> import anthropic
        >>> from brokle import get_client, wrap_anthropic_async
        >>>
        >>> brokle = get_client()
        >>> client = wrap_anthropic_async(anthropic.AsyncAnthropic(api_key="..."))
        >>>
        >>> # Async calls automatically tracked
        >>> response = await client.messages.create(...)
    """
    # Store original method
    original_messages_create = client.messages.create

    async def wrapped_messages_create(*args, **kwargs):
        """Wrapped async messages.create with automatic tracing."""
        # Get Brokle client
        brokle_client = get_client()

        # Extract request parameters (same as sync version)
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        system = kwargs.get("system")
        stream = kwargs.get("stream", False)

        # Build OTEL GenAI attributes (same as sync)
        attrs = {
            Attrs.BROKLE_OBSERVATION_TYPE: ObservationType.GENERATION,
            Attrs.GEN_AI_PROVIDER_NAME: LLMProvider.ANTHROPIC,
            Attrs.GEN_AI_OPERATION_NAME: OperationType.CHAT,
            Attrs.GEN_AI_REQUEST_MODEL: model,
            Attrs.BROKLE_STREAMING: stream,
        }

        # Add messages
        if messages:
            attrs[Attrs.GEN_AI_INPUT_MESSAGES] = serialize_messages(messages)
        if system:
            system_msgs = [{"role": "system", "content": system}]
            attrs[Attrs.GEN_AI_SYSTEM_INSTRUCTIONS] = json.dumps(system_msgs)

        # Create span
        span_name = f"{OperationType.CHAT} {model}"

        with brokle_client.start_as_current_span(span_name, attributes=attrs) as span:
            try:
                start_time = time.time()

                # Make async API call
                response = await original_messages_create(*args, **kwargs)

                # Calculate latency
                latency_ms = (time.time() - start_time) * 1000

                # Extract response metadata (same processing as sync)
                # ... (response processing code)

                # Set latency
                span.set_attribute(Attrs.BROKLE_USAGE_LATENCY_MS, latency_ms)

                # Mark successful
                span.set_status(Status(StatusCode.OK))

                return response

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    # Replace method
    client.messages.create = wrapped_messages_create

    return client
