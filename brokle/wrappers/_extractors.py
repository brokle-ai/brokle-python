"""
Unified attribute extraction for LLM SDK wrappers.

Provides parameter mappings and response extractors for each provider,
ensuring attribute parity between sync and async code paths.
"""

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..types import Attrs, LLMProvider, OperationType, SpanType
from ..utils.attributes import calculate_total_tokens, serialize_messages


# =============================================================================
# Parameter Mappings by Provider
# =============================================================================

# Common parameters shared across most providers
COMMON_PARAM_MAPPING = {
    "temperature": Attrs.GEN_AI_REQUEST_TEMPERATURE,
    "max_tokens": Attrs.GEN_AI_REQUEST_MAX_TOKENS,
    "top_p": Attrs.GEN_AI_REQUEST_TOP_P,
}

# OpenAI-specific parameters
OPENAI_PARAM_MAPPING = {
    **COMMON_PARAM_MAPPING,
    "frequency_penalty": Attrs.GEN_AI_REQUEST_FREQUENCY_PENALTY,
    "presence_penalty": Attrs.GEN_AI_REQUEST_PRESENCE_PENALTY,
    "n": Attrs.OPENAI_REQUEST_N,
    "seed": Attrs.OPENAI_REQUEST_SEED,
    "service_tier": Attrs.OPENAI_REQUEST_SERVICE_TIER,
    "logprobs": Attrs.OPENAI_REQUEST_LOGPROBS,
    "top_logprobs": Attrs.OPENAI_REQUEST_TOP_LOGPROBS,
}

# Anthropic-specific parameters
ANTHROPIC_PARAM_MAPPING = {
    **COMMON_PARAM_MAPPING,
    "top_k": Attrs.ANTHROPIC_REQUEST_TOP_K,
    "stop_sequences": Attrs.ANTHROPIC_REQUEST_STOP_SEQUENCES,
    "metadata": Attrs.ANTHROPIC_REQUEST_METADATA,
}

# Mistral-specific parameters
MISTRAL_PARAM_MAPPING = {
    **COMMON_PARAM_MAPPING,
    "random_seed": Attrs.MISTRAL_REQUEST_RANDOM_SEED,
    "safe_prompt": Attrs.MISTRAL_REQUEST_SAFE_PROMPT,
}

# Cohere-specific parameters (note: 'p' and 'k' are Cohere's naming)
COHERE_PARAM_MAPPING = {
    "temperature": Attrs.GEN_AI_REQUEST_TEMPERATURE,
    "max_tokens": Attrs.GEN_AI_REQUEST_MAX_TOKENS,
    "p": Attrs.GEN_AI_REQUEST_TOP_P,  # Cohere uses 'p' for top_p
    "k": Attrs.GEN_AI_REQUEST_TOP_K,  # Cohere uses 'k' for top_k
    "frequency_penalty": Attrs.GEN_AI_REQUEST_FREQUENCY_PENALTY,
    "presence_penalty": Attrs.GEN_AI_REQUEST_PRESENCE_PENALTY,
}

# Google GenAI parameters (extracted from config object)
GOOGLE_PARAM_MAPPING = {
    "temperature": Attrs.GEN_AI_REQUEST_TEMPERATURE,
    "max_output_tokens": Attrs.GEN_AI_REQUEST_MAX_TOKENS,
    "top_p": Attrs.GEN_AI_REQUEST_TOP_P,
    "top_k": Attrs.GEN_AI_REQUEST_TOP_K,
}


# =============================================================================
# Unified Parameter Extraction
# =============================================================================

def extract_params(kwargs: Dict[str, Any], param_mapping: Dict[str, str]) -> Dict[str, Any]:
    """
    Extract parameters using provider-specific mapping.

    Single code path for ALL providers, sync AND async.

    Args:
        kwargs: Request keyword arguments
        param_mapping: Mapping from kwarg keys to OTEL attribute keys

    Returns:
        Dict of extracted attributes
    """
    attrs = {}
    for kwarg_key, attr_key in param_mapping.items():
        value = kwargs.get(kwarg_key)
        if value is not None:
            # Handle special serialization cases
            if attr_key == Attrs.ANTHROPIC_REQUEST_METADATA and isinstance(value, dict):
                attrs[attr_key] = json.dumps(value)
            elif attr_key == Attrs.ANTHROPIC_REQUEST_STOP_SEQUENCES and isinstance(value, list):
                attrs[attr_key] = value
            else:
                attrs[attr_key] = value
    return attrs


def extract_stop_sequences(kwargs: Dict[str, Any], key: str = "stop") -> Optional[List[str]]:
    """Extract stop sequences, normalizing to list."""
    stop = kwargs.get(key)
    if stop is None:
        return None
    return stop if isinstance(stop, list) else [stop]


def extract_system_messages_openai(messages: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
    """
    Separate system and non-system messages (OpenAI/Mistral format).

    Returns:
        Tuple of (non_system_messages, system_messages)
    """
    system_msgs = []
    non_system_msgs = []

    for msg in messages:
        if isinstance(msg, dict):
            if msg.get("role") == "system":
                system_msgs.append(msg)
            else:
                non_system_msgs.append(msg)
        elif hasattr(msg, "role"):
            msg_dict = {"role": msg.role, "content": getattr(msg, "content", "")}
            if msg.role == "system":
                system_msgs.append(msg_dict)
            else:
                non_system_msgs.append(msg_dict)

    return non_system_msgs, system_msgs


# =============================================================================
# Base Attributes Builders
# =============================================================================

def build_base_attrs(
    provider: str,
    model: str,
    stream: bool = False,
) -> Dict[str, Any]:
    """Build base span attributes common to all providers."""
    return {
        Attrs.BROKLE_SPAN_TYPE: SpanType.GENERATION,
        Attrs.GEN_AI_PROVIDER_NAME: provider,
        Attrs.GEN_AI_OPERATION_NAME: OperationType.CHAT,
        Attrs.GEN_AI_REQUEST_MODEL: model,
        Attrs.BROKLE_STREAMING: stream,
    }


def add_messages_attrs(
    attrs: Dict[str, Any],
    input_messages: Optional[List[Dict]] = None,
    system_messages: Optional[List[Dict]] = None,
) -> None:
    """Add message attributes to attrs dict."""
    if input_messages:
        attrs[Attrs.GEN_AI_INPUT_MESSAGES] = serialize_messages(input_messages)
    if system_messages:
        attrs[Attrs.GEN_AI_SYSTEM_INSTRUCTIONS] = serialize_messages(system_messages)


# =============================================================================
# Response Extractors
# =============================================================================

def extract_openai_response(span: Any, response: Any, latency_ms: float) -> None:
    """
    Extract OpenAI/Azure response attributes and set on span.

    Works for both sync and async paths - single code path.
    """
    # Response ID
    if hasattr(response, "id") and response.id:
        span.set_attribute(Attrs.GEN_AI_RESPONSE_ID, response.id)

    # Response model
    if hasattr(response, "model") and response.model:
        span.set_attribute(Attrs.GEN_AI_RESPONSE_MODEL, response.model)

    # System fingerprint
    if hasattr(response, "system_fingerprint") and response.system_fingerprint:
        span.set_attribute(
            Attrs.OPENAI_RESPONSE_SYSTEM_FINGERPRINT,
            response.system_fingerprint,
        )

    # Output messages and finish reasons
    if hasattr(response, "choices") and response.choices:
        output_messages = []
        finish_reasons = []

        for choice in response.choices:
            if hasattr(choice, "message") and choice.message:
                msg_dict = {
                    "role": getattr(choice.message, "role", None),
                    "content": getattr(choice.message, "content", None),
                }

                # Tool calls
                if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                    tool_calls = []
                    for tc in choice.message.tool_calls:
                        tool_calls.append({
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        })
                    msg_dict["tool_calls"] = tool_calls

                # Refusal
                if hasattr(choice.message, "refusal") and choice.message.refusal:
                    msg_dict["refusal"] = choice.message.refusal

                output_messages.append(msg_dict)

            if hasattr(choice, "finish_reason") and choice.finish_reason:
                finish_reasons.append(choice.finish_reason)

        if output_messages:
            span.set_attribute(Attrs.GEN_AI_OUTPUT_MESSAGES, json.dumps(output_messages))
        if finish_reasons:
            span.set_attribute(Attrs.GEN_AI_RESPONSE_FINISH_REASONS, finish_reasons)

    # Token usage
    if hasattr(response, "usage") and response.usage:
        usage = response.usage
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)

        if input_tokens:
            span.set_attribute(Attrs.GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
        if output_tokens:
            span.set_attribute(Attrs.GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)

        total_tokens = calculate_total_tokens(input_tokens, output_tokens)
        if total_tokens:
            span.set_attribute(Attrs.BROKLE_USAGE_TOTAL_TOKENS, total_tokens)

    # Latency
    span.set_attribute(Attrs.BROKLE_USAGE_LATENCY_MS, latency_ms)


def extract_anthropic_response(span: Any, response: Any, latency_ms: float) -> None:
    """
    Extract Anthropic response attributes and set on span.

    Works for both sync and async paths - single code path.
    """
    # Response ID
    if hasattr(response, "id"):
        span.set_attribute(Attrs.GEN_AI_RESPONSE_ID, response.id)

    # Response model
    if hasattr(response, "model"):
        span.set_attribute(Attrs.GEN_AI_RESPONSE_MODEL, response.model)

    # Stop reason
    if hasattr(response, "stop_reason") and response.stop_reason:
        span.set_attribute(
            Attrs.GEN_AI_RESPONSE_FINISH_REASONS, [response.stop_reason]
        )
        span.set_attribute(
            Attrs.ANTHROPIC_RESPONSE_STOP_REASON, response.stop_reason
        )

    # Stop sequence (if custom stop sequence was hit)
    if hasattr(response, "stop_sequence") and response.stop_sequence:
        span.set_attribute(
            Attrs.ANTHROPIC_RESPONSE_STOP_SEQUENCE, response.stop_sequence
        )

    # Content blocks → output messages
    if hasattr(response, "content") and response.content:
        output_messages = []

        for content_block in response.content:
            if hasattr(content_block, "type"):
                if content_block.type == "text":
                    output_messages.append({
                        "role": "assistant",
                        "content": content_block.text,
                    })
                elif content_block.type == "tool_use":
                    output_messages.append({
                        "role": "assistant",
                        "tool_calls": [{
                            "id": content_block.id,
                            "type": "function",
                            "function": {
                                "name": content_block.name,
                                "arguments": json.dumps(content_block.input),
                            },
                        }],
                    })

        if output_messages:
            span.set_attribute(
                Attrs.GEN_AI_OUTPUT_MESSAGES, json.dumps(output_messages)
            )

    # Token usage
    if hasattr(response, "usage") and response.usage:
        usage = response.usage
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)

        if input_tokens:
            span.set_attribute(Attrs.GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
        if output_tokens:
            span.set_attribute(Attrs.GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)

        total_tokens = calculate_total_tokens(input_tokens, output_tokens)
        if total_tokens:
            span.set_attribute(Attrs.BROKLE_USAGE_TOTAL_TOKENS, total_tokens)

    # Latency
    span.set_attribute(Attrs.BROKLE_USAGE_LATENCY_MS, latency_ms)


def extract_mistral_response(span: Any, response: Any, latency_ms: float) -> None:
    """
    Extract Mistral response attributes and set on span.

    Works for both sync and async paths - single code path.
    """
    # Response ID
    if hasattr(response, "id"):
        span.set_attribute(Attrs.GEN_AI_RESPONSE_ID, response.id)

    # Response model
    if hasattr(response, "model"):
        span.set_attribute(Attrs.GEN_AI_RESPONSE_MODEL, response.model)

    # Choices → output messages and finish reasons
    if hasattr(response, "choices") and response.choices:
        output_messages = []
        finish_reasons = []

        for choice in response.choices:
            if hasattr(choice, "message"):
                msg = choice.message
                msg_dict = {
                    "role": getattr(msg, "role", "assistant"),
                    "content": getattr(msg, "content", ""),
                }

                # Tool calls
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_calls = []
                    for tc in msg.tool_calls:
                        tool_calls.append({
                            "id": getattr(tc, "id", ""),
                            "type": "function",
                            "function": {
                                "name": getattr(tc.function, "name", ""),
                                "arguments": getattr(tc.function, "arguments", ""),
                            },
                        })
                    msg_dict["tool_calls"] = tool_calls

                output_messages.append(msg_dict)

            if hasattr(choice, "finish_reason") and choice.finish_reason:
                reason = str(choice.finish_reason)
                finish_reasons.append(reason)
                span.set_attribute(Attrs.MISTRAL_RESPONSE_FINISH_REASON, reason)

        if output_messages:
            span.set_attribute(Attrs.GEN_AI_OUTPUT_MESSAGES, json.dumps(output_messages))
        if finish_reasons:
            span.set_attribute(Attrs.GEN_AI_RESPONSE_FINISH_REASONS, finish_reasons)

    # Token usage
    if hasattr(response, "usage") and response.usage:
        usage = response.usage
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)

        if input_tokens:
            span.set_attribute(Attrs.GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
        if output_tokens:
            span.set_attribute(Attrs.GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)

        total_tokens = calculate_total_tokens(input_tokens, output_tokens)
        if total_tokens:
            span.set_attribute(Attrs.BROKLE_USAGE_TOTAL_TOKENS, total_tokens)

    # Latency
    span.set_attribute(Attrs.BROKLE_USAGE_LATENCY_MS, latency_ms)


def extract_cohere_response(span: Any, response: Any, latency_ms: float) -> None:
    """
    Extract Cohere response attributes and set on span.

    Works for both sync and async paths - single code path.
    """
    # Generation ID
    if hasattr(response, "generation_id"):
        span.set_attribute(Attrs.GEN_AI_RESPONSE_ID, response.generation_id)

    # Response text → output message
    if hasattr(response, "text") and response.text:
        output_messages = [{"role": "assistant", "content": response.text}]
        span.set_attribute(Attrs.GEN_AI_OUTPUT_MESSAGES, json.dumps(output_messages))

    # Finish reason
    if hasattr(response, "finish_reason") and response.finish_reason:
        span.set_attribute(
            Attrs.GEN_AI_RESPONSE_FINISH_REASONS, [response.finish_reason]
        )

    # Citations
    if hasattr(response, "citations") and response.citations:
        citations = _serialize_cohere_citations(response.citations)
        span.set_attribute(Attrs.COHERE_RESPONSE_CITATIONS, json.dumps(citations))

    # Search results
    if hasattr(response, "search_results") and response.search_results:
        results = _serialize_cohere_search_results(response.search_results)
        span.set_attribute(Attrs.COHERE_RESPONSE_SEARCH_RESULTS, json.dumps(results))

    # Token usage (Cohere uses meta.billed_units)
    if hasattr(response, "meta") and response.meta:
        meta = response.meta
        if hasattr(meta, "billed_units"):
            units = meta.billed_units
            input_tokens = getattr(units, "input_tokens", None)
            output_tokens = getattr(units, "output_tokens", None)

            if input_tokens is not None:
                span.set_attribute(Attrs.GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
            if output_tokens is not None:
                span.set_attribute(Attrs.GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)

            total_tokens = calculate_total_tokens(input_tokens, output_tokens)
            if total_tokens:
                span.set_attribute(Attrs.BROKLE_USAGE_TOTAL_TOKENS, total_tokens)

    # Latency
    span.set_attribute(Attrs.BROKLE_USAGE_LATENCY_MS, latency_ms)


def extract_google_response(span: Any, response: Any, latency_ms: float) -> None:
    """
    Extract Google GenAI response attributes and set on span.

    Works for both sync and async paths - single code path.
    """
    try:
        # Candidates
        candidates = getattr(response, "candidates", None)
        if candidates and len(candidates) > 0:
            candidate = candidates[0]

            # Finish reason
            finish_reason = getattr(candidate, "finish_reason", None)
            if finish_reason:
                span.set_attribute(
                    Attrs.GEN_AI_RESPONSE_FINISH_REASONS,
                    [str(finish_reason)]
                )

            # Content parts → output message
            content = getattr(candidate, "content", None)
            if content and hasattr(content, "parts"):
                output_text = "".join(
                    part.text for part in content.parts
                    if hasattr(part, "text")
                )
                if output_text:
                    span.set_attribute(
                        Attrs.GEN_AI_OUTPUT_MESSAGES,
                        json.dumps([{"role": "assistant", "content": output_text}])
                    )

        # Usage metadata
        usage_metadata = getattr(response, "usage_metadata", None)
        if usage_metadata:
            prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0)
            completion_tokens = getattr(usage_metadata, "candidates_token_count", 0)
            total_tokens = getattr(usage_metadata, "total_token_count", 0)

            if prompt_tokens:
                span.set_attribute(Attrs.GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens)
            if completion_tokens:
                span.set_attribute(Attrs.GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens)
            if total_tokens:
                span.set_attribute(Attrs.BROKLE_USAGE_TOTAL_TOKENS, total_tokens)

        # Fallback: text property
        if not candidates and hasattr(response, "text") and response.text:
            span.set_attribute(
                Attrs.GEN_AI_OUTPUT_MESSAGES,
                json.dumps([{"role": "assistant", "content": response.text}])
            )

    except Exception:
        # Ignore extraction errors
        pass

    # Latency
    span.set_attribute(Attrs.BROKLE_USAGE_LATENCY_MS, latency_ms)


def extract_bedrock_response(span: Any, response: Dict[str, Any], latency_ms: float) -> None:
    """
    Extract Bedrock response attributes and set on span.

    Works for both sync and async paths - single code path.
    """
    # Output message
    if "output" in response and "message" in response["output"]:
        message = response["output"]["message"]
        output_messages = []

        if "content" in message:
            content_parts = []
            for content in message["content"]:
                if "text" in content:
                    content_parts.append(content["text"])
                elif "toolUse" in content:
                    tool = content["toolUse"]
                    output_messages.append({
                        "role": "assistant",
                        "tool_calls": [{
                            "id": tool.get("toolUseId", ""),
                            "type": "function",
                            "function": {
                                "name": tool.get("name", ""),
                                "arguments": json.dumps(tool.get("input", {})),
                            },
                        }],
                    })

            if content_parts:
                output_messages.insert(0, {
                    "role": "assistant",
                    "content": "".join(content_parts),
                })

        if output_messages:
            span.set_attribute(Attrs.GEN_AI_OUTPUT_MESSAGES, json.dumps(output_messages))

    # Stop reason
    if "stopReason" in response:
        span.set_attribute(
            Attrs.GEN_AI_RESPONSE_FINISH_REASONS, [response["stopReason"]]
        )
        span.set_attribute(Attrs.BEDROCK_RESPONSE_STOP_REASON, response["stopReason"])

    # Token usage
    if "usage" in response:
        usage = response["usage"]
        input_tokens = usage.get("inputTokens")
        output_tokens = usage.get("outputTokens")
        total_tokens = usage.get("totalTokens")

        if input_tokens:
            span.set_attribute(Attrs.GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
        if output_tokens:
            span.set_attribute(Attrs.GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)
        if total_tokens:
            span.set_attribute(Attrs.BROKLE_USAGE_TOTAL_TOKENS, total_tokens)
        elif input_tokens or output_tokens:
            total = calculate_total_tokens(input_tokens, output_tokens)
            if total:
                span.set_attribute(Attrs.BROKLE_USAGE_TOTAL_TOKENS, total)

    # Metrics
    if "metrics" in response:
        span.set_attribute(Attrs.BEDROCK_RESPONSE_METRICS, json.dumps(response["metrics"]))

    # Latency
    span.set_attribute(Attrs.BROKLE_USAGE_LATENCY_MS, latency_ms)


# =============================================================================
# Helper Serialization Functions
# =============================================================================

def _serialize_cohere_citations(citations) -> List[Dict[str, Any]]:
    """Serialize Cohere citations to JSON-compatible format."""
    result = []
    for cit in citations:
        if isinstance(cit, dict):
            result.append(cit)
        elif hasattr(cit, "start"):
            result.append({
                "start": cit.start,
                "end": getattr(cit, "end", None),
                "text": getattr(cit, "text", None),
                "document_ids": getattr(cit, "document_ids", []),
            })
    return result


def _serialize_cohere_search_results(search_results) -> List[Dict[str, Any]]:
    """Serialize Cohere search results to JSON-compatible format."""
    result = []
    for sr in search_results:
        if isinstance(sr, dict):
            result.append(sr)
        elif hasattr(sr, "document_ids"):
            result.append({
                "document_ids": sr.document_ids,
                "search_query": getattr(sr, "search_query", None),
            })
    return result


def serialize_cohere_connectors(connectors) -> List[Dict[str, Any]]:
    """Serialize Cohere connectors to JSON-compatible format."""
    result = []
    for conn in connectors:
        if isinstance(conn, dict):
            result.append(conn)
        elif hasattr(conn, "id"):
            result.append({
                "id": conn.id,
                "options": getattr(conn, "options", {}),
            })
    return result


def serialize_cohere_documents(documents) -> List[Dict[str, Any]]:
    """Serialize Cohere documents to JSON-compatible format."""
    result = []
    for doc in documents:
        if isinstance(doc, dict):
            result.append(doc)
        elif isinstance(doc, str):
            result.append({"text": doc})
        elif hasattr(doc, "text"):
            result.append({
                "id": getattr(doc, "id", None),
                "text": doc.text,
            })
    return result


def serialize_tools(tools) -> List[Dict[str, Any]]:
    """Serialize tools to JSON-compatible format (Mistral/OpenAI)."""
    result = []
    for tool in tools:
        if isinstance(tool, dict):
            result.append(tool)
        elif hasattr(tool, "model_dump"):
            result.append(tool.model_dump())
        elif hasattr(tool, "to_dict"):
            result.append(tool.to_dict())
        elif hasattr(tool, "__dict__"):
            result.append({
                k: v for k, v in tool.__dict__.items()
                if not k.startswith("_")
            })
    return result
