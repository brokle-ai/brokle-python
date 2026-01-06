"""
Mistral AI SDK wrapper for automatic observability.

Wraps Mistral AI client to automatically create OTEL spans with GenAI 1.28+ attributes.
Streaming responses are transparently instrumented with TTFT and ITL tracking.
"""

import json
import time
from typing import TYPE_CHECKING, Any, Dict

from opentelemetry.trace import Status, StatusCode

from .._client import get_client
from ..streaming import StreamingAccumulator
from ..types import Attrs, LLMProvider, OperationType, SpanType
from ..utils.attributes import calculate_total_tokens, serialize_messages
from ._common import add_prompt_attributes, extract_brokle_options

if TYPE_CHECKING:
    from mistralai import Mistral


def wrap_mistral(client: "Mistral") -> "Mistral":
    """
    Wrap Mistral AI client for automatic observability.

    This function wraps the Mistral client's chat.complete method
    to automatically create OTEL spans with GenAI semantic attributes.

    Args:
        client: Mistral client instance

    Returns:
        Wrapped Mistral client (same instance with instrumented methods)

    Example:
        >>> from mistralai import Mistral
        >>> from brokle import get_client, wrap_mistral
        >>>
        >>> # Initialize Brokle
        >>> brokle = get_client()
        >>>
        >>> # Wrap Mistral client
        >>> client = wrap_mistral(Mistral(api_key="..."))
        >>>
        >>> # All calls automatically tracked
        >>> response = client.chat.complete(
        ...     model="mistral-large-latest",
        ...     messages=[{"role": "user", "content": "Hello"}]
        ... )
        >>> brokle.flush()
    """
    # Return unwrapped if SDK disabled
    brokle_client = get_client()
    if not brokle_client.config.enabled:
        return client

    original_chat_complete = client.chat.complete

    def wrapped_chat_complete(*args, **kwargs):
        """Wrapped chat.complete with automatic tracing."""
        # Extract brokle_options before processing kwargs
        kwargs, brokle_opts = extract_brokle_options(kwargs)

        brokle_client = get_client()

        model = kwargs.get("model", "mistral-large-latest")
        messages = kwargs.get("messages", [])
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens")
        top_p = kwargs.get("top_p")
        random_seed = kwargs.get("random_seed")
        safe_prompt = kwargs.get("safe_prompt")
        stop = kwargs.get("stop")
        stream = kwargs.get("stream", False)
        tool_choice = kwargs.get("tool_choice")
        tools = kwargs.get("tools")

        # Separate system messages
        system_msgs = []
        non_system_msgs = []
        for msg in messages:
            if isinstance(msg, dict):
                if msg.get("role") == "system":
                    system_msgs.append(msg)
                else:
                    non_system_msgs.append(msg)
            elif hasattr(msg, "role"):
                if msg.role == "system":
                    system_msgs.append({"role": msg.role, "content": getattr(msg, "content", "")})
                else:
                    non_system_msgs.append({"role": msg.role, "content": getattr(msg, "content", "")})

        attrs = {
            Attrs.BROKLE_SPAN_TYPE: SpanType.GENERATION,
            Attrs.GEN_AI_PROVIDER_NAME: LLMProvider.MISTRAL,
            Attrs.GEN_AI_OPERATION_NAME: OperationType.CHAT,
            Attrs.GEN_AI_REQUEST_MODEL: model,
            Attrs.BROKLE_STREAMING: stream,
        }

        if non_system_msgs:
            attrs[Attrs.GEN_AI_INPUT_MESSAGES] = serialize_messages(non_system_msgs)
        if system_msgs:
            attrs[Attrs.GEN_AI_SYSTEM_INSTRUCTIONS] = json.dumps(system_msgs)

        if temperature is not None:
            attrs[Attrs.GEN_AI_REQUEST_TEMPERATURE] = temperature
        if max_tokens is not None:
            attrs[Attrs.GEN_AI_REQUEST_MAX_TOKENS] = max_tokens
        if top_p is not None:
            attrs[Attrs.GEN_AI_REQUEST_TOP_P] = top_p
        if stop is not None:
            attrs[Attrs.GEN_AI_REQUEST_STOP_SEQUENCES] = (
                stop if isinstance(stop, list) else [stop]
            )

        # Mistral-specific attributes
        if random_seed is not None:
            attrs[Attrs.MISTRAL_REQUEST_RANDOM_SEED] = random_seed
        if safe_prompt is not None:
            attrs[Attrs.MISTRAL_REQUEST_SAFE_PROMPT] = safe_prompt
        if tool_choice is not None:
            attrs[Attrs.MISTRAL_REQUEST_TOOL_CHOICE] = (
                tool_choice if isinstance(tool_choice, str) else json.dumps(tool_choice)
            )
        if tools is not None:
            attrs[Attrs.OPENAI_REQUEST_TOOLS] = json.dumps(_serialize_tools(tools))

        add_prompt_attributes(attrs, brokle_opts)

        span_name = f"{OperationType.CHAT} {model}"

        if stream:
            return _handle_streaming_response(
                brokle_client, original_chat_complete, args, kwargs, span_name, attrs
            )
        else:
            return _handle_sync_response(
                brokle_client, original_chat_complete, args, kwargs, span_name, attrs
            )

    def _handle_streaming_response(
        brokle_client, original_method, args, kwargs, span_name, attrs
    ):
        """Handle streaming response with transparent wrapper instrumentation."""
        tracer = brokle_client._tracer
        span = tracer.start_span(span_name, attributes=attrs)

        try:
            start_time = time.perf_counter()
            response = original_method(*args, **kwargs)
            accumulator = StreamingAccumulator(start_time)
            return _MistralStreamWrapper(response, span, accumulator)
        except BaseException as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            span.end()
            raise

    def _handle_sync_response(
        brokle_client, original_method, args, kwargs, span_name, attrs
    ):
        """Handle non-streaming response with standard span lifecycle."""
        with brokle_client.start_as_current_span(span_name, attributes=attrs) as span:
            try:
                start_time = time.time()
                response = original_method(*args, **kwargs)
                latency_ms = (time.time() - start_time) * 1000

                # Extract response metadata
                if hasattr(response, "id"):
                    span.set_attribute(Attrs.GEN_AI_RESPONSE_ID, response.id)
                if hasattr(response, "model"):
                    span.set_attribute(Attrs.GEN_AI_RESPONSE_MODEL, response.model)

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

                            # Handle tool calls
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

                        if hasattr(choice, "finish_reason"):
                            finish_reasons.append(str(choice.finish_reason))
                            span.set_attribute(
                                Attrs.MISTRAL_RESPONSE_FINISH_REASON,
                                str(choice.finish_reason),
                            )

                    if output_messages:
                        span.set_attribute(
                            Attrs.GEN_AI_OUTPUT_MESSAGES, json.dumps(output_messages)
                        )
                    if finish_reasons:
                        span.set_attribute(
                            Attrs.GEN_AI_RESPONSE_FINISH_REASONS, finish_reasons
                        )

                # Token usage
                if hasattr(response, "usage") and response.usage:
                    usage = response.usage
                    if hasattr(usage, "prompt_tokens") and usage.prompt_tokens:
                        span.set_attribute(
                            Attrs.GEN_AI_USAGE_INPUT_TOKENS, usage.prompt_tokens
                        )
                    if hasattr(usage, "completion_tokens") and usage.completion_tokens:
                        span.set_attribute(
                            Attrs.GEN_AI_USAGE_OUTPUT_TOKENS, usage.completion_tokens
                        )

                    total_tokens = calculate_total_tokens(
                        getattr(usage, "prompt_tokens", None),
                        getattr(usage, "completion_tokens", None),
                    )
                    if total_tokens:
                        span.set_attribute(
                            Attrs.BROKLE_USAGE_TOTAL_TOKENS, total_tokens
                        )

                span.set_attribute(Attrs.BROKLE_USAGE_LATENCY_MS, latency_ms)
                span.set_status(Status(StatusCode.OK))

                return response

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    client.chat.complete = wrapped_chat_complete

    return client


class _MistralStreamWrapper:
    """Wrapper for Mistral streaming responses."""

    def __init__(self, stream, span, accumulator):
        self._stream = stream
        self._span = span
        self._accumulator = accumulator
        self._content_parts = []
        self._finish_reason = None
        self._usage = None

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self._stream)
            self._accumulator.on_chunk_received()

            # Accumulate content
            if hasattr(chunk, "data") and chunk.data:
                data = chunk.data
                if hasattr(data, "choices") and data.choices:
                    for choice in data.choices:
                        if hasattr(choice, "delta") and choice.delta:
                            delta = choice.delta
                            if hasattr(delta, "content") and delta.content:
                                self._content_parts.append(delta.content)
                        if hasattr(choice, "finish_reason") and choice.finish_reason:
                            self._finish_reason = str(choice.finish_reason)

                if hasattr(data, "usage") and data.usage:
                    self._usage = data.usage

            return chunk

        except StopIteration:
            self._finalize()
            raise

    def _finalize(self):
        """Finalize span with accumulated data."""
        if self._content_parts:
            output_messages = [{
                "role": "assistant",
                "content": "".join(self._content_parts),
            }]
            self._span.set_attribute(
                Attrs.GEN_AI_OUTPUT_MESSAGES, json.dumps(output_messages)
            )

        if self._finish_reason:
            self._span.set_attribute(
                Attrs.GEN_AI_RESPONSE_FINISH_REASONS, [self._finish_reason]
            )
            self._span.set_attribute(
                Attrs.MISTRAL_RESPONSE_FINISH_REASON, self._finish_reason
            )

        if self._usage:
            if hasattr(self._usage, "prompt_tokens"):
                self._span.set_attribute(
                    Attrs.GEN_AI_USAGE_INPUT_TOKENS, self._usage.prompt_tokens
                )
            if hasattr(self._usage, "completion_tokens"):
                self._span.set_attribute(
                    Attrs.GEN_AI_USAGE_OUTPUT_TOKENS, self._usage.completion_tokens
                )

        # Set streaming metrics
        if self._accumulator.ttft_ms is not None:
            self._span.set_attribute(Attrs.GEN_AI_RESPONSE_TTFT, self._accumulator.ttft_ms)
        if self._accumulator.avg_itl_ms is not None:
            self._span.set_attribute(Attrs.GEN_AI_RESPONSE_ITL, self._accumulator.avg_itl_ms)
        if self._accumulator.duration_ms is not None:
            self._span.set_attribute(Attrs.BROKLE_USAGE_LATENCY_MS, self._accumulator.duration_ms)

        self._span.set_status(Status(StatusCode.OK))
        self._span.end()


def _serialize_tools(tools) -> list:
    """Serialize tools to JSON-compatible format."""
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
