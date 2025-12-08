"""
OpenAI SDK wrapper for automatic observability.

Wraps OpenAI client to automatically create OTEL spans with GenAI 1.28+ attributes.
Streaming responses are transparently instrumented with TTFT and ITL tracking.
"""

import json
import time
from typing import TYPE_CHECKING, Any, Optional

from opentelemetry.trace import Status, StatusCode

from ..client import get_client
from ..streaming import StreamingAccumulator
from ..streaming.wrappers import BrokleAsyncStreamWrapper, BrokleStreamWrapper
from ..types import Attrs, LLMProvider, OperationType, SpanType
from ..utils.attributes import (
    calculate_total_tokens,
    extract_model_parameters,
    extract_system_messages,
    serialize_messages,
)

if TYPE_CHECKING:
    import openai


def wrap_openai(client: "openai.OpenAI") -> "openai.OpenAI":
    """
    Wrap OpenAI client for automatic observability.

    This function wraps the OpenAI client's chat.completions.create method
    to automatically create OTEL spans with GenAI semantic attributes.

    Args:
        client: OpenAI client instance

    Returns:
        Wrapped OpenAI client (same instance with instrumented methods)

    Example:
        >>> import openai
        >>> from brokle import get_client, wrap_openai
        >>>
        >>> # Initialize Brokle
        >>> brokle = get_client()
        >>>
        >>> # Wrap OpenAI client
        >>> client = wrap_openai(openai.OpenAI(api_key="..."))
        >>>
        >>> # All calls automatically tracked
        >>> response = client.chat.completions.create(
        ...     model="gpt-4",
        ...     messages=[{"role": "user", "content": "Hello"}]
        ... )
        >>> brokle.flush()
    """
    # Store original method
    original_chat_create = client.chat.completions.create

    def wrapped_chat_create(*args, **kwargs):
        """Wrapped chat.completions.create with automatic tracing."""
        # Get Brokle client
        brokle_client = get_client()

        # Extract request parameters
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens")
        top_p = kwargs.get("top_p")
        frequency_penalty = kwargs.get("frequency_penalty")
        presence_penalty = kwargs.get("presence_penalty")
        n = kwargs.get("n", 1)
        stop = kwargs.get("stop")
        user = kwargs.get("user")
        stream = kwargs.get("stream", False)

        # Extract system messages
        non_system_msgs, system_msgs = extract_system_messages(messages)

        # Build OTEL GenAI attributes
        attrs = {
            Attrs.BROKLE_SPAN_TYPE: SpanType.GENERATION,
            Attrs.GEN_AI_PROVIDER_NAME: LLMProvider.OPENAI,
            Attrs.GEN_AI_OPERATION_NAME: OperationType.CHAT,
            Attrs.GEN_AI_REQUEST_MODEL: model,
            Attrs.BROKLE_STREAMING: stream,
        }

        # Add messages
        if non_system_msgs:
            attrs[Attrs.GEN_AI_INPUT_MESSAGES] = serialize_messages(non_system_msgs)
        if system_msgs:
            attrs[Attrs.GEN_AI_SYSTEM_INSTRUCTIONS] = serialize_messages(system_msgs)

        # Add model parameters
        if temperature is not None:
            attrs[Attrs.GEN_AI_REQUEST_TEMPERATURE] = temperature
        if max_tokens is not None:
            attrs[Attrs.GEN_AI_REQUEST_MAX_TOKENS] = max_tokens
        if top_p is not None:
            attrs[Attrs.GEN_AI_REQUEST_TOP_P] = top_p
        if frequency_penalty is not None:
            attrs[Attrs.GEN_AI_REQUEST_FREQUENCY_PENALTY] = frequency_penalty
        if presence_penalty is not None:
            attrs[Attrs.GEN_AI_REQUEST_PRESENCE_PENALTY] = presence_penalty
        if stop is not None:
            attrs[Attrs.GEN_AI_REQUEST_STOP_SEQUENCES] = (
                stop if isinstance(stop, list) else [stop]
            )
        if user is not None:
            attrs[Attrs.GEN_AI_REQUEST_USER] = user
            attrs[Attrs.USER_ID] = user  # Filterable

        # OpenAI-specific attributes
        if n is not None:
            attrs[Attrs.OPENAI_REQUEST_N] = n
        if kwargs.get("service_tier"):
            attrs[Attrs.OPENAI_REQUEST_SERVICE_TIER] = kwargs["service_tier"]
        if kwargs.get("seed"):
            attrs[Attrs.OPENAI_REQUEST_SEED] = kwargs["seed"]
        if kwargs.get("logprobs"):
            attrs[Attrs.OPENAI_REQUEST_LOGPROBS] = kwargs["logprobs"]
        if kwargs.get("top_logprobs"):
            attrs[Attrs.OPENAI_REQUEST_TOP_LOGPROBS] = kwargs["top_logprobs"]

        # Create span name following OTEL pattern: "{operation} {model}"
        span_name = f"{OperationType.CHAT} {model}"

        # Handle streaming vs non-streaming differently
        if stream:
            return _handle_streaming_response(
                brokle_client, original_chat_create, args, kwargs, span_name, attrs
            )
        else:
            return _handle_sync_response(
                brokle_client, original_chat_create, args, kwargs, span_name, attrs
            )

    def _handle_streaming_response(
        brokle_client, original_method, args, kwargs, span_name, attrs
    ):
        """Handle streaming response with transparent wrapper instrumentation."""
        # Start span manually using underlying tracer (will be ended by stream wrapper)
        # Access the private _tracer attribute to get non-context-manager span
        tracer = brokle_client._tracer
        span = tracer.start_span(span_name, attributes=attrs)

        try:
            # Record start time BEFORE API call
            start_time = time.perf_counter()

            # Make API call
            response = original_method(*args, **kwargs)

            # Create accumulator with start time
            accumulator = StreamingAccumulator(start_time)
            wrapped_stream = BrokleStreamWrapper(response, span, accumulator)

            return wrapped_stream

        except BaseException as e:
            # Record error and end span (catches all exceptions including KeyboardInterrupt)
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
                # Record start time for latency
                start_time = time.time()

                # Make actual API call
                response = original_method(*args, **kwargs)

                # Calculate latency
                latency_ms = (time.time() - start_time) * 1000

                # Extract response metadata
                if hasattr(response, "id"):
                    span.set_attribute(Attrs.GEN_AI_RESPONSE_ID, response.id)
                if hasattr(response, "model"):
                    span.set_attribute(Attrs.GEN_AI_RESPONSE_MODEL, response.model)
                if (
                    hasattr(response, "system_fingerprint")
                    and response.system_fingerprint
                ):
                    span.set_attribute(
                        Attrs.OPENAI_RESPONSE_SYSTEM_FINGERPRINT,
                        response.system_fingerprint,
                    )

                # Extract output messages
                if hasattr(response, "choices") and len(response.choices) > 0:
                    output_messages = []
                    finish_reasons = []

                    for choice in response.choices:
                        if hasattr(choice, "message"):
                            msg_dict = {
                                "role": choice.message.role,
                                "content": choice.message.content,
                            }

                            # Add tool calls if present
                            if (
                                hasattr(choice.message, "tool_calls")
                                and choice.message.tool_calls
                            ):
                                tool_calls = []
                                for tc in choice.message.tool_calls:
                                    tool_calls.append(
                                        {
                                            "id": tc.id,
                                            "type": tc.type,
                                            "function": {
                                                "name": tc.function.name,
                                                "arguments": tc.function.arguments,
                                            },
                                        }
                                    )
                                msg_dict["tool_calls"] = tool_calls

                            # Add refusal if present (GPT-4 moderation)
                            if (
                                hasattr(choice.message, "refusal")
                                and choice.message.refusal
                            ):
                                msg_dict["refusal"] = choice.message.refusal

                            output_messages.append(msg_dict)

                        # Track finish reason
                        if hasattr(choice, "finish_reason"):
                            finish_reasons.append(choice.finish_reason)

                    # Set output messages
                    if output_messages:
                        span.set_attribute(
                            Attrs.GEN_AI_OUTPUT_MESSAGES, json.dumps(output_messages)
                        )

                    # Set finish reasons
                    if finish_reasons:
                        span.set_attribute(
                            Attrs.GEN_AI_RESPONSE_FINISH_REASONS, finish_reasons
                        )

                # Extract usage statistics
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

                    # Calculate total tokens (Brokle custom attribute)
                    total_tokens = calculate_total_tokens(
                        (
                            usage.prompt_tokens
                            if hasattr(usage, "prompt_tokens")
                            else None
                        ),
                        (
                            usage.completion_tokens
                            if hasattr(usage, "completion_tokens")
                            else None
                        ),
                    )
                    if total_tokens:
                        span.set_attribute(
                            Attrs.BROKLE_USAGE_TOTAL_TOKENS, total_tokens
                        )

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
    client.chat.completions.create = wrapped_chat_create

    return client


def wrap_openai_async(client: "openai.AsyncOpenAI") -> "openai.AsyncOpenAI":
    """
    Wrap AsyncOpenAI client for automatic observability.

    Similar to wrap_openai but for async client.

    Args:
        client: AsyncOpenAI client instance

    Returns:
        Wrapped AsyncOpenAI client

    Example:
        >>> import openai
        >>> from brokle import get_client, wrap_openai_async
        >>>
        >>> brokle = get_client()
        >>> client = wrap_openai_async(openai.AsyncOpenAI(api_key="..."))
        >>>
        >>> # Async calls automatically tracked
        >>> response = await client.chat.completions.create(...)
    """
    # Store original method
    original_chat_create = client.chat.completions.create

    async def wrapped_chat_create(*args, **kwargs):
        """Wrapped async chat.completions.create with automatic tracing."""
        # Get Brokle client
        brokle_client = get_client()

        # Extract request parameters (same as sync version)
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        stream = kwargs.get("stream", False)

        # Extract system messages
        non_system_msgs, system_msgs = extract_system_messages(messages)

        # Build OTEL GenAI attributes (same as sync)
        attrs = {
            Attrs.BROKLE_SPAN_TYPE: SpanType.GENERATION,
            Attrs.GEN_AI_PROVIDER_NAME: LLMProvider.OPENAI,
            Attrs.GEN_AI_OPERATION_NAME: OperationType.CHAT,
            Attrs.GEN_AI_REQUEST_MODEL: model,
            Attrs.BROKLE_STREAMING: stream,
        }

        # Add messages
        if non_system_msgs:
            attrs[Attrs.GEN_AI_INPUT_MESSAGES] = serialize_messages(non_system_msgs)
        if system_msgs:
            attrs[Attrs.GEN_AI_SYSTEM_INSTRUCTIONS] = serialize_messages(system_msgs)

        # Add model parameters
        model_params = extract_model_parameters(kwargs)
        attrs.update(model_params)

        # Create span
        span_name = f"{OperationType.CHAT} {model}"

        # Handle streaming vs non-streaming
        if stream:
            return await _handle_async_streaming_response(
                brokle_client, original_chat_create, args, kwargs, span_name, attrs
            )
        else:
            return await _handle_async_response(
                brokle_client, original_chat_create, args, kwargs, span_name, attrs
            )

    async def _handle_async_streaming_response(
        brokle_client, original_method, args, kwargs, span_name, attrs
    ):
        """Handle async streaming response with transparent wrapper instrumentation."""
        # Start span manually using underlying tracer (will be ended by stream wrapper)
        tracer = brokle_client._tracer
        span = tracer.start_span(span_name, attributes=attrs)

        try:
            # Record start time BEFORE API call
            start_time = time.perf_counter()

            # Make async API call
            response = await original_method(*args, **kwargs)

            # Create accumulator with start time
            accumulator = StreamingAccumulator(start_time)
            wrapped_stream = BrokleAsyncStreamWrapper(response, span, accumulator)

            return wrapped_stream

        except BaseException as e:
            # Record error and end span (catches all exceptions including CancelledError)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            span.end()
            raise

    async def _handle_async_response(
        brokle_client, original_method, args, kwargs, span_name, attrs
    ):
        """Handle async non-streaming response with standard span lifecycle."""
        with brokle_client.start_as_current_span(span_name, attributes=attrs) as span:
            try:
                start_time = time.time()

                # Make async API call
                response = await original_method(*args, **kwargs)

                # Calculate latency
                latency_ms = (time.time() - start_time) * 1000

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
    client.chat.completions.create = wrapped_chat_create

    return client
