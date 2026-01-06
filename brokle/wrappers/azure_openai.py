"""
Azure OpenAI SDK wrapper for automatic observability.

Wraps Azure OpenAI client to automatically create OTEL spans with GenAI 1.28+ attributes.
Extends the OpenAI wrapper pattern with Azure-specific attributes.
"""

import json
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

from opentelemetry.trace import Status, StatusCode

from .._client import get_client
from ..streaming import StreamingAccumulator
from ..streaming.wrappers import BrokleAsyncStreamWrapper, BrokleStreamWrapper
from ..types import Attrs, LLMProvider, OperationType, SpanType
from ..utils.attributes import (
    calculate_total_tokens,
    extract_model_parameters,
    extract_system_messages,
    serialize_messages,
)
from ._common import add_prompt_attributes, extract_brokle_options, record_openai_response

if TYPE_CHECKING:
    from openai import AzureOpenAI, AsyncAzureOpenAI


def wrap_azure_openai(client: "AzureOpenAI") -> "AzureOpenAI":
    """
    Wrap Azure OpenAI client for automatic observability.

    This function wraps the Azure OpenAI client's chat.completions.create method
    to automatically create OTEL spans with GenAI semantic attributes.

    Args:
        client: AzureOpenAI client instance

    Returns:
        Wrapped AzureOpenAI client (same instance with instrumented methods)

    Example:
        >>> from openai import AzureOpenAI
        >>> from brokle import get_client, wrap_azure_openai
        >>>
        >>> # Initialize Brokle
        >>> brokle = get_client()
        >>>
        >>> # Wrap Azure OpenAI client
        >>> client = wrap_azure_openai(AzureOpenAI(
        ...     azure_endpoint="https://YOUR_RESOURCE.openai.azure.com",
        ...     api_key="...",
        ...     api_version="2024-02-15-preview"
        ... ))
        >>>
        >>> # All calls automatically tracked
        >>> response = client.chat.completions.create(
        ...     model="gpt-4",  # Your deployment name
        ...     messages=[{"role": "user", "content": "Hello"}]
        ... )
        >>> brokle.flush()
    """
    # Return unwrapped if SDK disabled
    brokle_client = get_client()
    if not brokle_client.config.enabled:
        return client

    # Extract Azure-specific metadata from client
    azure_endpoint = getattr(client, "_azure_endpoint", None)
    api_version = getattr(client, "_api_version", None)

    original_chat_create = client.chat.completions.create

    def wrapped_chat_create(*args, **kwargs):
        """Wrapped chat.completions.create with automatic tracing."""
        # Extract brokle_options before processing kwargs
        kwargs, brokle_opts = extract_brokle_options(kwargs)

        brokle_client = get_client()
        model = kwargs.get("model", "unknown")  # In Azure, this is deployment name
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

        non_system_msgs, system_msgs = extract_system_messages(messages)
        attrs = {
            Attrs.BROKLE_SPAN_TYPE: SpanType.GENERATION,
            Attrs.GEN_AI_PROVIDER_NAME: LLMProvider.AZURE_OPENAI,
            Attrs.GEN_AI_OPERATION_NAME: OperationType.CHAT,
            Attrs.GEN_AI_REQUEST_MODEL: model,
            Attrs.BROKLE_STREAMING: stream,
            Attrs.AZURE_OPENAI_DEPLOYMENT_NAME: model,
        }

        # Add Azure-specific attributes
        if azure_endpoint:
            # Extract resource name from endpoint
            resource_name = azure_endpoint.replace("https://", "").split(".")[0]
            attrs[Attrs.AZURE_OPENAI_RESOURCE_NAME] = resource_name
        if api_version:
            attrs[Attrs.AZURE_OPENAI_API_VERSION] = api_version

        if non_system_msgs:
            attrs[Attrs.GEN_AI_INPUT_MESSAGES] = serialize_messages(non_system_msgs)
        if system_msgs:
            attrs[Attrs.GEN_AI_SYSTEM_INSTRUCTIONS] = serialize_messages(system_msgs)

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
            attrs[Attrs.USER_ID] = user

        if n is not None:
            attrs[Attrs.OPENAI_REQUEST_N] = n
        if kwargs.get("seed"):
            attrs[Attrs.OPENAI_REQUEST_SEED] = kwargs["seed"]
        if kwargs.get("logprobs"):
            attrs[Attrs.OPENAI_REQUEST_LOGPROBS] = kwargs["logprobs"]
        if kwargs.get("top_logprobs"):
            attrs[Attrs.OPENAI_REQUEST_TOP_LOGPROBS] = kwargs["top_logprobs"]

        add_prompt_attributes(attrs, brokle_opts)

        span_name = f"{OperationType.CHAT} {model}"

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
        tracer = brokle_client._tracer
        span = tracer.start_span(span_name, attributes=attrs)

        try:
            start_time = time.perf_counter()
            response = original_method(*args, **kwargs)
            accumulator = StreamingAccumulator(start_time)
            return BrokleStreamWrapper(response, span, accumulator)
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
                record_openai_response(span, response, latency_ms)
                span.set_status(Status(StatusCode.OK))
                return response

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    client.chat.completions.create = wrapped_chat_create

    return client


def wrap_azure_openai_async(
    client: "AsyncAzureOpenAI",
) -> "AsyncAzureOpenAI":
    """
    Wrap AsyncAzureOpenAI client for automatic observability.

    Similar to wrap_azure_openai but for async client.

    Args:
        client: AsyncAzureOpenAI client instance

    Returns:
        Wrapped AsyncAzureOpenAI client

    Example:
        >>> from openai import AsyncAzureOpenAI
        >>> from brokle import get_client, wrap_azure_openai_async
        >>>
        >>> brokle = get_client()
        >>> client = wrap_azure_openai_async(AsyncAzureOpenAI(
        ...     azure_endpoint="https://YOUR_RESOURCE.openai.azure.com",
        ...     api_key="...",
        ...     api_version="2024-02-15-preview"
        ... ))
        >>>
        >>> # Async calls automatically tracked
        >>> response = await client.chat.completions.create(...)
    """
    # Return unwrapped if SDK disabled
    brokle_client = get_client()
    if not brokle_client.config.enabled:
        return client

    # Extract Azure-specific metadata from client
    azure_endpoint = getattr(client, "_azure_endpoint", None)
    api_version = getattr(client, "_api_version", None)

    original_chat_create = client.chat.completions.create

    async def wrapped_chat_create(*args, **kwargs):
        """Wrapped async chat.completions.create with automatic tracing."""
        # Extract brokle_options before processing kwargs
        kwargs, brokle_opts = extract_brokle_options(kwargs)

        brokle_client = get_client()
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        stream = kwargs.get("stream", False)

        non_system_msgs, system_msgs = extract_system_messages(messages)
        attrs = {
            Attrs.BROKLE_SPAN_TYPE: SpanType.GENERATION,
            Attrs.GEN_AI_PROVIDER_NAME: LLMProvider.AZURE_OPENAI,
            Attrs.GEN_AI_OPERATION_NAME: OperationType.CHAT,
            Attrs.GEN_AI_REQUEST_MODEL: model,
            Attrs.BROKLE_STREAMING: stream,
            Attrs.AZURE_OPENAI_DEPLOYMENT_NAME: model,
        }

        if azure_endpoint:
            resource_name = azure_endpoint.replace("https://", "").split(".")[0]
            attrs[Attrs.AZURE_OPENAI_RESOURCE_NAME] = resource_name
        if api_version:
            attrs[Attrs.AZURE_OPENAI_API_VERSION] = api_version

        if non_system_msgs:
            attrs[Attrs.GEN_AI_INPUT_MESSAGES] = serialize_messages(non_system_msgs)
        if system_msgs:
            attrs[Attrs.GEN_AI_SYSTEM_INSTRUCTIONS] = serialize_messages(system_msgs)

        model_params = extract_model_parameters(kwargs)
        attrs.update(model_params)

        add_prompt_attributes(attrs, brokle_opts)

        span_name = f"{OperationType.CHAT} {model}"

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
        tracer = brokle_client._tracer
        span = tracer.start_span(span_name, attributes=attrs)

        try:
            start_time = time.perf_counter()
            response = await original_method(*args, **kwargs)
            accumulator = StreamingAccumulator(start_time)
            return BrokleAsyncStreamWrapper(response, span, accumulator)
        except BaseException as e:
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
                response = await original_method(*args, **kwargs)
                latency_ms = (time.time() - start_time) * 1000
                record_openai_response(span, response, latency_ms)
                span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    client.chat.completions.create = wrapped_chat_create

    return client
