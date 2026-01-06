"""
Google GenAI SDK wrapper for automatic observability.

Supports the google-genai SDK (GA as of May 2025)
for accessing Gemini models.

Wraps GoogleGenAI client to automatically create OTEL spans with GenAI 1.28+ attributes.
Streaming responses are transparently instrumented with TTFT and ITL tracking.
"""

import json
import time
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional

from opentelemetry.trace import Status, StatusCode

from .._client import get_client
from ..streaming import StreamingAccumulator
from ..types import Attrs, LLMProvider, OperationType, SpanType
from ._common import add_prompt_attributes, extract_brokle_options

if TYPE_CHECKING:
    from google import genai


def wrap_google(client: "genai.Client") -> "genai.Client":
    """
    Wrap Google GenAI client for automatic observability.

    This function wraps the GoogleGenAI client's models namespace
    to automatically create OTEL spans with GenAI semantic attributes.

    Args:
        client: GoogleGenAI client instance from google-genai package

    Returns:
        Wrapped GoogleGenAI client (same instance with instrumented methods)

    Example:
        >>> from google import genai
        >>> from brokle import get_client
        >>> from brokle.wrappers import wrap_google
        >>>
        >>> # Initialize Brokle
        >>> brokle = get_client()
        >>>
        >>> # Create and wrap Google GenAI client
        >>> ai = genai.Client(api_key="...")
        >>> ai = wrap_google(ai)
        >>>
        >>> # All calls automatically tracked
        >>> response = ai.models.generate_content(
        ...     model="gemini-2.0-flash",
        ...     contents="Hello!",
        ... )
        >>> brokle.flush()
    """
    # Return unwrapped if SDK disabled
    brokle_client = get_client()
    if not brokle_client.config.enabled:
        return client

    # Validate client structure
    if not hasattr(client, "models"):
        raise ValueError(
            "Invalid GoogleGenAI client passed to wrap_google. "
            "The 'google-genai' package is required. "
            "Install with: pip install google-genai"
        )

    # Wrap the models namespace
    original_models = client.models
    client.models = _WrappedModelsNamespace(original_models)

    return client


class _WrappedModelsNamespace:
    """Wrapper for the models namespace with tracing."""

    def __init__(self, models):
        self._models = models

    def __getattr__(self, name):
        attr = getattr(self._models, name)

        if name == "generate_content" and callable(attr):
            return _traced_generate_content(attr)
        elif name == "generate_content_stream" and callable(attr):
            return _traced_generate_content_stream(attr)
        elif name == "embed_content" and callable(attr):
            return _traced_embed_content(attr)

        return attr


def _traced_generate_content(original_fn):
    """Traced generate_content for new SDK."""

    def wrapper(*args, **kwargs):
        # Extract brokle options
        kwargs, brokle_opts = extract_brokle_options(kwargs)

        brokle_client = get_client()

        # Extract model name
        model_name = kwargs.get("model", "gemini")

        # Extract contents
        contents = kwargs.get("contents", "")
        input_messages = _build_input_messages(contents)

        # Build attributes
        attrs = {
            Attrs.BROKLE_SPAN_TYPE: SpanType.GENERATION,
            Attrs.GEN_AI_PROVIDER_NAME: LLMProvider.GOOGLE,
            Attrs.GEN_AI_OPERATION_NAME: OperationType.CHAT,
            Attrs.GEN_AI_REQUEST_MODEL: model_name,
        }

        if input_messages:
            attrs[Attrs.GEN_AI_INPUT_MESSAGES] = json.dumps(input_messages)

        # Extract config parameters
        config = kwargs.get("config", {})
        if config:
            if hasattr(config, "temperature") and config.temperature is not None:
                attrs[Attrs.GEN_AI_REQUEST_TEMPERATURE] = config.temperature
            elif isinstance(config, dict) and "temperature" in config:
                attrs[Attrs.GEN_AI_REQUEST_TEMPERATURE] = config["temperature"]

            if hasattr(config, "max_output_tokens") and config.max_output_tokens is not None:
                attrs[Attrs.GEN_AI_REQUEST_MAX_TOKENS] = config.max_output_tokens
            elif isinstance(config, dict) and "max_output_tokens" in config:
                attrs[Attrs.GEN_AI_REQUEST_MAX_TOKENS] = config["max_output_tokens"]

            if hasattr(config, "top_p") and config.top_p is not None:
                attrs[Attrs.GEN_AI_REQUEST_TOP_P] = config.top_p
            elif isinstance(config, dict) and "top_p" in config:
                attrs[Attrs.GEN_AI_REQUEST_TOP_P] = config["top_p"]

        add_prompt_attributes(attrs, brokle_opts)

        span_name = f"{OperationType.CHAT} {model_name}"

        with brokle_client.start_as_current_span(span_name, attributes=attrs) as span:
            try:
                start_time = time.time()
                response = original_fn(*args, **kwargs)
                latency_ms = (time.time() - start_time) * 1000

                # Extract response attributes
                _extract_response_attributes(response, span)

                span.set_attribute(Attrs.BROKLE_USAGE_LATENCY_MS, latency_ms)
                span.set_status(Status(StatusCode.OK))

                return response

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


def _traced_generate_content_stream(original_fn):
    """Traced generate_content_stream for new SDK."""

    def wrapper(*args, **kwargs):
        # Extract brokle options
        kwargs, brokle_opts = extract_brokle_options(kwargs)

        brokle_client = get_client()

        # Extract model name
        model_name = kwargs.get("model", "gemini")

        # Extract contents
        contents = kwargs.get("contents", "")
        input_messages = _build_input_messages(contents)

        # Build attributes
        attrs = {
            Attrs.BROKLE_SPAN_TYPE: SpanType.GENERATION,
            Attrs.GEN_AI_PROVIDER_NAME: LLMProvider.GOOGLE,
            Attrs.GEN_AI_OPERATION_NAME: OperationType.CHAT,
            Attrs.GEN_AI_REQUEST_MODEL: model_name,
            Attrs.BROKLE_STREAMING: True,
        }

        if input_messages:
            attrs[Attrs.GEN_AI_INPUT_MESSAGES] = json.dumps(input_messages)

        add_prompt_attributes(attrs, brokle_opts)

        span_name = f"{OperationType.CHAT} {model_name}"

        tracer = brokle_client._tracer
        span = tracer.start_span(span_name, attributes=attrs)

        try:
            start_time = time.perf_counter()
            stream = original_fn(*args, **kwargs)
            accumulator = StreamingAccumulator(start_time)
            return _GoogleGenAIStreamWrapper(stream, span, accumulator)
        except BaseException as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            span.end()
            raise

    return wrapper


def _traced_embed_content(original_fn):
    """Traced embed_content for new SDK."""

    def wrapper(*args, **kwargs):
        brokle_client = get_client()

        # Extract model name
        model_name = kwargs.get("model", "embedding")

        # Extract content
        content = kwargs.get("content") or kwargs.get("contents")

        attrs = {
            Attrs.BROKLE_SPAN_TYPE: SpanType.EMBEDDING,
            Attrs.GEN_AI_PROVIDER_NAME: LLMProvider.GOOGLE,
            Attrs.GEN_AI_OPERATION_NAME: "embeddings",
            Attrs.GEN_AI_REQUEST_MODEL: model_name,
        }

        if content:
            if isinstance(content, str):
                attrs[Attrs.GEN_AI_INPUT_MESSAGES] = content
            else:
                attrs[Attrs.GEN_AI_INPUT_MESSAGES] = json.dumps(content)

        span_name = f"embedding {model_name}"

        with brokle_client.start_as_current_span(span_name, attributes=attrs) as span:
            try:
                start_time = time.time()
                response = original_fn(*args, **kwargs)
                latency_ms = (time.time() - start_time) * 1000

                span.set_attribute(Attrs.BROKLE_USAGE_LATENCY_MS, latency_ms)
                span.set_status(Status(StatusCode.OK))

                return response

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


class _GoogleGenAIStreamWrapper:
    """Wrapper for Google GenAI streaming responses (new SDK)."""

    def __init__(self, stream, span, accumulator):
        self._stream = stream
        self._span = span
        self._accumulator = accumulator
        self._content_parts = []
        self._finish_reason = None

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self._stream)
            self._accumulator.on_chunk_received()

            # Accumulate content from new SDK format
            if hasattr(chunk, "candidates") and chunk.candidates:
                for candidate in chunk.candidates:
                    if hasattr(candidate, "content") and candidate.content:
                        if hasattr(candidate.content, "parts"):
                            for part in candidate.content.parts:
                                if hasattr(part, "text"):
                                    self._content_parts.append(part.text)
                    if hasattr(candidate, "finish_reason"):
                        self._finish_reason = str(candidate.finish_reason)

            # Also check for text property directly (convenience accessor)
            if hasattr(chunk, "text") and chunk.text:
                # Only add if not already captured from candidates
                if not self._content_parts or chunk.text not in self._content_parts:
                    self._content_parts.append(chunk.text)

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

        # Set streaming metrics
        if self._accumulator.ttft_ms is not None:
            self._span.set_attribute(Attrs.GEN_AI_RESPONSE_TTFT, self._accumulator.ttft_ms)
        if self._accumulator.avg_itl_ms is not None:
            self._span.set_attribute(Attrs.GEN_AI_RESPONSE_ITL, self._accumulator.avg_itl_ms)
        if self._accumulator.duration_ms is not None:
            self._span.set_attribute(Attrs.BROKLE_USAGE_LATENCY_MS, self._accumulator.duration_ms)

        self._span.set_status(Status(StatusCode.OK))
        self._span.end()


def _build_input_messages(contents) -> List[Dict[str, Any]]:
    """Build input messages from Google GenAI content format (new SDK)."""
    messages = []

    if isinstance(contents, str):
        messages.append({"role": "user", "content": contents})
    elif isinstance(contents, list):
        for item in contents:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
            elif hasattr(item, "parts"):
                # Content object
                role = getattr(item, "role", "user")
                parts_text = []
                for part in item.parts:
                    if hasattr(part, "text"):
                        parts_text.append(part.text)
                if parts_text:
                    messages.append({"role": role, "content": "".join(parts_text)})
            elif isinstance(item, dict):
                role = item.get("role", "user")
                parts = item.get("parts", [])
                content_parts = []
                for part in parts:
                    if isinstance(part, str):
                        content_parts.append(part)
                    elif isinstance(part, dict) and "text" in part:
                        content_parts.append(part["text"])
                if content_parts:
                    messages.append({"role": role, "content": "".join(content_parts)})

    return messages


def _extract_response_attributes(response, span) -> None:
    """Extract attributes from generateContent response (new SDK format)."""
    try:
        # New SDK response structure
        candidates = getattr(response, "candidates", None)
        if candidates and len(candidates) > 0:
            candidate = candidates[0]
            finish_reason = getattr(candidate, "finish_reason", None)
            if finish_reason:
                span.set_attribute(
                    Attrs.GEN_AI_RESPONSE_FINISH_REASONS,
                    [str(finish_reason)]
                )

            # Extract text from content parts
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

        # Also check for text property directly (convenience accessor)
        if not candidates and hasattr(response, "text") and response.text:
            span.set_attribute(
                Attrs.GEN_AI_OUTPUT_MESSAGES,
                json.dumps([{"role": "assistant", "content": response.text}])
            )

    except Exception:
        # Ignore extraction errors
        pass
