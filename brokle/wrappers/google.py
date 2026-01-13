"""
Google GenAI SDK wrapper for automatic observability.

Supports the google-genai SDK (GA as of May 2025)
for accessing Gemini models.

Wraps GoogleGenAI client to automatically create OTEL spans with GenAI 1.28+ attributes.
Streaming responses are transparently instrumented with TTFT and ITL tracking.

Uses the unified factory pattern for consistent sync/async behavior.
"""

import json
import time
from typing import TYPE_CHECKING, Any, Dict, List

from opentelemetry.trace import Status, StatusCode

from .._client import get_client
from ..streaming import StreamingAccumulator
from ..types import Attrs, LLMProvider, OperationType, SpanType
from ._common import add_prompt_attributes, extract_brokle_options
from ._extractors import extract_google_response
from ._provider_config import build_google_attrs, google_span_name

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
        if not brokle_client.config.enabled:
            return original_fn(*args, **kwargs)

        # Extract model name
        model_name = kwargs.get("model", "gemini")

        # Build attributes using unified extractor
        attrs = build_google_attrs(kwargs, model_name=model_name)
        add_prompt_attributes(attrs, brokle_opts)

        span_name = google_span_name(kwargs, model_name=model_name)

        with brokle_client.start_as_current_span(span_name, attributes=attrs) as span:
            try:
                start_time = time.time()
                response = original_fn(*args, **kwargs)
                latency_ms = (time.time() - start_time) * 1000

                # Extract response attributes using unified extractor
                extract_google_response(span, response, latency_ms)
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
        if not brokle_client.config.enabled:
            return original_fn(*args, **kwargs)

        # Extract model name
        model_name = kwargs.get("model", "gemini")

        # Build attributes using unified extractor
        attrs = build_google_attrs(kwargs, model_name=model_name)
        attrs[Attrs.BROKLE_STREAMING] = True
        add_prompt_attributes(attrs, brokle_opts)

        span_name = google_span_name(kwargs, model_name=model_name)

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
        if not brokle_client.config.enabled:
            return original_fn(*args, **kwargs)

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
    """
    Wrapper for Google GenAI streaming responses.

    Handles Google's streaming format with candidates and usage_metadata.
    """

    def __init__(self, stream, span, accumulator):
        self._stream = stream
        self._span = span
        self._accumulator = accumulator
        self._content_parts = []
        self._finish_reason = None
        self._usage_metadata = None

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

            # Capture usage metadata (usually in last chunk)
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                self._usage_metadata = chunk.usage_metadata

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

        # Token usage (now captured from stream)
        if self._usage_metadata:
            prompt_tokens = getattr(self._usage_metadata, "prompt_token_count", 0)
            completion_tokens = getattr(self._usage_metadata, "candidates_token_count", 0)
            total_tokens = getattr(self._usage_metadata, "total_token_count", 0)

            if prompt_tokens:
                self._span.set_attribute(Attrs.GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens)
            if completion_tokens:
                self._span.set_attribute(Attrs.GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens)
            if total_tokens:
                self._span.set_attribute(Attrs.BROKLE_USAGE_TOTAL_TOKENS, total_tokens)

        # Set streaming metrics
        if self._accumulator.ttft_ms is not None:
            self._span.set_attribute(Attrs.GEN_AI_RESPONSE_TTFT, self._accumulator.ttft_ms)
        if self._accumulator.avg_itl_ms is not None:
            self._span.set_attribute(Attrs.GEN_AI_RESPONSE_ITL, self._accumulator.avg_itl_ms)
        if self._accumulator.duration_ms is not None:
            self._span.set_attribute(Attrs.BROKLE_USAGE_LATENCY_MS, self._accumulator.duration_ms)

        self._span.set_status(Status(StatusCode.OK))
        self._span.end()
