"""
Mistral AI SDK wrapper for automatic observability.

Wraps Mistral AI client to automatically create OTEL spans with GenAI 1.28+ attributes.
Streaming responses are transparently instrumented with TTFT and ITL tracking.

Uses the unified factory pattern for consistent sync/async behavior.
"""

import json
from typing import TYPE_CHECKING, TypeVar

from opentelemetry.trace import Status, StatusCode

from ..streaming import StreamingAccumulator
from ..types import Attrs
from ..utils.attributes import calculate_total_tokens
from .._client import get_client
from ._extractors import extract_mistral_response
from ._factory import create_wrapper
from ._provider_config import build_mistral_attrs, mistral_span_name

if TYPE_CHECKING:
    from mistralai import Mistral

C = TypeVar("C")


class _MistralStreamWrapper:
    """
    Wrapper for Mistral streaming responses.

    Handles Mistral-specific streaming format with data.choices.delta pattern.
    """

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
            input_tokens = getattr(self._usage, "prompt_tokens", None)
            output_tokens = getattr(self._usage, "completion_tokens", None)

            if input_tokens is not None:
                self._span.set_attribute(Attrs.GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
            if output_tokens is not None:
                self._span.set_attribute(Attrs.GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)

            total_tokens = calculate_total_tokens(input_tokens, output_tokens)
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

    # Create wrapper using unified factory with Mistral-specific stream wrapper
    wrapped = create_wrapper(
        original_method=original_chat_complete,
        build_attrs=build_mistral_attrs,
        extract_response=extract_mistral_response,
        get_span_name=mistral_span_name,
        get_model=lambda kw: kw.get("model", "mistral-large-latest"),
        is_stream=lambda kw: kw.get("stream", False),
        stream_wrapper_class=_MistralStreamWrapper,
    )

    client.chat.complete = wrapped
    return client
