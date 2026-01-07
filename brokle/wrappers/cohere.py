"""
Cohere SDK wrapper for automatic observability.

Wraps Cohere client to automatically create OTEL spans with GenAI 1.28+ attributes.
Streaming responses are transparently instrumented with TTFT and ITL tracking.

Uses the unified factory pattern for consistent sync/async behavior.
"""

import json
from typing import TYPE_CHECKING, Any, Dict, List, TypeVar, Union

from opentelemetry.trace import Status, StatusCode

from ..streaming import StreamingAccumulator
from ..types import Attrs
from .._client import get_client
from ._extractors import extract_cohere_response, serialize_cohere_connectors, serialize_cohere_documents
from ._factory import create_wrapper
from ._provider_config import build_cohere_attrs, cohere_span_name

if TYPE_CHECKING:
    import cohere

C = TypeVar("C")


class _CohereStreamWrapper:
    """
    Wrapper for Cohere streaming responses.

    Handles Cohere-specific streaming events like text-generation,
    citation-generation, and stream-end.
    """

    def __init__(self, stream, span, accumulator):
        self._stream = stream
        self._span = span
        self._accumulator = accumulator
        self._content_parts = []
        self._finish_reason = None
        self._citations = []
        self._generation_id = None
        self._input_tokens = None
        self._output_tokens = None

    def __iter__(self):
        return self

    def __next__(self):
        try:
            event = next(self._stream)
            self._accumulator.on_chunk_received()

            # Handle different event types
            event_type = getattr(event, "event_type", None)

            if event_type == "text-generation":
                if hasattr(event, "text"):
                    self._content_parts.append(event.text)
            elif event_type == "stream-end":
                if hasattr(event, "finish_reason"):
                    self._finish_reason = event.finish_reason
                if hasattr(event, "response"):
                    resp = event.response
                    if hasattr(resp, "generation_id"):
                        self._generation_id = resp.generation_id
                    if hasattr(resp, "citations"):
                        self._citations = resp.citations
                    # Extract token usage from final response
                    if hasattr(resp, "meta") and resp.meta:
                        meta = resp.meta
                        if hasattr(meta, "billed_units"):
                            units = meta.billed_units
                            self._input_tokens = getattr(units, "input_tokens", None)
                            self._output_tokens = getattr(units, "output_tokens", None)
            elif event_type == "citation-generation":
                if hasattr(event, "citations"):
                    self._citations.extend(event.citations)

            return event

        except StopIteration:
            self._finalize()
            raise

    def _finalize(self):
        """Finalize span with accumulated data."""
        if self._generation_id:
            self._span.set_attribute(Attrs.GEN_AI_RESPONSE_ID, self._generation_id)

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

        if self._citations:
            self._span.set_attribute(
                Attrs.COHERE_RESPONSE_CITATIONS,
                json.dumps(_serialize_citations(self._citations)),
            )

        # Token usage (now captured from stream-end event)
        if self._input_tokens is not None:
            self._span.set_attribute(Attrs.GEN_AI_USAGE_INPUT_TOKENS, self._input_tokens)
        if self._output_tokens is not None:
            self._span.set_attribute(Attrs.GEN_AI_USAGE_OUTPUT_TOKENS, self._output_tokens)
        if self._input_tokens is not None and self._output_tokens is not None:
            self._span.set_attribute(
                Attrs.BROKLE_USAGE_TOTAL_TOKENS,
                self._input_tokens + self._output_tokens
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


def _serialize_citations(citations) -> List[Dict[str, Any]]:
    """Serialize citations to JSON-compatible format."""
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


def wrap_cohere(
    client: "cohere.Client",
) -> "cohere.Client":
    """
    Wrap Cohere client for automatic observability.

    This function wraps the Cohere client's chat method
    to automatically create OTEL spans with GenAI semantic attributes.

    Args:
        client: Cohere client instance

    Returns:
        Wrapped Cohere client (same instance with instrumented methods)

    Example:
        >>> import cohere
        >>> from brokle import get_client, wrap_cohere
        >>>
        >>> # Initialize Brokle
        >>> brokle = get_client()
        >>>
        >>> # Wrap Cohere client
        >>> client = wrap_cohere(cohere.Client(api_key="..."))
        >>>
        >>> # All calls automatically tracked
        >>> response = client.chat(
        ...     model="command-r-plus",
        ...     message="Hello!"
        ... )
        >>> brokle.flush()
    """
    # Return unwrapped if SDK disabled
    brokle_client = get_client()
    if not brokle_client.config.enabled:
        return client

    original_chat = client.chat

    # Create wrapper using unified factory with Cohere-specific stream wrapper
    wrapped = create_wrapper(
        original_method=original_chat,
        build_attrs=build_cohere_attrs,
        extract_response=extract_cohere_response,
        get_span_name=cohere_span_name,
        get_model=lambda kw: kw.get("model", "command"),
        is_stream=lambda kw: kw.get("stream", False),
        stream_wrapper_class=_CohereStreamWrapper,
    )

    client.chat = wrapped
    return client
