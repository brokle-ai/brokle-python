"""
Cohere SDK wrapper for automatic observability.

Wraps Cohere client to automatically create OTEL spans with GenAI 1.28+ attributes.
Streaming responses are transparently instrumented with TTFT and ITL tracking.
"""

import json
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from opentelemetry.trace import Status, StatusCode

from .._client import get_client
from ..streaming import StreamingAccumulator
from ..types import Attrs, LLMProvider, OperationType, SpanType
from ..utils.attributes import calculate_total_tokens, serialize_messages
from ._common import add_prompt_attributes, extract_brokle_options

if TYPE_CHECKING:
    import cohere


def wrap_cohere(client: "cohere.Client") -> "cohere.Client":
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

    def wrapped_chat(*args, **kwargs):
        """Wrapped chat with automatic tracing."""
        # Extract brokle_options before processing kwargs
        kwargs, brokle_opts = extract_brokle_options(kwargs)

        brokle_client = get_client()

        model = kwargs.get("model", "command-r-plus")
        message = kwargs.get("message", "")
        chat_history = kwargs.get("chat_history", [])
        preamble = kwargs.get("preamble")
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens")
        p = kwargs.get("p")  # Cohere uses 'p' for top_p
        k = kwargs.get("k")  # Cohere uses 'k' for top_k
        stop_sequences = kwargs.get("stop_sequences")
        frequency_penalty = kwargs.get("frequency_penalty")
        presence_penalty = kwargs.get("presence_penalty")
        stream = kwargs.get("stream", False)
        connectors = kwargs.get("connectors")
        documents = kwargs.get("documents")
        search_queries_only = kwargs.get("search_queries_only")
        citation_quality = kwargs.get("citation_quality")

        # Build input messages from chat history and current message
        input_messages = []
        for hist in chat_history:
            if isinstance(hist, dict):
                input_messages.append({
                    "role": hist.get("role", "user"),
                    "content": hist.get("message", ""),
                })
            elif hasattr(hist, "role") and hasattr(hist, "message"):
                input_messages.append({
                    "role": hist.role,
                    "content": hist.message,
                })
        # Add current message
        input_messages.append({"role": "user", "content": message})

        attrs = {
            Attrs.BROKLE_SPAN_TYPE: SpanType.GENERATION,
            Attrs.GEN_AI_PROVIDER_NAME: LLMProvider.COHERE,
            Attrs.GEN_AI_OPERATION_NAME: OperationType.CHAT,
            Attrs.GEN_AI_REQUEST_MODEL: model,
            Attrs.BROKLE_STREAMING: stream,
        }

        if input_messages:
            attrs[Attrs.GEN_AI_INPUT_MESSAGES] = json.dumps(input_messages)
        if preamble:
            attrs[Attrs.GEN_AI_SYSTEM_INSTRUCTIONS] = json.dumps([
                {"role": "system", "content": preamble}
            ])
            attrs[Attrs.COHERE_REQUEST_PREAMBLE] = preamble

        if temperature is not None:
            attrs[Attrs.GEN_AI_REQUEST_TEMPERATURE] = temperature
        if max_tokens is not None:
            attrs[Attrs.GEN_AI_REQUEST_MAX_TOKENS] = max_tokens
        if p is not None:
            attrs[Attrs.GEN_AI_REQUEST_TOP_P] = p
        if k is not None:
            attrs[Attrs.GEN_AI_REQUEST_TOP_K] = k
        if stop_sequences is not None:
            attrs[Attrs.GEN_AI_REQUEST_STOP_SEQUENCES] = stop_sequences
        if frequency_penalty is not None:
            attrs[Attrs.GEN_AI_REQUEST_FREQUENCY_PENALTY] = frequency_penalty
        if presence_penalty is not None:
            attrs[Attrs.GEN_AI_REQUEST_PRESENCE_PENALTY] = presence_penalty

        # Cohere-specific attributes
        if connectors is not None:
            attrs[Attrs.COHERE_REQUEST_CONNECTORS] = json.dumps(
                _serialize_connectors(connectors)
            )
        if documents is not None:
            attrs[Attrs.COHERE_REQUEST_DOCUMENTS] = json.dumps(
                _serialize_documents(documents)
            )
        if search_queries_only is not None:
            attrs[Attrs.COHERE_REQUEST_SEARCH_QUERIES_ONLY] = search_queries_only
        if citation_quality is not None:
            attrs[Attrs.COHERE_REQUEST_CITATION_QUALITY] = citation_quality

        add_prompt_attributes(attrs, brokle_opts)

        span_name = f"{OperationType.CHAT} {model}"

        if stream:
            return _handle_streaming_response(
                brokle_client, original_chat, args, kwargs, span_name, attrs
            )
        else:
            return _handle_sync_response(
                brokle_client, original_chat, args, kwargs, span_name, attrs
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
            return _CohereStreamWrapper(response, span, accumulator)
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
                if hasattr(response, "generation_id"):
                    span.set_attribute(Attrs.GEN_AI_RESPONSE_ID, response.generation_id)

                if hasattr(response, "text") and response.text:
                    output_messages = [{
                        "role": "assistant",
                        "content": response.text,
                    }]
                    span.set_attribute(
                        Attrs.GEN_AI_OUTPUT_MESSAGES, json.dumps(output_messages)
                    )

                if hasattr(response, "finish_reason") and response.finish_reason:
                    span.set_attribute(
                        Attrs.GEN_AI_RESPONSE_FINISH_REASONS, [response.finish_reason]
                    )

                # Citations
                if hasattr(response, "citations") and response.citations:
                    span.set_attribute(
                        Attrs.COHERE_RESPONSE_CITATIONS,
                        json.dumps(_serialize_citations(response.citations)),
                    )

                # Search results
                if hasattr(response, "search_results") and response.search_results:
                    span.set_attribute(
                        Attrs.COHERE_RESPONSE_SEARCH_RESULTS,
                        json.dumps(_serialize_search_results(response.search_results)),
                    )

                # Token usage
                if hasattr(response, "meta") and response.meta:
                    meta = response.meta
                    if hasattr(meta, "billed_units"):
                        units = meta.billed_units
                        input_tokens = getattr(units, "input_tokens", None)
                        output_tokens = getattr(units, "output_tokens", None)

                        if input_tokens is not None:
                            span.set_attribute(
                                Attrs.GEN_AI_USAGE_INPUT_TOKENS, input_tokens
                            )
                        if output_tokens is not None:
                            span.set_attribute(
                                Attrs.GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens
                            )

                        total_tokens = calculate_total_tokens(input_tokens, output_tokens)
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

    client.chat = wrapped_chat

    return client


class _CohereStreamWrapper:
    """Wrapper for Cohere streaming responses."""

    def __init__(self, stream, span, accumulator):
        self._stream = stream
        self._span = span
        self._accumulator = accumulator
        self._content_parts = []
        self._finish_reason = None
        self._citations = []
        self._generation_id = None

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

        # Set streaming metrics
        if self._accumulator.ttft_ms is not None:
            self._span.set_attribute(Attrs.GEN_AI_RESPONSE_TTFT, self._accumulator.ttft_ms)
        if self._accumulator.avg_itl_ms is not None:
            self._span.set_attribute(Attrs.GEN_AI_RESPONSE_ITL, self._accumulator.avg_itl_ms)
        if self._accumulator.duration_ms is not None:
            self._span.set_attribute(Attrs.BROKLE_USAGE_LATENCY_MS, self._accumulator.duration_ms)

        self._span.set_status(Status(StatusCode.OK))
        self._span.end()


def _serialize_connectors(connectors) -> List[Dict[str, Any]]:
    """Serialize connectors to JSON-compatible format."""
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


def _serialize_documents(documents) -> List[Dict[str, Any]]:
    """Serialize documents to JSON-compatible format."""
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


def _serialize_search_results(search_results) -> List[Dict[str, Any]]:
    """Serialize search results to JSON-compatible format."""
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
