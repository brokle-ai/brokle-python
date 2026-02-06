"""
AWS Bedrock SDK wrapper for automatic observability.

Wraps AWS Bedrock Runtime client to automatically create OTEL spans with GenAI 1.28+ attributes.
Supports the Converse API for cross-model compatibility.

Uses the unified factory pattern for consistent sync/async behavior.
"""

import json
import threading
import time
from typing import TYPE_CHECKING

from opentelemetry.trace import Status, StatusCode

from .._client import get_client
from ..streaming import StreamingAccumulator
from ..types import Attrs
from ._common import add_prompt_attributes, extract_brokle_options
from ._extractors import extract_bedrock_response
from ._factory import create_wrapper
from ._provider_config import build_bedrock_attrs, bedrock_span_name

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient


class _BedrockStreamWrapper:
    """
    Wrapper for Bedrock streaming responses.

    Handles Bedrock-specific streaming events like contentBlockDelta,
    messageStop, and metadata.
    """

    def __init__(self, response, span, accumulator):
        self._response = response
        self._span = span
        self._accumulator = accumulator
        self._content_parts = []
        self._stop_reason = None
        self._usage = {}
        self._stream = response.get("stream", iter([]))
        self._finalized = False
        self._lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            event = next(self._stream)
            self._accumulator.on_chunk_received()

            # Handle different event types
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"].get("delta", {})
                if "text" in delta:
                    self._content_parts.append(delta["text"])
            elif "messageStop" in event:
                self._stop_reason = event["messageStop"].get("stopReason")
            elif "metadata" in event:
                metadata = event["metadata"]
                if "usage" in metadata:
                    self._usage = metadata["usage"]

            return event

        except StopIteration:
            self._finalize()
            raise

    def _finalize(self):
        """Finalize span with accumulated data (thread-safe)."""
        with self._lock:
            if self._finalized:
                return
            self._finalized = True

        if self._content_parts:
            output_messages = [{
                "role": "assistant",
                "content": "".join(self._content_parts),
            }]
            self._span.set_attribute(
                Attrs.GEN_AI_OUTPUT_MESSAGES, json.dumps(output_messages)
            )

        if self._stop_reason:
            self._span.set_attribute(
                Attrs.GEN_AI_RESPONSE_FINISH_REASONS, [self._stop_reason]
            )
            self._span.set_attribute(
                Attrs.BEDROCK_RESPONSE_STOP_REASON, self._stop_reason
            )

        # Token usage (now properly captured from metadata event)
        if self._usage:
            input_tokens = self._usage.get("inputTokens")
            output_tokens = self._usage.get("outputTokens")

            if input_tokens:
                self._span.set_attribute(
                    Attrs.GEN_AI_USAGE_INPUT_TOKENS, input_tokens
                )
            if output_tokens:
                self._span.set_attribute(
                    Attrs.GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens
                )
            # Calculate and set total tokens
            if input_tokens or output_tokens:
                total = (input_tokens or 0) + (output_tokens or 0)
                if total:
                    self._span.set_attribute(Attrs.BROKLE_USAGE_TOTAL_TOKENS, total)

        # Set streaming metrics
        if self._accumulator.ttft_ms is not None:
            self._span.set_attribute(Attrs.GEN_AI_RESPONSE_TTFT, self._accumulator.ttft_ms)
        if self._accumulator.avg_itl_ms is not None:
            self._span.set_attribute(Attrs.GEN_AI_RESPONSE_ITL, self._accumulator.avg_itl_ms)
        if self._accumulator.duration_ms is not None:
            self._span.set_attribute(Attrs.BROKLE_USAGE_LATENCY_MS, self._accumulator.duration_ms)

        self._span.set_status(Status(StatusCode.OK))
        self._span.end()


def wrap_bedrock(client: "BedrockRuntimeClient") -> "BedrockRuntimeClient":
    """
    Wrap AWS Bedrock Runtime client for automatic observability.

    This function wraps the Bedrock Runtime client's converse method
    to automatically create OTEL spans with GenAI semantic attributes.

    Args:
        client: BedrockRuntimeClient instance

    Returns:
        Wrapped BedrockRuntimeClient (same instance with instrumented methods)

    Example:
        >>> import boto3
        >>> from brokle import get_client, wrap_bedrock
        >>>
        >>> # Initialize Brokle
        >>> brokle = get_client()
        >>>
        >>> # Wrap Bedrock client
        >>> bedrock = wrap_bedrock(boto3.client("bedrock-runtime"))
        >>>
        >>> # All calls automatically tracked
        >>> response = bedrock.converse(
        ...     modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        ...     messages=[{"role": "user", "content": [{"text": "Hello!"}]}]
        ... )
        >>> brokle.flush()
    """
    original_converse = client.converse

    # Create wrapper using unified factory
    wrapped = create_wrapper(
        original_method=original_converse,
        build_attrs=build_bedrock_attrs,
        extract_response=extract_bedrock_response,
        get_span_name=bedrock_span_name,
        get_model=lambda kw: kw.get("modelId", "unknown"),
        is_stream=lambda kw: False,  # Bedrock uses separate converse_stream method
    )

    client.converse = wrapped

    # Also wrap converse_stream if available
    if hasattr(client, "converse_stream"):
        original_converse_stream = client.converse_stream

        def wrapped_converse_stream(*args, **kwargs):
            """Wrapped converse_stream with automatic tracing."""
            kwargs, brokle_opts = extract_brokle_options(kwargs)

            brokle_client = get_client()
            if not brokle_client.config.enabled:
                return original_converse_stream(*args, **kwargs)

            # Build attributes using unified extractor
            attrs = build_bedrock_attrs(kwargs)
            attrs[Attrs.BROKLE_STREAMING] = True
            add_prompt_attributes(attrs, brokle_opts)

            span_name = bedrock_span_name(kwargs)

            tracer = brokle_client._tracer
            span = tracer.start_span(span_name, attributes=attrs)

            try:
                start_time = time.perf_counter()
                response = original_converse_stream(*args, **kwargs)
                accumulator = StreamingAccumulator(start_time)
                return _BedrockStreamWrapper(response, span, accumulator)
            except BaseException as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                span.end()
                raise

        client.converse_stream = wrapped_converse_stream

    return client
