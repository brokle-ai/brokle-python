"""
Streaming accumulator for LLM responses.

Tracks time-to-first-token (TTFT), inter-token latency, and accumulates
streamed content for final span attributes.

Based on patterns from Langfuse SDK's completion_start_time tracking
and OpenLIT's streaming metrics implementation.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StreamingResult:
    """
    Result of streaming accumulation.

    Contains all metrics and accumulated content from a streaming response.

    Attributes:
        content: Accumulated response content
        ttft_ms: Time to first token in milliseconds
        total_duration_ms: Total streaming duration in milliseconds
        token_count: Estimated token count (based on chunks received)
        inter_token_latencies: List of inter-token latencies in ms
        avg_inter_token_latency_ms: Average inter-token latency
        chunk_count: Number of chunks received
        finish_reason: LLM finish reason (e.g., "stop", "length")
        model: Model name if extracted from stream
        usage: Token usage info if available from stream
    """

    content: str = ""
    ttft_ms: Optional[float] = None
    total_duration_ms: Optional[float] = None
    token_count: int = 0
    inter_token_latencies: List[float] = field(default_factory=list)
    avg_inter_token_latency_ms: Optional[float] = None
    chunk_count: int = 0
    finish_reason: Optional[str] = None
    model: Optional[str] = None
    usage: Optional[Dict[str, int]] = None

    def to_attributes(self) -> Dict[str, Any]:
        """
        Convert streaming result to span attributes.

        Returns:
            Dictionary of span attributes following OTEL GenAI conventions
        """
        import json

        from ..types import Attrs

        attrs = {}

        if self.ttft_ms is not None:
            attrs[Attrs.GEN_AI_RESPONSE_TTFT] = self.ttft_ms

        if self.avg_inter_token_latency_ms is not None:
            attrs[Attrs.GEN_AI_RESPONSE_ITL] = self.avg_inter_token_latency_ms

        if self.total_duration_ms is not None:
            attrs[Attrs.GEN_AI_RESPONSE_DURATION] = self.total_duration_ms

        if self.finish_reason:
            attrs[Attrs.GEN_AI_RESPONSE_FINISH_REASON] = self.finish_reason

        if self.model:
            attrs[Attrs.GEN_AI_RESPONSE_MODEL] = self.model

        if self.usage:
            if "prompt_tokens" in self.usage:
                attrs[Attrs.GEN_AI_USAGE_INPUT_TOKENS] = self.usage["prompt_tokens"]
            if "completion_tokens" in self.usage:
                attrs[Attrs.GEN_AI_USAGE_OUTPUT_TOKENS] = self.usage[
                    "completion_tokens"
                ]
            if "total_tokens" in self.usage:
                attrs[Attrs.BROKLE_USAGE_TOTAL_TOKENS] = self.usage["total_tokens"]

        if self.content:
            output_messages = [{"role": "assistant", "content": self.content}]
            attrs[Attrs.GEN_AI_OUTPUT_MESSAGES] = json.dumps(output_messages)

        return attrs


class StreamingAccumulator:
    """
    Accumulator for streaming LLM responses.

    Tracks timing metrics and accumulates content from streaming responses.
    Works with both OpenAI and Anthropic streaming formats.

    Example:
        >>> import time
        >>> accumulator = StreamingAccumulator(time.perf_counter())
        >>> for chunk in stream:
        ...     content = accumulator.on_chunk(chunk)
        ...     if content:
        ...         print(content, end="", flush=True)
        >>> result = accumulator.finalize()
        >>> span.set_attributes(result.to_attributes())
    """

    def __init__(
        self,
        start_time: float,
        content_extractor: Optional[Callable[[Any], Optional[str]]] = None,
        finish_reason_extractor: Optional[Callable[[Any], Optional[str]]] = None,
        model_extractor: Optional[Callable[[Any], Optional[str]]] = None,
        usage_extractor: Optional[Callable[[Any], Optional[Dict[str, int]]]] = None,
    ):
        """
        Initialize streaming accumulator.

        Args:
            start_time: Time when API call was initiated (from time.perf_counter())
            content_extractor: Custom function to extract content from chunks
            finish_reason_extractor: Custom function to extract finish reason
            model_extractor: Custom function to extract model name
            usage_extractor: Custom function to extract token usage
        """
        self._start_time: float = start_time
        self._first_token_time: Optional[float] = None
        self._last_chunk_time: Optional[float] = None
        self._chunks: List[str] = []
        self._inter_token_latencies: List[float] = []
        self._chunk_count: int = 0
        self._finish_reason: Optional[str] = None
        self._model: Optional[str] = None
        self._usage: Optional[Dict[str, int]] = None
        self._finalized: bool = False
        self._anthropic_input_tokens: Optional[int] = (
            None  # Track input tokens from message_start
        )

        # Custom extractors (fallback to auto-detection)
        self._content_extractor = content_extractor
        self._finish_reason_extractor = finish_reason_extractor
        self._model_extractor = model_extractor
        self._usage_extractor = usage_extractor

    def on_chunk(self, chunk: Any) -> Optional[str]:
        """
        Process a streaming chunk.

        Extracts content, tracks timing, and accumulates response.

        Args:
            chunk: Streaming chunk from LLM provider

        Returns:
            Extracted content string, or None if no content in chunk
        """
        if self._finalized:
            logger.warning("on_chunk called after finalize()")
            return None

        now = time.perf_counter()

        content = self._extract_content(chunk)

        if content and self._first_token_time is None:
            self._first_token_time = now

        if content and self._last_chunk_time is not None:
            latency_ms = (now - self._last_chunk_time) * 1000
            self._inter_token_latencies.append(latency_ms)

        if content:
            self._last_chunk_time = now
            self._chunks.append(content)

        self._chunk_count += 1

        if self._finish_reason is None:
            self._finish_reason = self._extract_finish_reason(chunk)
        if self._model is None:
            self._model = self._extract_model(chunk)

        extracted_usage = self._extract_usage(chunk)
        if extracted_usage:
            if (
                extracted_usage.get("prompt_tokens", 0) > 0
                and self._anthropic_input_tokens is None
            ):
                self._anthropic_input_tokens = extracted_usage["prompt_tokens"]

            if (
                self._anthropic_input_tokens
                and extracted_usage.get("prompt_tokens", 0) == 0
            ):
                extracted_usage["prompt_tokens"] = self._anthropic_input_tokens
                extracted_usage["total_tokens"] = (
                    self._anthropic_input_tokens + extracted_usage["completion_tokens"]
                )

            self._usage = extracted_usage

        return content

    def finalize(self) -> StreamingResult:
        """
        Finalize streaming and compute metrics.

        Returns:
            StreamingResult with accumulated content and metrics
        """
        if self._finalized:
            logger.warning("finalize() called multiple times")

        self._finalized = True
        now = time.perf_counter()

        ttft_ms = None
        if self._start_time is not None and self._first_token_time is not None:
            ttft_ms = (self._first_token_time - self._start_time) * 1000

        total_duration_ms = None
        if self._start_time is not None:
            total_duration_ms = (now - self._start_time) * 1000

        avg_itl = None
        if self._inter_token_latencies:
            avg_itl = sum(self._inter_token_latencies) / len(
                self._inter_token_latencies
            )

        token_count = len(self._chunks)
        if self._usage and "completion_tokens" in self._usage:
            token_count = self._usage["completion_tokens"]

        return StreamingResult(
            content="".join(self._chunks),
            ttft_ms=ttft_ms,
            total_duration_ms=total_duration_ms,
            token_count=token_count,
            inter_token_latencies=self._inter_token_latencies.copy(),
            avg_inter_token_latency_ms=avg_itl,
            chunk_count=self._chunk_count,
            finish_reason=self._finish_reason,
            model=self._model,
            usage=self._usage,
        )

    def _extract_content(self, chunk: Any) -> Optional[str]:
        """Extract content from chunk using custom extractor or auto-detection."""
        if self._content_extractor:
            return self._content_extractor(chunk)

        if hasattr(chunk, "choices"):
            choices = chunk.choices
            if choices and len(choices) > 0:
                delta = getattr(choices[0], "delta", None)
                if delta:
                    return getattr(delta, "content", None)

        if hasattr(chunk, "delta"):
            delta = chunk.delta
            if hasattr(delta, "text"):
                return delta.text

        if hasattr(chunk, "type") and chunk.type == "content_block_delta":
            delta = getattr(chunk, "delta", None)
            if delta and hasattr(delta, "text"):
                return delta.text

        if isinstance(chunk, dict):
            choices = chunk.get("choices", [])
            if choices and len(choices) > 0:
                delta = choices[0].get("delta", {})
                return delta.get("content")

            delta = chunk.get("delta", {})
            return delta.get("text")

        return None

    def _extract_finish_reason(self, chunk: Any) -> Optional[str]:
        """Extract finish reason from chunk."""
        if self._finish_reason_extractor:
            return self._finish_reason_extractor(chunk)

        if hasattr(chunk, "choices"):
            choices = chunk.choices
            if choices and len(choices) > 0:
                return getattr(choices[0], "finish_reason", None)

        if hasattr(chunk, "type") and chunk.type == "message_stop":
            return "stop"

        if isinstance(chunk, dict):
            choices = chunk.get("choices", [])
            if choices and len(choices) > 0:
                return choices[0].get("finish_reason")

        return None

    def _extract_model(self, chunk: Any) -> Optional[str]:
        """Extract model name from chunk."""
        if self._model_extractor:
            return self._model_extractor(chunk)

        if hasattr(chunk, "model"):
            return chunk.model

        if isinstance(chunk, dict):
            return chunk.get("model")

        return None

    def _extract_usage(self, chunk: Any) -> Optional[Dict[str, int]]:
        """
        Extract token usage from chunk.

        Handles both OpenAI and Anthropic streaming formats:
        - OpenAI: usage in final chunk with prompt_tokens/completion_tokens/total_tokens
        - Anthropic: input_tokens in message_start, output_tokens in message_delta

        Returns normalized dict with prompt_tokens, completion_tokens, total_tokens.
        """
        if self._usage_extractor:
            return self._usage_extractor(chunk)

        if hasattr(chunk, "type") and chunk.type == "message_start":
            message = getattr(chunk, "message", None)
            if message and hasattr(message, "usage") and message.usage:
                usage = message.usage
                input_tokens = getattr(usage, "input_tokens", 0)
                output_tokens = getattr(usage, "output_tokens", 0)
                return {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                }

        if hasattr(chunk, "type") and chunk.type == "message_delta":
            usage = getattr(chunk, "usage", None)
            if usage and hasattr(usage, "output_tokens"):
                output_tokens = getattr(usage, "output_tokens", 0)
                return {
                    "prompt_tokens": 0,
                    "completion_tokens": output_tokens,
                    "total_tokens": output_tokens,
                }

        if hasattr(chunk, "usage") and chunk.usage is not None:
            usage = chunk.usage
            if hasattr(usage, "prompt_tokens"):
                return {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                }
            if hasattr(usage, "input_tokens"):
                input_tokens = getattr(usage, "input_tokens", 0)
                output_tokens = getattr(usage, "output_tokens", 0)
                return {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                }

        if isinstance(chunk, dict):
            if chunk.get("type") == "message_start":
                message = chunk.get("message", {})
                usage = message.get("usage", {})
                if usage:
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)
                    return {
                        "prompt_tokens": input_tokens,
                        "completion_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    }

            if chunk.get("type") == "message_delta":
                usage = chunk.get("usage", {})
                if usage:
                    output_tokens = usage.get("output_tokens", 0)
                    return {
                        "prompt_tokens": 0,
                        "completion_tokens": output_tokens,
                        "total_tokens": output_tokens,
                    }

            usage = chunk.get("usage")
            if usage:
                prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
                completion = usage.get(
                    "completion_tokens", usage.get("output_tokens", 0)
                )
                total = usage.get("total_tokens", prompt + completion)
                return {
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "total_tokens": total,
                }

        return None

    @property
    def content(self) -> str:
        """Get accumulated content so far."""
        return "".join(self._chunks)

    @property
    def ttft_ms(self) -> Optional[float]:
        """Get time to first token in milliseconds (None until first token received)."""
        if self._start_time is None or self._first_token_time is None:
            return None
        return (self._first_token_time - self._start_time) * 1000

    @property
    def is_started(self) -> bool:
        """Check if streaming has started."""
        return self._start_time is not None

    @property
    def is_finalized(self) -> bool:
        """Check if streaming has been finalized."""
        return self._finalized
