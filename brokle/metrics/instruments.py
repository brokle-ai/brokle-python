"""
Pre-defined GenAI metrics instruments.

Following the OpenLIT pattern, all metrics are created at initialization
rather than on-demand. This provides:
- Predictable memory usage
- Consistent metric cardinality
- Better performance (no allocation on hot path)

Example:
    >>> from brokle.metrics import create_genai_metrics
    >>> metrics = create_genai_metrics(meter)
    >>> metrics.record_tokens(input_tokens=100, output_tokens=50, model="gpt-4")
    >>> metrics.record_duration(duration_ms=1500, model="gpt-4")
"""

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

from opentelemetry.metrics import Counter, Histogram, Meter

from .constants import MetricNames

if TYPE_CHECKING:
    from opentelemetry.sdk.metrics import Meter as SDKMeter

logger = logging.getLogger(__name__)


class GenAIMetrics:
    """
    Pre-defined GenAI metrics following OTEL semantic conventions.

    This class creates all metrics at initialization and provides
    convenient recording methods for common LLM operations.

    Attributes:
        token_usage: Histogram of token counts
        operation_duration: Histogram of operation durations (ms)
        time_to_first_token: Histogram of TTFT (ms)
        inter_token_latency: Histogram of inter-token latency (ms)
        request_count: Counter of total requests
        error_count: Counter of errors
        input_tokens: Counter of input tokens
        output_tokens: Counter of output tokens
    """

    def __init__(self, meter: Meter):
        """
        Initialize GenAI metrics instruments.

        Args:
            meter: OpenTelemetry Meter instance
        """
        self._meter = meter

        # ========== Histograms ==========
        # Token usage histogram (combined input + output)
        self.token_usage: Histogram = meter.create_histogram(
            name=MetricNames.TOKEN_USAGE,
            description="Number of tokens processed in GenAI operations",
            unit="{token}",
        )

        # Operation duration histogram
        self.operation_duration: Histogram = meter.create_histogram(
            name=MetricNames.OPERATION_DURATION,
            description="Duration of GenAI operations",
            unit="ms",
        )

        # Time to first token histogram (streaming)
        self.time_to_first_token: Histogram = meter.create_histogram(
            name=MetricNames.TIME_TO_FIRST_TOKEN,
            description="Time to first token in streaming responses",
            unit="ms",
        )

        # Inter-token latency histogram (streaming)
        self.inter_token_latency: Histogram = meter.create_histogram(
            name=MetricNames.INTER_TOKEN_LATENCY,
            description="Latency between tokens in streaming responses",
            unit="ms",
        )

        # ========== Counters ==========
        # Request counter
        self.request_count: Counter = meter.create_counter(
            name=MetricNames.REQUEST_COUNT,
            description="Total number of GenAI requests",
            unit="{request}",
        )

        # Error counter
        self.error_count: Counter = meter.create_counter(
            name=MetricNames.ERROR_COUNT,
            description="Total number of GenAI errors",
            unit="{error}",
        )

        # Input tokens counter (cumulative)
        self.input_tokens: Counter = meter.create_counter(
            name=MetricNames.INPUT_TOKENS,
            description="Total input tokens processed",
            unit="{token}",
        )

        # Output tokens counter (cumulative)
        self.output_tokens: Counter = meter.create_counter(
            name=MetricNames.OUTPUT_TOKENS,
            description="Total output tokens generated",
            unit="{token}",
        )

    def record_tokens(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        operation: Optional[str] = None,
        **extra_attributes,
    ) -> None:
        """
        Record token usage metrics.

        Records both histogram (for distribution) and counters (for cumulative).

        Args:
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens
            model: Model name (e.g., "gpt-4", "claude-3-opus")
            provider: Provider name (e.g., "openai", "anthropic")
            operation: Operation type (e.g., "chat", "embeddings")
            **extra_attributes: Additional metric attributes
        """
        attributes = self._build_attributes(
            model=model,
            provider=provider,
            operation=operation,
            **extra_attributes,
        )

        total_tokens = input_tokens + output_tokens

        # Record histogram (single data point with total)
        if total_tokens > 0:
            self.token_usage.record(total_tokens, attributes)

        # Record counters (cumulative)
        if input_tokens > 0:
            self.input_tokens.add(input_tokens, attributes)
        if output_tokens > 0:
            self.output_tokens.add(output_tokens, attributes)

    def record_duration(
        self,
        duration_ms: float,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        operation: Optional[str] = None,
        is_error: bool = False,
        **extra_attributes,
    ) -> None:
        """
        Record operation duration.

        Args:
            duration_ms: Duration in milliseconds
            model: Model name
            provider: Provider name
            operation: Operation type
            is_error: Whether the operation resulted in an error
            **extra_attributes: Additional metric attributes
        """
        attributes = self._build_attributes(
            model=model,
            provider=provider,
            operation=operation,
            **extra_attributes,
        )

        # Record duration histogram
        self.operation_duration.record(duration_ms, attributes)

        # Increment request counter
        self.request_count.add(1, attributes)

        # Increment error counter if applicable
        if is_error:
            self.error_count.add(1, attributes)

    def record_ttft(
        self,
        ttft_ms: float,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **extra_attributes,
    ) -> None:
        """
        Record time to first token (streaming).

        Args:
            ttft_ms: Time to first token in milliseconds
            model: Model name
            provider: Provider name
            **extra_attributes: Additional metric attributes
        """
        attributes = self._build_attributes(
            model=model,
            provider=provider,
            **extra_attributes,
        )

        self.time_to_first_token.record(ttft_ms, attributes)

    def record_inter_token_latency(
        self,
        latency_ms: float,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **extra_attributes,
    ) -> None:
        """
        Record inter-token latency (streaming).

        Args:
            latency_ms: Inter-token latency in milliseconds
            model: Model name
            provider: Provider name
            **extra_attributes: Additional metric attributes
        """
        attributes = self._build_attributes(
            model=model,
            provider=provider,
            **extra_attributes,
        )

        self.inter_token_latency.record(latency_ms, attributes)

    def record_completion(
        self,
        start_time: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        operation: str = "chat",
        is_error: bool = False,
        ttft_ms: Optional[float] = None,
        **extra_attributes,
    ) -> None:
        """
        Record all metrics for a completion operation.

        Convenience method that records tokens, duration, and optionally TTFT
        in a single call. Use this after each LLM completion.

        Args:
            start_time: Operation start time (from time.time())
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name
            provider: Provider name
            operation: Operation type
            is_error: Whether operation resulted in error
            ttft_ms: Time to first token (if streaming)
            **extra_attributes: Additional metric attributes

        Example:
            >>> start = time.time()
            >>> response = openai.chat.completions.create(...)
            >>> metrics.record_completion(
            ...     start_time=start,
            ...     input_tokens=response.usage.prompt_tokens,
            ...     output_tokens=response.usage.completion_tokens,
            ...     model="gpt-4",
            ...     provider="openai",
            ... )
        """
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Record tokens
        self.record_tokens(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            provider=provider,
            operation=operation,
            **extra_attributes,
        )

        # Record duration and request count
        self.record_duration(
            duration_ms=duration_ms,
            model=model,
            provider=provider,
            operation=operation,
            is_error=is_error,
            **extra_attributes,
        )

        # Record TTFT if available (streaming)
        if ttft_ms is not None:
            self.record_ttft(
                ttft_ms=ttft_ms,
                model=model,
                provider=provider,
                **extra_attributes,
            )

    @staticmethod
    def _build_attributes(
        model: Optional[str] = None,
        provider: Optional[str] = None,
        operation: Optional[str] = None,
        **extra_attributes,
    ) -> Dict[str, Any]:
        """
        Build metric attributes dictionary.

        Follows OTEL GenAI semantic conventions for attribute names.

        Args:
            model: Model name
            provider: Provider name
            operation: Operation type
            **extra_attributes: Additional attributes

        Returns:
            Dictionary of metric attributes
        """
        attributes = {}

        if model:
            attributes["gen_ai.request.model"] = model
        if provider:
            attributes["gen_ai.provider.name"] = provider
        if operation:
            attributes["gen_ai.operation.name"] = operation

        # Add extra attributes
        attributes.update(extra_attributes)

        return attributes


def create_genai_metrics(meter: Meter) -> GenAIMetrics:
    """
    Create GenAI metrics instruments.

    Factory function for creating pre-configured metrics.

    Args:
        meter: OpenTelemetry Meter instance

    Returns:
        Configured GenAIMetrics instance

    Example:
        >>> from opentelemetry.sdk.metrics import MeterProvider
        >>> provider = MeterProvider()
        >>> meter = provider.get_meter("brokle")
        >>> metrics = create_genai_metrics(meter)
    """
    return GenAIMetrics(meter)
