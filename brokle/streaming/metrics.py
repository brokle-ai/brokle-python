"""
Streaming-specific metrics for LLM responses.

Provides metrics recording for TTFT, inter-token latency, and other
streaming-related metrics using the GenAIMetrics infrastructure.
"""

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..metrics import GenAIMetrics
    from .accumulator import StreamingResult

logger = logging.getLogger(__name__)


class StreamingMetrics:
    """
    Helper class for recording streaming-specific metrics.

    Wraps GenAIMetrics to provide convenient methods for recording
    streaming response metrics.

    Example:
        >>> metrics = client.get_metrics()
        >>> streaming_metrics = StreamingMetrics(metrics)
        >>> streaming_metrics.record_from_result(result, model="gpt-4")
    """

    def __init__(self, metrics: Optional["GenAIMetrics"] = None):
        """
        Initialize streaming metrics helper.

        Args:
            metrics: GenAIMetrics instance (can be None if metrics disabled)
        """
        self._metrics = metrics

    def record_ttft(
        self,
        ttft_ms: float,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **extra_attributes,
    ) -> None:
        """
        Record time to first token metric.

        Args:
            ttft_ms: Time to first token in milliseconds
            model: Model name
            provider: Provider name
            **extra_attributes: Additional metric attributes
        """
        if self._metrics:
            self._metrics.record_ttft(
                ttft_ms=ttft_ms,
                model=model,
                provider=provider,
                **extra_attributes,
            )

    def record_inter_token_latency(
        self,
        latency_ms: float,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **extra_attributes,
    ) -> None:
        """
        Record inter-token latency metric.

        Args:
            latency_ms: Inter-token latency in milliseconds
            model: Model name
            provider: Provider name
            **extra_attributes: Additional metric attributes
        """
        if self._metrics:
            self._metrics.record_inter_token_latency(
                latency_ms=latency_ms,
                model=model,
                provider=provider,
                **extra_attributes,
            )

    def record_from_result(
        self,
        result: "StreamingResult",
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **extra_attributes,
    ) -> None:
        """
        Record all streaming metrics from a StreamingResult.

        This is the recommended way to record metrics after streaming completes.

        Args:
            result: StreamingResult from StreamingAccumulator.finalize()
            model: Model name (uses result.model if not provided)
            provider: Provider name
            **extra_attributes: Additional metric attributes
        """
        if not self._metrics:
            return

        # Use model from result if not explicitly provided
        effective_model = model or result.model

        # Record TTFT
        if result.ttft_ms is not None:
            self.record_ttft(
                ttft_ms=result.ttft_ms,
                model=effective_model,
                provider=provider,
                **extra_attributes,
            )

        # Record average inter-token latency
        if result.avg_inter_token_latency_ms is not None:
            self.record_inter_token_latency(
                latency_ms=result.avg_inter_token_latency_ms,
                model=effective_model,
                provider=provider,
                **extra_attributes,
            )

        # Record individual inter-token latencies for distribution analysis
        # (optional - can be expensive for long streams)
        # for latency in result.inter_token_latencies:
        #     self.record_inter_token_latency(
        #         latency_ms=latency,
        #         model=effective_model,
        #         provider=provider,
        #         **extra_attributes,
        #     )

        # Record token usage
        if result.usage:
            input_tokens = result.usage.get("prompt_tokens", 0)
            output_tokens = result.usage.get("completion_tokens", 0)
            if input_tokens > 0 or output_tokens > 0:
                self._metrics.record_tokens(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=effective_model,
                    provider=provider,
                    operation="streaming",
                    **extra_attributes,
                )

        # Record duration
        if result.total_duration_ms is not None:
            self._metrics.record_duration(
                duration_ms=result.total_duration_ms,
                model=effective_model,
                provider=provider,
                operation="streaming",
                **extra_attributes,
            )

    @property
    def enabled(self) -> bool:
        """Check if metrics are enabled."""
        return self._metrics is not None
