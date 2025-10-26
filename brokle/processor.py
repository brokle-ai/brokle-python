"""
Brokle span processor extending OpenTelemetry's BatchSpanProcessor.

Provides span-level filtering and processing while delegating batching,
queuing, and retry logic to OpenTelemetry SDK.
"""

import random
from typing import Optional
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.context import Context

from .config import BrokleConfig


class BrokleSpanProcessor(BatchSpanProcessor):
    """
    Custom span processor for Brokle observability.

    Extends BatchSpanProcessor to provide:
    - Sampling based on config.sample_rate
    - Span-level filtering (future: PII masking)
    - Resource attribute enrichment (project_id, environment)

    All batching, flushing, queuing, and retry logic is handled by the
    parent BatchSpanProcessor class from OpenTelemetry SDK.

    Note: Resource attributes (project_id, environment) are set at
    TracerProvider initialization and automatically included in all spans.
    This processor is primarily for span-level filtering and sampling.
    """

    def __init__(
        self,
        span_exporter: SpanExporter,
        config: BrokleConfig,
        *,
        max_queue_size: Optional[int] = None,
        schedule_delay_millis: Optional[int] = None,
        max_export_batch_size: Optional[int] = None,
        export_timeout_millis: Optional[int] = None,
    ):
        """
        Initialize Brokle span processor.

        Args:
            span_exporter: OTLP span exporter instance
            config: Brokle configuration
            max_queue_size: Max spans in queue (default: from config or 2048)
            schedule_delay_millis: Flush interval in ms (default: from config or 5000)
            max_export_batch_size: Max spans per batch (default: from config or 512)
            export_timeout_millis: Export timeout in ms (default: from config or 30000)
        """
        # Use config values with fallbacks
        queue_size = max_queue_size or config.max_queue_size
        delay_millis = schedule_delay_millis or int(config.flush_interval * 1000)
        batch_size = max_export_batch_size or config.flush_at
        timeout_millis = export_timeout_millis or config.export_timeout

        # Initialize parent BatchSpanProcessor
        super().__init__(
            span_exporter=span_exporter,
            max_queue_size=queue_size,
            schedule_delay_millis=delay_millis,
            max_export_batch_size=batch_size,
            export_timeout_millis=timeout_millis,
        )

        self.config = config
        self._sample_rate = config.sample_rate

    def on_start(
        self,
        span: "Span",  # type: ignore
        parent_context: Optional[Context] = None,
    ) -> None:
        """
        Called when a span is started.

        This is where we could add span start-time processing, but
        currently we don't need any custom logic here.

        Args:
            span: The span that was started
            parent_context: Parent context (if any)
        """
        # Resource attributes are already set at TracerProvider level
        # No additional processing needed at span start
        super().on_start(span, parent_context)

    def on_end(self, span: ReadableSpan) -> None:
        """
        Called when span ends.

        Implements sampling logic before passing to batch processor.
        If span is sampled out, it won't be exported.

        Args:
            span: The span that ended
        """
        # Apply sampling if configured
        if self._sample_rate < 1.0:
            # Random sampling based on sample_rate
            if random.random() > self._sample_rate:
                # Drop this span (don't export)
                return

        # Future: Apply PII masking here if configured
        # if self.config.mask:
        #     self._apply_masking(span)

        # Pass to parent for batching and export
        super().on_end(span)

    def shutdown(self) -> None:
        """
        Shut down the processor.

        Flushes all pending spans and closes the exporter.
        """
        super().shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """
        Force flush all pending spans.

        Args:
            timeout_millis: Timeout in milliseconds

        Returns:
            True if successful, False otherwise
        """
        return super().force_flush(timeout_millis)


class SimpleSampler:
    """
    Simple probabilistic sampler.

    This is a helper class for understanding sampling logic.
    In production, we use the built-in sampling in BrokleSpanProcessor.
    """

    def __init__(self, sample_rate: float):
        """
        Initialize sampler.

        Args:
            sample_rate: Probability of sampling (0.0 to 1.0)
        """
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError("sample_rate must be between 0.0 and 1.0")
        self.sample_rate = sample_rate

    def should_sample(self) -> bool:
        """
        Determine if this span should be sampled.

        Returns:
            True if span should be sampled, False otherwise
        """
        if self.sample_rate >= 1.0:
            return True
        if self.sample_rate <= 0.0:
            return False
        return random.random() <= self.sample_rate
