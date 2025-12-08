"""
Brokle span processor extending OpenTelemetry's BatchSpanProcessor.

Provides span-level processing (future: PII masking) while delegating
batching, queuing, retry logic, and sampling to OpenTelemetry SDK.

Note: Sampling is handled by TracerProvider's TraceIdRatioBased sampler
(configured in client.py), not by this processor. This ensures entire
traces are sampled together (not individual spans).
"""

from typing import Optional

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

from .config import BrokleConfig
from .types.attributes import BrokleOtelSpanAttributes as Attrs


class BrokleSpanProcessor(BatchSpanProcessor):
    """
    Custom span processor for Brokle observability.

    Extends BatchSpanProcessor to provide span-level processing hooks.
    Future use cases: PII masking, attribute transformation, custom filtering.

    All batching, flushing, queuing, retry logic, and sampling is handled by
    OpenTelemetry SDK. Sampling decisions are made at the TracerProvider level
    using TraceIdRatioBased sampler (see client.py) to ensure entire traces
    are sampled together.

    Note: Resource attributes (project_id, environment) are set at
    TracerProvider initialization and automatically included in all spans.
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

    def on_start(
        self,
        span: "Span",  # type: ignore
        parent_context: Optional[Context] = None,
    ) -> None:
        """
        Called when a span is started.

        Sets environment as span attribute.

        Args:
            span: The span that was started
            parent_context: Parent context (if any)
        """
        # Add environment as span attribute (not resource attribute)
        if self.config.environment:
            span.set_attribute(Attrs.BROKLE_ENVIRONMENT, self.config.environment)

        # Add release as span attribute (for experiment tracking)
        if self.config.release:
            span.set_attribute(Attrs.BROKLE_RELEASE, self.config.release)

        super().on_start(span, parent_context)

    def on_end(self, span: ReadableSpan) -> None:
        """
        Called when span ends.

        Sampling is handled by TracerProvider's TraceIdRatioBased sampler
        (configured in client.py). This ensures entire traces are sampled
        together based on trace_id hash (deterministic, not random per-span).

        Args:
            span: The span that ended
        """
        # Sampling decision already made by TracerProvider sampler
        # If span.is_recording() is False, OpenTelemetry won't call this method

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
