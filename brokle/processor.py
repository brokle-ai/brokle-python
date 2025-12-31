"""
Brokle span processor extending OpenTelemetry's BatchSpanProcessor.

Provides span-level attribute enrichment (environment, release) while
delegating batching, queuing, retry logic, and sampling to OpenTelemetry SDK.

Note: Sampling is handled by TracerProvider's TraceIdRatioBased sampler
(configured in client.py), not by this processor. This ensures entire
traces are sampled together (not individual spans).

Note: PII masking is handled at the exporter layer via MaskingSpanExporter,
not in this processor. This uses only public OpenTelemetry APIs and avoids
accessing internal span attributes.
"""

import logging
from typing import Optional

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

from .config import BrokleConfig

# Re-export MASKABLE_ATTRIBUTES for backwards compatibility
# (now defined in types/attributes.py)
from .types.attributes import MASKABLE_ATTRIBUTES  # noqa: F401
from .types.attributes import BrokleOtelSpanAttributes as Attrs

logger = logging.getLogger(__name__)


class BrokleSpanProcessor(BatchSpanProcessor):
    """
    Brokle span processor extending OpenTelemetry's BatchSpanProcessor.

    Enriches spans with environment and release attributes on start.
    Batching, queuing, retry, and export are delegated to OTEL SDK.

    Note: PII masking is handled at the exporter layer (MaskingSpanExporter),
    not in this processor. This ensures we use only public OTEL APIs.
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
        """Initialize Brokle span processor with BatchSpanProcessor configuration."""
        queue_size = max_queue_size or config.max_queue_size
        delay_millis = schedule_delay_millis or int(config.flush_interval * 1000)
        batch_size = max_export_batch_size or config.flush_at
        timeout_millis = export_timeout_millis or config.export_timeout

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
        span: "Span",
        parent_context: Optional[Context] = None,
    ) -> None:
        """Called when a span is started. Sets environment and release attributes."""
        if self.config.environment:
            span.set_attribute(Attrs.BROKLE_ENVIRONMENT, self.config.environment)

        if self.config.release:
            span.set_attribute(Attrs.BROKLE_RELEASE, self.config.release)

        super().on_start(span, parent_context)

    def on_end(self, span: ReadableSpan) -> None:
        """
        Called when span ends. Forward to BatchSpanProcessor for batching.

        Note: PII masking is handled at the exporter layer (MaskingSpanExporter),
        not here. This ensures we use only public OTEL APIs and don't access
        internal span attributes.
        """
        super().on_end(span)

    def shutdown(self) -> None:
        """Shut down the processor and flush pending spans."""
        super().shutdown()

    def force_flush(self, timeout_millis: Optional[int] = None) -> bool:
        """Force flush all pending spans."""
        return super().force_flush(timeout_millis)
