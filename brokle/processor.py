"""
Brokle span processor extending OpenTelemetry's BatchSpanProcessor.

Provides span-level processing including PII masking while delegating
batching, queuing, retry logic, and sampling to OpenTelemetry SDK.

Note: Sampling is handled by TracerProvider's TraceIdRatioBased sampler
(configured in client.py), not by this processor. This ensures entire
traces are sampled together (not individual spans).
"""

import logging
from typing import Any, Optional

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

from .config import BrokleConfig
from .types.attributes import BrokleOtelSpanAttributes as Attrs

logger = logging.getLogger(__name__)

# Attributes that should be masked if masking is configured
MASKABLE_ATTRIBUTES = [
    Attrs.INPUT_VALUE,
    Attrs.OUTPUT_VALUE,
    Attrs.GEN_AI_INPUT_MESSAGES,
    Attrs.GEN_AI_OUTPUT_MESSAGES,
    Attrs.METADATA,
]


class BrokleSpanProcessor(BatchSpanProcessor):
    """
    Brokle span processor extending OpenTelemetry's BatchSpanProcessor.

    Provides PII masking while delegating batching, queuing, and retry to OTEL.
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
        span: "Span",  # type: ignore
        parent_context: Optional[Context] = None,
    ) -> None:
        """Called when a span is started. Sets environment and release attributes."""
        if self.config.environment:
            span.set_attribute(Attrs.BROKLE_ENVIRONMENT, self.config.environment)

        if self.config.release:
            span.set_attribute(Attrs.BROKLE_RELEASE, self.config.release)

        super().on_start(span, parent_context)

    def on_end(self, span: ReadableSpan) -> None:
        """Called when span ends. Applies PII masking if configured."""
        if self.config.mask:
            self._apply_masking(span)

        super().on_end(span)

    def shutdown(self) -> None:
        """Shut down the processor and flush pending spans."""
        super().shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush all pending spans."""
        return super().force_flush(timeout_millis)

    def _apply_masking(self, span: ReadableSpan) -> None:
        """
        Apply PII masking to sensitive span attributes.

        Uses span._attributes (internal OpenTelemetry API) because span.attributes
        is immutable (MappingProxyType). No official API exists for post-processing.
        May break in future OTEL versions.

        See: https://github.com/open-telemetry/opentelemetry-specification/issues/2990
        """
        if not span._attributes:
            return

        for attr_key in MASKABLE_ATTRIBUTES:
            if attr_key in span._attributes:
                original_value = span._attributes[attr_key]
                masked_value = self._mask_attribute(original_value)
                span._attributes[attr_key] = masked_value

    def _mask_attribute(self, data: Any) -> Any:
        """Apply masking function with error fallback."""
        try:
            return self.config.mask(data)
        except Exception as e:
            logger.error(f"Masking failed: {type(e).__name__}: {str(e)[:100]}")
            return "<fully masked due to failed mask function>"
