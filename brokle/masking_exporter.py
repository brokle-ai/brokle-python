"""
Masking Span Exporter for PII redaction.

Provides a SpanExporter wrapper that applies PII masking using only
public OpenTelemetry APIs. This is the proper implementation that
avoids internal API access (like span._attributes).

Architecture:
    BrokleSpanProcessor (BatchSpanProcessor)
        -> on_end() receives ReadableSpan
        -> adds to batch queue (no modification)
        -> MaskingSpanExporter.export() receives spans
        -> creates MaskedReadableSpan wrappers with masked attributes
        -> forwards to underlying OTLP exporter

Why this works:
    - SpanExporter.export() receives ReadableSpan objects
    - We create MaskedReadableSpan wrappers implementing ReadableSpan protocol
    - The wrapper returns masked values from the attributes property
    - Forward the wrapped spans to the real exporter
    - Uses ONLY public APIs - no internal attribute access
"""

import logging
from types import MappingProxyType
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.trace import Link, SpanContext, SpanKind
from opentelemetry.trace.status import Status

logger = logging.getLogger(__name__)


class MaskedReadableSpan(ReadableSpan):
    """
    A ReadableSpan wrapper that returns masked attribute values.

    Implements the ReadableSpan protocol using only public APIs.
    All properties delegate to the wrapped span except for `attributes`,
    which returns masked values for sensitive keys.

    This is the proper way to mask span attributes without accessing
    internal APIs like span._attributes.
    """

    def __init__(
        self,
        span: ReadableSpan,
        mask_fn: Callable[[Any], Any],
        maskable_keys: Sequence[str],
    ):
        """
        Initialize masked span wrapper.

        Args:
            span: The original ReadableSpan to wrap
            mask_fn: Function to apply masking to attribute values
            maskable_keys: List of attribute keys that should be masked
        """
        self._span = span
        self._mask_fn = mask_fn
        self._maskable_keys = frozenset(maskable_keys)
        self._masked_attributes: Optional[Mapping[str, Any]] = None

    @property
    def attributes(self) -> Mapping[str, Any]:
        """Return attributes with sensitive values masked."""
        if self._masked_attributes is not None:
            return self._masked_attributes

        original = self._span.attributes
        if not original:
            self._masked_attributes = MappingProxyType({})
            return self._masked_attributes

        masked = {}
        for key, value in original.items():
            if key in self._maskable_keys:
                try:
                    masked[key] = self._mask_fn(value)
                except Exception as e:
                    logger.error(
                        f"Masking failed for attribute '{key}': "
                        f"{type(e).__name__}: {str(e)[:100]}"
                    )
                    masked[key] = "<fully masked due to failed mask function>"
            else:
                masked[key] = value

        self._masked_attributes = MappingProxyType(masked)
        return self._masked_attributes

    # ========== Delegate all other ReadableSpan properties to wrapped span ==========

    @property
    def name(self) -> str:
        """Return span name."""
        return self._span.name

    @property
    def context(self) -> Optional[SpanContext]:
        """Return span context."""
        return self._span.context  # type: ignore[no-any-return]

    def get_span_context(self) -> SpanContext:
        """Return span context (alternate method)."""
        return self._span.get_span_context()  # type: ignore[no-any-return]

    @property
    def parent(self) -> Optional[SpanContext]:
        """Return parent span context."""
        return self._span.parent

    @property
    def start_time(self) -> Optional[int]:
        """Return span start time in nanoseconds."""
        return self._span.start_time

    @property
    def end_time(self) -> Optional[int]:
        """Return span end time in nanoseconds."""
        return self._span.end_time

    @property
    def status(self) -> Status:
        """Return span status."""
        return self._span.status

    @property
    def kind(self) -> SpanKind:
        """Return span kind."""
        return self._span.kind

    @property
    def events(self) -> Sequence[Event]:
        """Return span events."""
        return self._span.events

    @property
    def links(self) -> Sequence[Link]:
        """Return span links."""
        return self._span.links

    @property
    def resource(self) -> Resource:
        """Return span resource."""
        return self._span.resource

    @property
    def instrumentation_scope(self) -> Optional[InstrumentationScope]:
        """Return instrumentation scope."""
        return self._span.instrumentation_scope

    @property
    def dropped_attributes(self) -> int:
        """Return count of dropped attributes."""
        return self._span.dropped_attributes

    @property
    def dropped_events(self) -> int:
        """Return count of dropped events."""
        return self._span.dropped_events

    @property
    def dropped_links(self) -> int:
        """Return count of dropped links."""
        return self._span.dropped_links

    # For compatibility with older OTEL versions that may use instrumentation_info
    @property
    def instrumentation_info(self) -> Any:
        """Return instrumentation info (deprecated, use instrumentation_scope)."""
        return getattr(self._span, "instrumentation_info", self.instrumentation_scope)


class MaskingSpanExporter(SpanExporter):
    """
    SpanExporter wrapper that applies PII masking before forwarding.

    This exporter wraps another exporter and applies masking to sensitive
    attributes using only public OpenTelemetry APIs. It creates
    MaskedReadableSpan wrappers for each span before forwarding to the
    underlying exporter.

    This is the proper implementation for PII masking that:
    - Uses only public APIs
    - Is future-proof across OTEL versions
    - Follows OTEL's plugin architecture
    - Is easy to test independently

    Example:
        >>> from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        >>> from brokle.masking_exporter import MaskingSpanExporter
        >>>
        >>> def mask_pii(value):
        ...     return "[MASKED]" if value else value
        >>>
        >>> otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")
        >>> masking_exporter = MaskingSpanExporter(
        ...     exporter=otlp_exporter,
        ...     mask_fn=mask_pii,
        ...     maskable_keys=["input.value", "output.value"],
        ... )
    """

    def __init__(
        self,
        exporter: SpanExporter,
        mask_fn: Callable[[Any], Any],
        maskable_keys: Sequence[str],
    ):
        """
        Initialize masking exporter wrapper.

        Args:
            exporter: The underlying SpanExporter to forward masked spans to
            mask_fn: Function to apply masking to attribute values
            maskable_keys: List of attribute keys that should be masked
        """
        self._exporter = exporter
        self._mask_fn = mask_fn
        self._maskable_keys = list(maskable_keys)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """
        Mask sensitive attributes and forward to underlying exporter.

        Args:
            spans: Sequence of ReadableSpan objects to export

        Returns:
            SpanExportResult from the underlying exporter
        """
        masked_spans: List[ReadableSpan] = [
            MaskedReadableSpan(span, self._mask_fn, self._maskable_keys)
            for span in spans
        ]
        return self._exporter.export(masked_spans)

    def shutdown(self) -> None:
        """Shutdown the underlying exporter."""
        self._exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush the underlying exporter."""
        return self._exporter.force_flush(timeout_millis)
