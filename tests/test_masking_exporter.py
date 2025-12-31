"""
Tests for MaskingSpanExporter and MaskedReadableSpan.

Tests the proper PII masking implementation using public OpenTelemetry APIs.
This replaces the old internal API approach (span._attributes) with an
exporter wrapper pattern.
"""

from types import MappingProxyType
from typing import Any
from unittest.mock import Mock

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.trace import SpanContext, TraceFlags
from opentelemetry.trace.status import Status, StatusCode

from brokle.masking_exporter import MaskedReadableSpan, MaskingSpanExporter
from brokle.types.attributes import MASKABLE_ATTRIBUTES
from brokle.types.attributes import BrokleOtelSpanAttributes as Attrs


def create_mock_span(attributes: dict) -> Mock:
    """Create a mock ReadableSpan with given attributes."""
    span = Mock(spec=ReadableSpan)
    span.attributes = (
        MappingProxyType(attributes) if attributes else MappingProxyType({})
    )
    span.name = "test-span"
    span.context = SpanContext(
        trace_id=0x12345678901234567890123456789012,
        span_id=0x1234567890123456,
        is_remote=False,
        trace_flags=TraceFlags(0x01),
    )
    span.get_span_context.return_value = span.context
    span.parent = None
    span.start_time = 1000000000
    span.end_time = 2000000000
    span.status = Status(StatusCode.OK)
    span.kind = None
    span.events = ()
    span.links = ()
    span.resource = Resource.create({})
    span.instrumentation_scope = InstrumentationScope("test", "1.0.0")
    span.dropped_attributes = 0
    span.dropped_events = 0
    span.dropped_links = 0
    return span


class TestMaskedReadableSpan:
    """Tests for MaskedReadableSpan wrapper class."""

    def test_masks_specified_attributes(self):
        """Verify only specified attributes are masked."""
        span = create_mock_span(
            {
                Attrs.INPUT_VALUE: "sensitive input",
                Attrs.OUTPUT_VALUE: "sensitive output",
                Attrs.GEN_AI_REQUEST_MODEL: "gpt-4",  # Not in maskable keys
            }
        )

        def simple_mask(value):
            return "[MASKED]"

        masked = MaskedReadableSpan(
            span=span,
            mask_fn=simple_mask,
            maskable_keys=[Attrs.INPUT_VALUE, Attrs.OUTPUT_VALUE],
        )

        assert masked.attributes[Attrs.INPUT_VALUE] == "[MASKED]"
        assert masked.attributes[Attrs.OUTPUT_VALUE] == "[MASKED]"
        assert masked.attributes[Attrs.GEN_AI_REQUEST_MODEL] == "gpt-4"

    def test_preserves_non_maskable_attributes(self):
        """Verify non-maskable attributes are passed through unchanged."""
        span = create_mock_span(
            {
                Attrs.GEN_AI_REQUEST_MODEL: "gpt-4",
                Attrs.GEN_AI_REQUEST_TEMPERATURE: 0.7,
                Attrs.GEN_AI_USAGE_INPUT_TOKENS: 100,
            }
        )

        masked = MaskedReadableSpan(
            span=span,
            mask_fn=lambda x: "[MASKED]",
            maskable_keys=[Attrs.INPUT_VALUE],  # None of the attributes match
        )

        assert masked.attributes[Attrs.GEN_AI_REQUEST_MODEL] == "gpt-4"
        assert masked.attributes[Attrs.GEN_AI_REQUEST_TEMPERATURE] == 0.7
        assert masked.attributes[Attrs.GEN_AI_USAGE_INPUT_TOKENS] == 100

    def test_handles_empty_attributes(self):
        """Verify empty attributes are handled correctly."""
        span = create_mock_span({})

        masked = MaskedReadableSpan(
            span=span,
            mask_fn=lambda x: "[MASKED]",
            maskable_keys=[Attrs.INPUT_VALUE],
        )

        assert dict(masked.attributes) == {}

    def test_handles_none_attributes(self):
        """Verify None attributes are handled correctly."""
        span = create_mock_span({})
        span.attributes = None

        masked = MaskedReadableSpan(
            span=span,
            mask_fn=lambda x: "[MASKED]",
            maskable_keys=[Attrs.INPUT_VALUE],
        )

        assert dict(masked.attributes) == {}

    def test_caches_masked_attributes(self):
        """Verify masked attributes are cached for efficiency."""
        call_count = 0

        def counting_mask(value):
            nonlocal call_count
            call_count += 1
            return f"[MASKED-{call_count}]"

        span = create_mock_span(
            {
                Attrs.INPUT_VALUE: "sensitive",
            }
        )

        masked = MaskedReadableSpan(
            span=span,
            mask_fn=counting_mask,
            maskable_keys=[Attrs.INPUT_VALUE],
        )

        # Access attributes multiple times
        _ = masked.attributes
        _ = masked.attributes
        _ = masked.attributes

        # Mask function should only be called once due to caching
        assert call_count == 1

    def test_handles_mask_function_error(self):
        """Verify graceful error handling when mask function fails."""

        def failing_mask(value):
            raise ValueError("Mask error")

        span = create_mock_span(
            {
                Attrs.INPUT_VALUE: "sensitive",
                Attrs.OUTPUT_VALUE: "also sensitive",
            }
        )

        masked = MaskedReadableSpan(
            span=span,
            mask_fn=failing_mask,
            maskable_keys=[Attrs.INPUT_VALUE, Attrs.OUTPUT_VALUE],
        )

        # Should return fallback value, not raise
        assert (
            masked.attributes[Attrs.INPUT_VALUE]
            == "<fully masked due to failed mask function>"
        )
        assert (
            masked.attributes[Attrs.OUTPUT_VALUE]
            == "<fully masked due to failed mask function>"
        )

    def test_delegates_name_property(self):
        """Verify name property delegates to wrapped span."""
        span = create_mock_span({})
        span.name = "custom-span-name"

        masked = MaskedReadableSpan(span, lambda x: x, [])

        assert masked.name == "custom-span-name"

    def test_delegates_context_property(self):
        """Verify context property delegates to wrapped span."""
        span = create_mock_span({})

        masked = MaskedReadableSpan(span, lambda x: x, [])

        assert masked.context == span.context
        assert masked.get_span_context() == span.context

    def test_delegates_timing_properties(self):
        """Verify timing properties delegate to wrapped span."""
        span = create_mock_span({})
        span.start_time = 1234567890
        span.end_time = 1234567899

        masked = MaskedReadableSpan(span, lambda x: x, [])

        assert masked.start_time == 1234567890
        assert masked.end_time == 1234567899

    def test_delegates_status_property(self):
        """Verify status property delegates to wrapped span."""
        span = create_mock_span({})
        span.status = Status(StatusCode.ERROR, "Something went wrong")

        masked = MaskedReadableSpan(span, lambda x: x, [])

        assert masked.status.status_code == StatusCode.ERROR
        assert masked.status.description == "Something went wrong"

    def test_delegates_resource_property(self):
        """Verify resource property delegates to wrapped span."""
        span = create_mock_span({})
        span.resource = Resource.create({"service.name": "test-service"})

        masked = MaskedReadableSpan(span, lambda x: x, [])

        assert masked.resource.attributes["service.name"] == "test-service"

    def test_complex_nested_masking(self):
        """Verify masking works with nested data structures."""

        def recursive_mask(data: Any) -> Any:
            if isinstance(data, str):
                return data.replace("secret", "[REDACTED]")
            elif isinstance(data, dict):
                return {k: recursive_mask(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [recursive_mask(item) for item in data]
            return data

        span = create_mock_span(
            {
                Attrs.METADATA: {
                    "user": {"password": "secret123", "name": "John"},
                    "data": ["secret info", "public info"],
                }
            }
        )

        masked = MaskedReadableSpan(
            span=span,
            mask_fn=recursive_mask,
            maskable_keys=[Attrs.METADATA],
        )

        result = masked.attributes[Attrs.METADATA]
        assert result["user"]["password"] == "[REDACTED]123"
        assert result["user"]["name"] == "John"
        assert result["data"][0] == "[REDACTED] info"
        assert result["data"][1] == "public info"


class TestMaskingSpanExporter:
    """Tests for MaskingSpanExporter wrapper class."""

    def test_exports_masked_spans(self):
        """Verify exporter wraps spans and forwards to underlying exporter."""
        underlying = Mock()
        underlying.export.return_value = SpanExportResult.SUCCESS

        exporter = MaskingSpanExporter(
            exporter=underlying,
            mask_fn=lambda x: "[MASKED]",
            maskable_keys=[Attrs.INPUT_VALUE],
        )

        span = create_mock_span(
            {
                Attrs.INPUT_VALUE: "sensitive",
                Attrs.GEN_AI_REQUEST_MODEL: "gpt-4",
            }
        )

        result = exporter.export([span])

        assert result == SpanExportResult.SUCCESS
        underlying.export.assert_called_once()

        # Get the masked spans that were passed to underlying exporter
        exported_spans = underlying.export.call_args[0][0]
        assert len(exported_spans) == 1

        masked_span = exported_spans[0]
        assert masked_span.attributes[Attrs.INPUT_VALUE] == "[MASKED]"
        assert masked_span.attributes[Attrs.GEN_AI_REQUEST_MODEL] == "gpt-4"

    def test_exports_multiple_spans(self):
        """Verify exporter handles multiple spans correctly."""
        underlying = Mock()
        underlying.export.return_value = SpanExportResult.SUCCESS

        exporter = MaskingSpanExporter(
            exporter=underlying,
            mask_fn=lambda x: f"[MASKED:{x}]",
            maskable_keys=[Attrs.INPUT_VALUE],
        )

        spans = [
            create_mock_span({Attrs.INPUT_VALUE: "input1"}),
            create_mock_span({Attrs.INPUT_VALUE: "input2"}),
            create_mock_span({Attrs.INPUT_VALUE: "input3"}),
        ]

        result = exporter.export(spans)

        assert result == SpanExportResult.SUCCESS
        exported_spans = underlying.export.call_args[0][0]
        assert len(exported_spans) == 3

        assert exported_spans[0].attributes[Attrs.INPUT_VALUE] == "[MASKED:input1]"
        assert exported_spans[1].attributes[Attrs.INPUT_VALUE] == "[MASKED:input2]"
        assert exported_spans[2].attributes[Attrs.INPUT_VALUE] == "[MASKED:input3]"

    def test_delegates_shutdown(self):
        """Verify shutdown is delegated to underlying exporter."""
        underlying = Mock()

        exporter = MaskingSpanExporter(
            exporter=underlying,
            mask_fn=lambda x: x,
            maskable_keys=[],
        )

        exporter.shutdown()

        underlying.shutdown.assert_called_once()

    def test_delegates_force_flush(self):
        """Verify force_flush is delegated to underlying exporter."""
        underlying = Mock()
        underlying.force_flush.return_value = True

        exporter = MaskingSpanExporter(
            exporter=underlying,
            mask_fn=lambda x: x,
            maskable_keys=[],
        )

        result = exporter.force_flush(timeout_millis=5000)

        assert result is True
        underlying.force_flush.assert_called_once_with(5000)

    def test_handles_empty_spans_list(self):
        """Verify empty spans list is handled correctly."""
        underlying = Mock()
        underlying.export.return_value = SpanExportResult.SUCCESS

        exporter = MaskingSpanExporter(
            exporter=underlying,
            mask_fn=lambda x: "[MASKED]",
            maskable_keys=[Attrs.INPUT_VALUE],
        )

        result = exporter.export([])

        assert result == SpanExportResult.SUCCESS
        underlying.export.assert_called_once_with([])

    def test_propagates_export_failure(self):
        """Verify export failures are propagated from underlying exporter."""
        underlying = Mock()
        underlying.export.return_value = SpanExportResult.FAILURE

        exporter = MaskingSpanExporter(
            exporter=underlying,
            mask_fn=lambda x: "[MASKED]",
            maskable_keys=[Attrs.INPUT_VALUE],
        )

        span = create_mock_span({Attrs.INPUT_VALUE: "sensitive"})
        result = exporter.export([span])

        assert result == SpanExportResult.FAILURE


class TestIntegrationWithMaskableAttributes:
    """Test integration with the MASKABLE_ATTRIBUTES constant."""

    def test_masks_all_maskable_attributes(self):
        """Verify all MASKABLE_ATTRIBUTES are properly masked."""
        underlying = Mock()
        underlying.export.return_value = SpanExportResult.SUCCESS

        exporter = MaskingSpanExporter(
            exporter=underlying,
            mask_fn=lambda x: "[REDACTED]",
            maskable_keys=MASKABLE_ATTRIBUTES,
        )

        # Create span with all maskable attributes plus some non-maskable ones
        span = create_mock_span(
            {
                Attrs.INPUT_VALUE: "user input",
                Attrs.OUTPUT_VALUE: "model output",
                Attrs.GEN_AI_INPUT_MESSAGES: '[{"role": "user", "content": "secret"}]',
                Attrs.GEN_AI_OUTPUT_MESSAGES: '[{"role": "assistant", "content": "response"}]',
                Attrs.METADATA: {"key": "value"},
                # Non-maskable attributes
                Attrs.GEN_AI_REQUEST_MODEL: "gpt-4",
                Attrs.GEN_AI_USAGE_INPUT_TOKENS: 100,
            }
        )

        exporter.export([span])

        exported_spans = underlying.export.call_args[0][0]
        masked_span = exported_spans[0]

        # All maskable attributes should be masked
        assert masked_span.attributes[Attrs.INPUT_VALUE] == "[REDACTED]"
        assert masked_span.attributes[Attrs.OUTPUT_VALUE] == "[REDACTED]"
        assert masked_span.attributes[Attrs.GEN_AI_INPUT_MESSAGES] == "[REDACTED]"
        assert masked_span.attributes[Attrs.GEN_AI_OUTPUT_MESSAGES] == "[REDACTED]"
        assert masked_span.attributes[Attrs.METADATA] == "[REDACTED]"

        # Non-maskable attributes should be unchanged
        assert masked_span.attributes[Attrs.GEN_AI_REQUEST_MODEL] == "gpt-4"
        assert masked_span.attributes[Attrs.GEN_AI_USAGE_INPUT_TOKENS] == 100


class TestPerformance:
    """Performance-related tests."""

    def test_lazy_attribute_masking(self):
        """Verify attributes are not masked until accessed."""
        mask_call_count = 0

        def counting_mask(value):
            nonlocal mask_call_count
            mask_call_count += 1
            return "[MASKED]"

        span = create_mock_span({Attrs.INPUT_VALUE: "sensitive"})

        # Creating masked span should not call mask function
        masked = MaskedReadableSpan(
            span=span,
            mask_fn=counting_mask,
            maskable_keys=[Attrs.INPUT_VALUE],
        )

        assert mask_call_count == 0

        # Accessing attributes should trigger masking
        _ = masked.attributes

        assert mask_call_count == 1
