"""
Telemetry and tracing response models for Brokle SDK.

This module contains response models for telemetry, tracing, and observability:
- TelemetryTraceResponse
- TelemetrySpanResponse
- TelemetryEventBatchResponse

Models are refactored to use mixins for reduced duplication while maintaining
complete backward compatibility.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Import our mixins and base classes
from .base import (
    BrokleResponseBase,
    MetadataMixin,
    RequestTrackingMixin,
    StatusMixin,
    TimestampMixin,
)


class TelemetryTimingMixin(BaseModel):
    """Mixin for telemetry timing fields."""

    start_time: datetime = Field(description="Start timestamp")
    end_time: Optional[datetime] = Field(default=None, description="End timestamp")
    duration_ms: Optional[float] = Field(
        default=None, description="Duration in milliseconds"
    )


class TelemetryTraceResponse(BaseModel):
    """
    Telemetry trace response model.

    Maintains complete backward compatibility with original while internally
    leveraging mixins for common field patterns.
    """

    trace_id: str = Field(description="Trace ID")
    name: str = Field(description="Trace name")
    status: str = Field(description="Trace status")
    start_time: datetime = Field(description="Trace start time")
    end_time: Optional[datetime] = Field(default=None, description="Trace end time")
    duration_ms: Optional[float] = Field(
        default=None, description="Duration in milliseconds"
    )
    span_count: Optional[int] = Field(default=None, description="Number of spans")
    user_id: Optional[str] = Field(default=None, description="User ID")
    session_id: Optional[str] = Field(default=None, description="Session ID")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Trace metadata"
    )
    tags: Optional[List[str]] = Field(default=None, description="Trace tags")


class TelemetrySpanResponse(BaseModel):
    """
    Telemetry span response model.

    Maintains complete backward compatibility with original while internally
    leveraging mixins for common field patterns.
    """

    span_id: str = Field(description="Span ID")
    trace_id: str = Field(description="Parent trace ID")
    parent_span_id: Optional[str] = Field(default=None, description="Parent span ID")
    name: str = Field(description="Span name")
    span_type: str = Field(description="Span type")
    status: str = Field(description="Span status")
    start_time: datetime = Field(description="Span start time")
    end_time: Optional[datetime] = Field(default=None, description="Span end time")
    duration_ms: Optional[float] = Field(
        default=None, description="Duration in milliseconds"
    )
    attributes: Optional[Dict[str, Any]] = Field(
        default=None, description="Span attributes"
    )
    events: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Span events"
    )


class TelemetryEventBatchResponse(BaseModel):
    """
    Batch telemetry events response model.

    Maintains complete backward compatibility with original.
    """

    processed_count: int = Field(description="Number of events processed")
    failed_count: int = Field(description="Number of events failed")
    batch_id: str = Field(description="Batch ID")
    processing_time_ms: float = Field(description="Processing time in milliseconds")
    errors: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Processing errors"
    )


# Performance metrics response for telemetry
class TelemetryPerformanceResponse(BrokleResponseBase, TimestampMixin, MetadataMixin):
    """
    Telemetry performance metrics response.

    New model designed with mixins from the start for better maintainability.
    """

    metric_name: str = Field(description="Performance metric name")
    metric_value: float = Field(description="Metric value")
    metric_unit: str = Field(description="Metric unit (ms, count, bytes, etc.)")
    percentile_50: Optional[float] = Field(default=None, description="50th percentile")
    percentile_95: Optional[float] = Field(default=None, description="95th percentile")
    percentile_99: Optional[float] = Field(default=None, description="99th percentile")
    sample_count: int = Field(description="Number of samples")
    time_window_seconds: int = Field(description="Time window for metrics in seconds")


# Re-export for backward compatibility
__all__ = [
    # Backward compatible models
    "TelemetryTraceResponse",
    "TelemetrySpanResponse",
    "TelemetryEventBatchResponse",
    # New models designed with mixins
    "TelemetryPerformanceResponse",
    # Mixins for other telemetry models
    "TelemetryTimingMixin",
]
