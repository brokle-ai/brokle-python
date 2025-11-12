"""
Telemetry batch API types for unified event submission.

This module defines the event envelope pattern for the unified
/v1/ingest/batch endpoint, supporting traces, spans,
quality scores, and generic events.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TelemetryEventType(str, Enum):
    """
    Immutable event types for telemetry batch API.

    IMPORTANT: Batch events are write-once and immutable. They represent
    initial creation only. For updates/corrections after initial submission,
    use the REST API endpoints:
    - PUT /api/v1/analytics/traces/:id
    - PUT /api/v1/analytics/spans/:id
    - PUT /api/v1/analytics/scores/:id
    - PUT /api/v1/analytics/sessions/:id

    This follows industry patterns where:
    - Batch API = immutable, async, high-throughput (create operations)
    - REST API = mutable, sync, for corrections/enrichment (update operations)
    """

    # Structured observability (immutable batch creation)
    TRACE = "trace"
    OBSERVATION = "span"
    QUALITY_SCORE = "quality_score"
    SESSION = "session"


class TelemetryEvent(BaseModel):
    """
    Telemetry event envelope.

    Wraps individual telemetry operations (traces, spans, scores)
    in a consistent envelope format for batch submission.

    Attributes:
        event_id: Unique ULID identifier for deduplication
        event_type: Type of telemetry event
        payload: Event-specific data (trace fields, span fields, etc.)
        timestamp: Optional Unix timestamp (defaults to server time)
    """

    event_id: str = Field(
        ...,
        description="Unique ULID identifier for event deduplication",
        min_length=26,
        max_length=26,
    )
    event_type: TelemetryEventType = Field(
        ..., description="Type of telemetry event"
    )
    payload: Dict[str, Any] = Field(
        ..., description="Event-specific data"
    )
    timestamp: Optional[int] = Field(
        None, description="Unix timestamp in seconds (defaults to server time)"
    )

    model_config = ConfigDict(use_enum_values=True)


class TelemetryBatchRequest(BaseModel):
    """
    Unified telemetry batch request for immutable event creation.

    Submits multiple telemetry events (traces, spans, scores) in a
    single batch to /v1/ingest/batch endpoint.

    IMPORTANT: This endpoint is for INITIAL CREATION ONLY. Events are immutable
    once submitted. For updates/corrections after creation, use the REST API
    PUT endpoints on the dashboard.

    Best Practice: Buffer complete event data before submission. Ensure all
    fields (input, output, metadata, cost details, etc.) are finalized before
    sending to this endpoint.

    Attributes:
        events: List of telemetry events to submit (immutable after processing)
        environment: Optional environment tag (e.g., "production", "staging")
        metadata: Optional batch-level metadata
        async_mode: Process batch asynchronously (default: False)
    """

    events: List[TelemetryEvent] = Field(
        ..., description="List of telemetry events", min_length=1, max_length=1000
    )
    environment: Optional[str] = Field(
        None, description="Environment tag", max_length=40
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Batch-level metadata"
    )
    async_mode: bool = Field(False, description="Process batch asynchronously")


class BatchEventError(BaseModel):
    """
    Error details for failed event in batch.

    Attributes:
        event_id: ULID of failed event
        error: Error message
        details: Optional detailed error information
    """

    event_id: str = Field(..., description="ULID of failed event")
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Detailed error information")


class TelemetryBatchResponse(BaseModel):
    """
    Unified telemetry batch response.

    Contains processing results including success/failure counts and
    error details for partial failures.

    Attributes:
        batch_id: Unique identifier for this batch
        processed_events: Number of successfully processed events
        duplicate_events: Number of duplicate events skipped
        failed_events: Number of failed events
        processing_time_ms: Batch processing time in milliseconds
        errors: List of errors for failed events
        duplicate_event_ids: List of duplicate event IDs
        job_id: Background job ID if async_mode=True
    """

    batch_id: str = Field(..., description="Unique batch identifier (ULID)")
    processed_events: int = Field(..., description="Successfully processed events", ge=0)
    duplicate_events: int = Field(..., description="Duplicate events skipped", ge=0)
    failed_events: int = Field(..., description="Failed events", ge=0)
    processing_time_ms: int = Field(..., description="Processing time in milliseconds", ge=0)
    errors: List[BatchEventError] = Field(
        default_factory=list, description="Errors for failed events"
    )
    duplicate_event_ids: List[str] = Field(
        default_factory=list, description="Duplicate event IDs"
    )
    job_id: Optional[str] = Field(
        None, description="Background job ID for async processing"
    )


__all__ = [
    "TelemetryEventType",
    "TelemetryEvent",
    "TelemetryBatchRequest",
    "TelemetryBatchResponse",
    "BatchEventError",
]
