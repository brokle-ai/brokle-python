"""
Telemetry batch API types for unified event submission.

This module defines the event envelope pattern for the unified
/v1/telemetry/batch endpoint, supporting traces, observations,
quality scores, and generic events.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TelemetryEventType(str, Enum):
    """
    Supported telemetry event types.

    The unified batch API supports these event types for structured
    observability data submission.
    """

    # Trace operations
    TRACE_CREATE = "trace_create"
    TRACE_UPDATE = "trace_update"

    # Observation operations
    OBSERVATION_CREATE = "observation_create"
    OBSERVATION_UPDATE = "observation_update"
    OBSERVATION_COMPLETE = "observation_complete"

    # Quality score operations
    QUALITY_SCORE_CREATE = "quality_score_create"
    QUALITY_SCORE_UPDATE = "quality_score_update"

    # Generic event operation
    EVENT_CREATE = "event_create"


class TelemetryEvent(BaseModel):
    """
    Telemetry event envelope.

    Wraps individual telemetry operations (traces, observations, scores)
    in a consistent envelope format for batch submission.

    Attributes:
        event_id: Unique ULID identifier for deduplication
        event_type: Type of telemetry event
        payload: Event-specific data (trace fields, observation fields, etc.)
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

    class Config:
        use_enum_values = True


class DeduplicationConfig(BaseModel):
    """
    Deduplication configuration for batch submission.

    Controls how the batch API handles duplicate events based on event_id.

    Attributes:
        enabled: Enable deduplication (default: True)
        ttl: Time-to-live for deduplication cache in seconds (default: 3600)
        use_redis_cache: Use Redis for distributed deduplication (default: True)
        fail_on_duplicate: Fail entire batch if duplicates found (default: False)
    """

    enabled: bool = Field(True, description="Enable event deduplication")
    ttl: int = Field(3600, description="Deduplication cache TTL in seconds", ge=60, le=86400)
    use_redis_cache: bool = Field(
        True, description="Use Redis for distributed deduplication"
    )
    fail_on_duplicate: bool = Field(
        False, description="Fail entire batch on duplicate events"
    )


class TelemetryBatchRequest(BaseModel):
    """
    Unified telemetry batch request.

    Submits multiple telemetry events (traces, observations, scores) in a
    single batch to /v1/telemetry/batch endpoint.

    Attributes:
        events: List of telemetry events to submit
        environment: Optional environment tag (e.g., "production", "staging")
        metadata: Optional batch-level metadata
        async_mode: Process batch asynchronously (default: False)
        deduplication: Deduplication configuration
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
    deduplication: Optional[DeduplicationConfig] = Field(
        None, description="Deduplication configuration"
    )


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
    "DeduplicationConfig",
    "TelemetryBatchRequest",
    "TelemetryBatchResponse",
    "BatchEventError",
]
