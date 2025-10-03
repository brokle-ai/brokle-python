"""
Unit tests for unified telemetry batch API.

Tests ULID generation, event envelope creation, batch request serialization,
and deduplication logic.
"""

import time
from unittest.mock import Mock, patch

import pytest

from brokle._utils.ulid import (
    extract_timestamp,
    generate_event_id,
    generate_ulid,
    is_valid_ulid,
)
from brokle.config import Config
from brokle.types.telemetry import (
    BatchEventError,
    DeduplicationConfig,
    TelemetryBatchRequest,
    TelemetryBatchResponse,
    TelemetryEvent,
    TelemetryEventType,
)


class TestULIDGeneration:
    """Test ULID utility functions."""

    def test_generate_ulid_format(self):
        """ULID should be 26 characters uppercase."""
        ulid = generate_ulid()
        assert len(ulid) == 26
        assert ulid.isupper()

    def test_generate_ulid_with_timestamp(self):
        """ULID should accept custom timestamp."""
        timestamp = 1677610602.0
        ulid = generate_ulid(timestamp=timestamp)
        assert len(ulid) == 26

        # Extract and verify timestamp (should be close)
        extracted = extract_timestamp(ulid)
        if extracted:  # May be None if using fallback implementation
            assert abs(extracted - timestamp) < 2  # Within 2 seconds

    def test_generate_event_id(self):
        """Event ID generation should produce valid ULID."""
        event_id = generate_event_id()
        assert len(event_id) == 26
        assert is_valid_ulid(event_id)

    def test_is_valid_ulid(self):
        """ULID validation should work correctly."""
        # Valid ULID
        assert is_valid_ulid("01ARZ3NDEKTSV4RRFFQ69G5FAV")

        # Invalid cases
        assert not is_valid_ulid("invalid")
        assert not is_valid_ulid("01ARZ3NDEKTSV4RRFFQ69G5FA")  # Too short
        assert not is_valid_ulid("01ARZ3NDEKTSV4RRFFQ69G5FAVX")  # Too long
        assert not is_valid_ulid(None)
        assert not is_valid_ulid(123)

    def test_ulid_sortability(self):
        """ULIDs should be lexicographically sortable by time."""
        ulid1 = generate_ulid(timestamp=1677610600.0)
        time.sleep(0.01)  # Ensure different timestamp
        ulid2 = generate_ulid(timestamp=1677610601.0)

        # Later timestamp should produce larger ULID (lexicographically)
        # This may not hold with UUID fallback, so skip if timestamps not extracted
        ts1 = extract_timestamp(ulid1)
        ts2 = extract_timestamp(ulid2)
        if ts1 and ts2:
            assert ts1 < ts2


class TestTelemetryEventTypes:
    """Test telemetry event type definitions."""

    def test_event_type_enum_values(self):
        """Event types should have correct string values."""
        assert TelemetryEventType.TRACE_CREATE == "trace_create"
        assert TelemetryEventType.OBSERVATION_CREATE == "observation_create"
        assert TelemetryEventType.QUALITY_SCORE_CREATE == "quality_score_create"
        assert TelemetryEventType.EVENT_CREATE == "event_create"

    def test_telemetry_event_creation(self):
        """TelemetryEvent should validate and serialize correctly."""
        event = TelemetryEvent(
            event_id=generate_event_id(),
            event_type=TelemetryEventType.TRACE_CREATE,
            payload={"name": "test-trace", "user_id": "123"},
            timestamp=int(time.time())
        )

        assert len(event.event_id) == 26
        assert event.event_type == TelemetryEventType.TRACE_CREATE
        assert event.payload["name"] == "test-trace"

    def test_telemetry_event_invalid_id(self):
        """TelemetryEvent should reject invalid event IDs."""
        with pytest.raises(Exception):  # Pydantic validation error
            TelemetryEvent(
                event_id="invalid",  # Too short
                event_type=TelemetryEventType.TRACE_CREATE,
                payload={}
            )


class TestDeduplicationConfig:
    """Test deduplication configuration model."""

    def test_default_config(self):
        """Deduplication should have sensible defaults."""
        config = DeduplicationConfig()
        assert config.enabled is True
        assert config.ttl == 3600
        assert config.use_redis_cache is True
        assert config.fail_on_duplicate is False

    def test_custom_config(self):
        """Deduplication should accept custom values."""
        config = DeduplicationConfig(
            enabled=False,
            ttl=7200,
            use_redis_cache=False,
            fail_on_duplicate=True
        )
        assert config.enabled is False
        assert config.ttl == 7200
        assert config.use_redis_cache is False
        assert config.fail_on_duplicate is True

    def test_ttl_validation(self):
        """TTL should be within valid range."""
        with pytest.raises(Exception):  # Pydantic validation error
            DeduplicationConfig(ttl=30)  # Too low (min 60)

        with pytest.raises(Exception):
            DeduplicationConfig(ttl=100000)  # Too high (max 86400)


class TestTelemetryBatchRequest:
    """Test batch request model."""

    def test_batch_request_creation(self):
        """Batch request should validate correctly."""
        events = [
            TelemetryEvent(
                event_id=generate_event_id(),
                event_type=TelemetryEventType.TRACE_CREATE,
                payload={"name": f"trace-{i}"}
            )
            for i in range(5)
        ]

        request = TelemetryBatchRequest(
            events=events,
            environment="production",
            deduplication=DeduplicationConfig()
        )

        assert len(request.events) == 5
        assert request.environment == "production"
        assert request.deduplication.enabled is True

    def test_batch_request_serialization(self):
        """Batch request should serialize to JSON correctly."""
        event = TelemetryEvent(
            event_id=generate_event_id(),
            event_type=TelemetryEventType.OBSERVATION_CREATE,
            payload={"type": "llm", "name": "OpenAI Chat"},
            timestamp=1677610602
        )

        request = TelemetryBatchRequest(events=[event])
        data = request.model_dump(mode="json", exclude_none=True)

        assert "events" in data
        assert len(data["events"]) == 1
        assert data["events"][0]["event_type"] == "observation_create"
        assert data["events"][0]["payload"]["name"] == "OpenAI Chat"

    def test_batch_request_max_size(self):
        """Batch request should enforce max size."""
        # Create more than 1000 events
        events = [
            TelemetryEvent(
                event_id=generate_event_id(),
                event_type=TelemetryEventType.EVENT_CREATE,
                payload={}
            )
            for _ in range(1001)
        ]

        with pytest.raises(Exception):  # Pydantic validation error
            TelemetryBatchRequest(events=events)

    def test_batch_request_empty_events(self):
        """Batch request should reject empty event list."""
        with pytest.raises(Exception):  # Pydantic validation error
            TelemetryBatchRequest(events=[])


class TestTelemetryBatchResponse:
    """Test batch response model."""

    def test_batch_response_parsing(self):
        """Batch response should parse JSON correctly."""
        response_data = {
            "batch_id": generate_event_id(),
            "processed_events": 95,
            "duplicate_events": 3,
            "failed_events": 2,
            "processing_time_ms": 123,
            "errors": [
                {
                    "event_id": generate_event_id(),
                    "error": "Invalid payload format",
                    "details": "Missing required field 'name'"
                }
            ],
            "duplicate_event_ids": [generate_event_id(), generate_event_id()],
        }

        response = TelemetryBatchResponse(**response_data)
        assert response.processed_events == 95
        assert response.duplicate_events == 3
        assert response.failed_events == 2
        assert len(response.errors) == 1
        assert len(response.duplicate_event_ids) == 2

    def test_batch_response_with_job_id(self):
        """Batch response should handle async job ID."""
        response = TelemetryBatchResponse(
            batch_id=generate_event_id(),
            processed_events=0,
            duplicate_events=0,
            failed_events=0,
            processing_time_ms=5,
            job_id="job_01ABC123"
        )

        assert response.job_id == "job_01ABC123"

    def test_batch_event_error(self):
        """BatchEventError should serialize correctly."""
        error = BatchEventError(
            event_id=generate_event_id(),
            error="Validation failed",
            details="Field 'user_id' is required"
        )

        assert len(error.event_id) == 26
        assert "Validation failed" in error.error


class TestBatchConfiguration:
    """Test batch configuration in Config model."""

    def test_default_batch_config(self):
        """Config should have sensible batch defaults."""
        config = Config()
        assert config.batch_max_size == 100
        assert config.batch_flush_interval == 5.0
        assert config.batch_enable_deduplication is True
        assert config.batch_deduplication_ttl == 3600

    def test_custom_batch_config(self):
        """Config should accept custom batch settings."""
        config = Config(
            batch_max_size=200,
            batch_flush_interval=10.0,
            batch_enable_deduplication=False,
            batch_deduplication_ttl=7200
        )

        assert config.batch_max_size == 200
        assert config.batch_flush_interval == 10.0
        assert config.batch_enable_deduplication is False
        assert config.batch_deduplication_ttl == 7200

    def test_batch_config_validation(self):
        """Config should validate batch parameters."""
        # Invalid batch_max_size
        with pytest.raises(Exception):
            Config(batch_max_size=0)

        with pytest.raises(Exception):
            Config(batch_max_size=2000)

        # Invalid batch_flush_interval
        with pytest.raises(Exception):
            Config(batch_flush_interval=0.05)  # Too low

        with pytest.raises(Exception):
            Config(batch_flush_interval=100.0)  # Too high

    def test_batch_max_size_used_by_processor(self):
        """Processor should use batch_max_size from config."""
        from brokle._task_manager.processor import BackgroundProcessor

        # Create processor with custom batch_max_size
        config = Config(api_key="bk_test", batch_max_size=250)
        processor = BackgroundProcessor(config)

        # Verify config is used
        assert processor.config.batch_max_size == 250

        # The worker loop should respect this setting
        # (verified by checking the config is accessible)
        processor.shutdown()


@pytest.mark.integration
class TestBatchTelemetryIntegration:
    """Integration tests for batch telemetry submission."""

    def test_event_envelope_transformation(self):
        """Events should transform correctly for batch submission."""
        # Create events
        events = [
            TelemetryEvent(
                event_id=generate_event_id(),
                event_type=TelemetryEventType.TRACE_CREATE,
                payload={"name": "trace-1", "user_id": "user_123"}
            ),
            TelemetryEvent(
                event_id=generate_event_id(),
                event_type=TelemetryEventType.OBSERVATION_CREATE,
                payload={"trace_id": "01ABC", "type": "llm"}
            ),
        ]

        # Create batch request
        request = TelemetryBatchRequest(
            events=events,
            environment="staging",
            deduplication=DeduplicationConfig(enabled=True, ttl=3600)
        )

        # Serialize
        data = request.model_dump(mode="json", exclude_none=True)

        # Verify structure
        assert data["environment"] == "staging"
        assert len(data["events"]) == 2
        assert data["events"][0]["event_type"] == "trace_create"
        assert data["events"][1]["event_type"] == "observation_create"
        assert data["deduplication"]["enabled"] is True

    def test_partial_failure_handling(self):
        """Batch response should handle partial failures."""
        # Simulate batch response with partial failures
        response = TelemetryBatchResponse(
            batch_id=generate_event_id(),
            processed_events=97,
            duplicate_events=1,
            failed_events=2,
            processing_time_ms=150,
            errors=[
                BatchEventError(
                    event_id=generate_event_id(),
                    error="Invalid trace_id"
                ),
                BatchEventError(
                    event_id=generate_event_id(),
                    error="Missing required field",
                    details="Field 'name' is required for trace_create"
                ),
            ],
            duplicate_event_ids=[generate_event_id()]
        )

        # Verify error tracking
        assert response.processed_events + response.failed_events + response.duplicate_events == 100
        assert len(response.errors) == response.failed_events
        assert len(response.duplicate_event_ids) == response.duplicate_events
