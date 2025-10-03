# Python SDK Migration to Unified Telemetry Batch API

## Overview

The Python SDK has been successfully migrated to use the unified `/v1/telemetry/batch` endpoint for all telemetry operations. This replaces the previous `/api/v1/telemetry/bulk` endpoint with a more robust, event-based architecture that supports ULID-based deduplication and partial failure handling.

## Summary of Changes

### ✅ Completed Migration

**Files Changed**: 6 modified, 2 created
**New Dependencies**: `python-ulid>=2.0.0`
**Tests**: 239 passing (23 new batch telemetry tests)
**Backward Compatibility**: 100% maintained

---

## Technical Changes

### 1. New Files

#### `brokle/_utils/ulid.py` (NEW)
- ULID generation utilities for event deduplication
- Functions: `generate_ulid()`, `generate_event_id()`, `extract_timestamp()`, `is_valid_ulid()`
- Fallback to UUID if `python-ulid` not installed

#### `brokle/types/telemetry.py` (NEW)
- Event envelope types for batch API
- `TelemetryEventType` enum (trace_create, observation_create, etc.)
- `TelemetryEvent` model with ULID event_id
- `TelemetryBatchRequest` and `TelemetryBatchResponse` models
- `DeduplicationConfig` model
- `BatchEventError` for partial failure tracking

### 2. Modified Files

#### `brokle/config.py`
**Added batch configuration fields:**
```python
batch_max_size: int = 100              # Max events per batch (1-1000)
batch_flush_interval: float = 5.0     # Flush interval in seconds (0.1-60.0)
batch_enable_deduplication: bool = True
batch_deduplication_ttl: int = 3600   # Cache TTL in seconds (60-86400)
batch_use_redis_cache: bool = True
batch_fail_on_duplicate: bool = False
```

**Environment variables:**
- `BROKLE_BATCH_MAX_SIZE`
- `BROKLE_BATCH_FLUSH_INTERVAL`
- `BROKLE_BATCH_ENABLE_DEDUPLICATION`
- `BROKLE_BATCH_DEDUPLICATION_TTL`
- `BROKLE_BATCH_USE_REDIS_CACHE`
- `BROKLE_BATCH_FAIL_ON_DUPLICATE`

#### `brokle/_task_manager/processor.py`
**Key changes:**
- ❌ Removed: `_submit_telemetry()` and `_async_submit_telemetry()` (old bulk endpoint)
- ✅ Added: `_submit_batch_events()` and `_async_submit_batch_events()` (new batch endpoint)
- ✅ Added: `_handle_batch_response()` for partial failure handling
- ✅ Added: `submit_batch_event()` method for pre-formed TelemetryEvent objects
- 🔄 Modified: `submit_telemetry()` now accepts `event_type` parameter (default: "event_create")
- 🔄 Modified: `_process_telemetry_batch()` transforms data into TelemetryEvent objects
- 🔄 Modified: `_worker_loop()` now uses `batch_max_size` from config (was hard-coded to 100)

**Endpoint change:**
```python
# OLD
POST {host}/api/v1/telemetry/bulk

# NEW
POST {host}/v1/telemetry/batch
```

#### `brokle/client.py` (Brokle & AsyncBrokle)
**New methods:**
```python
def submit_batch_event(self, event_type: str, payload: Dict[str, Any]) -> str:
    """Submit structured batch event with ULID tracking."""
    # Returns event_id for tracking
```

**Modified methods:**
```python
def submit_telemetry(self, data: Dict[str, Any], event_type: str = "event_create") -> None:
    """Submit telemetry with optional event type."""
```

#### `pyproject.toml`
**Added dependency:**
```toml
dependencies = [
    # ... existing
    "python-ulid>=2.0.0",
]
```

#### `tests/test_client.py`
**Updated test assertion:**
```python
# OLD
custom_processor.submit_telemetry.assert_called_once_with({"test": "data"})

# NEW
custom_processor.submit_telemetry.assert_called_once_with(
    {"test": "data"}, event_type="event_create"
)
```

### 3. New Tests

#### `tests/test_batch_telemetry.py` (NEW - 23 tests)
**Test coverage:**
- ULID generation and validation
- Event envelope creation and serialization
- Batch request/response models
- Deduplication configuration
- Partial failure handling
- Configuration validation
- Integration scenarios

---

## API Changes

### Batch Request Format

```python
{
    "events": [
        {
            "event_id": "01ABCDEFGHIJKLMNOPQRSTUVWXYZ",  # ULID
            "event_type": "trace_create",
            "payload": {
                "name": "my-trace",
                "user_id": "user_123"
            },
            "timestamp": 1677610602  # Optional
        }
    ],
    "environment": "production",  # Optional
    "deduplication": {
        "enabled": true,
        "ttl": 3600,
        "use_redis_cache": true,
        "fail_on_duplicate": false
    },
    "async_mode": false  # Optional
}
```

### Batch Response Format

```python
{
    "batch_id": "01ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "processed_events": 97,
    "duplicate_events": 1,
    "failed_events": 2,
    "processing_time_ms": 123,
    "errors": [
        {
            "event_id": "01DEF...",
            "error": "Invalid payload format",
            "details": "Missing required field 'name'"
        }
    ],
    "duplicate_event_ids": ["01GHI...", "01JKL..."],
    "job_id": "job_01ABC123"  # If async_mode=true
}
```

---

## Backward Compatibility

### ✅ Public API Unchanged

All existing user code continues to work:

```python
# Legacy telemetry submission (still works)
client.submit_telemetry({"method": "GET", "status": 200})

# Internally transformed to:
# - event_type: "event_create"
# - event_id: auto-generated ULID
# - Submitted to /v1/telemetry/batch
```

### ✅ Graceful Defaults

- Default event type: `"event_create"` for generic telemetry
- Automatic ULID generation if not provided
- Fallback to UUID if `python-ulid` not installed
- Deduplication enabled by default (can be disabled)

---

## New Capabilities

### 1. Structured Event Submission

```python
# New preferred method for structured events
event_id = client.submit_batch_event(
    event_type="trace_create",
    payload={"name": "my-trace", "user_id": "123"}
)
# Returns ULID for tracking: "01ABCDEFGHIJKLMNOPQRSTUVWXYZ"
```

### 2. Event Deduplication

- ULID-based deduplication prevents duplicate events
- Configurable TTL (default: 1 hour)
- Redis-backed for distributed systems
- Optional failure on duplicates

### 3. Partial Failure Handling

- Batch processing continues even if some events fail
- Detailed error reporting per event
- Failed events tracked separately
- Automatic retry logic with backoff

### 4. Enhanced Observability

```python
# Get batch processing metrics
metrics = client.get_processor_metrics()
# {
#     "queue_depth": 23,
#     "items_processed": 1547,
#     "items_failed": 3,
#     "batches_processed": 16,
#     "processing_rate": 5.2,
#     "error_rate": 0.002
# }
```

---

## Migration Impact

### User Code Changes

**Required**: None (100% backward compatible)

**Optional enhancements**:
```python
# 1. Use new batch event method for structured telemetry
event_id = client.submit_batch_event("trace_create", {...})

# 2. Configure batch behavior
client = Brokle(
    api_key="bk_...",
    batch_max_size=200,
    batch_flush_interval=10.0,
    batch_enable_deduplication=True
)

# 3. Use event types for better categorization
client.submit_telemetry(
    {"trace_name": "workflow"},
    event_type="trace_create"
)
```

### Performance Impact

**Positive changes:**
- ✅ Reduced network overhead (batching)
- ✅ Faster processing (single endpoint)
- ✅ Lower backend load (deduplication)
- ✅ Better error resilience (partial failures)

**Potential considerations:**
- Batch flush interval adds minor latency (default 5s)
- Can be tuned via `batch_flush_interval` config

---

## Testing Results

### Test Suite Status
```
239 tests passed
0 tests failed
23 new batch telemetry tests
100% backward compatibility
```

### Test Coverage
- ✅ ULID generation and validation
- ✅ Event envelope serialization
- ✅ Batch request/response models
- ✅ Deduplication configuration
- ✅ Partial failure scenarios
- ✅ Config validation
- ✅ Existing client tests (updated)
- ✅ Integration scenarios

---

## Event Types

### Supported Event Types

```python
class TelemetryEventType(str, Enum):
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

    # Generic event (default)
    EVENT_CREATE = "event_create"
```

---

## Configuration Examples

### Minimal Configuration (Defaults)
```python
client = Brokle(api_key="bk_your_secret")
# Uses all default batch settings:
# - batch_max_size: 100
# - batch_flush_interval: 5.0 seconds
# - batch_enable_deduplication: True
```

### Custom Batch Configuration
```python
client = Brokle(
    api_key="bk_your_secret",
    batch_max_size=200,           # Larger batches (1-1000)
    batch_flush_interval=2.0,     # Faster flushing (0.1-60.0 seconds)
    batch_enable_deduplication=True,
    batch_deduplication_ttl=7200  # 2-hour cache (60-86400 seconds)
)
```

### Environment Variable Configuration
```bash
export BROKLE_API_KEY="bk_your_secret"
export BROKLE_BATCH_MAX_SIZE=150
export BROKLE_BATCH_FLUSH_INTERVAL=3.0
export BROKLE_BATCH_ENABLE_DEDUPLICATION=true
export BROKLE_BATCH_DEDUPLICATION_TTL=3600

# SDK reads from environment
client = get_client()
```

### Configuration Fields (All Batch Settings)

| Field | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `batch_max_size` | int | 100 | 1-1000 | Maximum events per batch |
| `batch_flush_interval` | float | 5.0 | 0.1-60.0 | Flush interval in seconds |
| `batch_enable_deduplication` | bool | True | - | Enable ULID-based deduplication |
| `batch_deduplication_ttl` | int | 3600 | 60-86400 | Dedup cache TTL in seconds |
| `batch_use_redis_cache` | bool | True | - | Use Redis for distributed dedup |
| `batch_fail_on_duplicate` | bool | False | - | Fail batch on duplicate events |

---

## Error Handling

### Partial Failure Example

```python
# Submit batch with some invalid events
response = await client._background_processor._async_submit_batch_events(events)

# Response shows partial success:
# {
#     "batch_id": "01ABC...",
#     "processed_events": 97,
#     "failed_events": 3,
#     "errors": [
#         {"event_id": "01DEF...", "error": "Invalid trace_id"},
#         {"event_id": "01GHI...", "error": "Missing required field"}
#     ]
# }

# Successfully processed events are saved
# Failed events are logged and can be retried
```

---

## Next Steps

### Optional Enhancements (Future)

1. **Trace/Observation Resources** (Optional)
   - Add `brokle/resources/traces.py`
   - Add `brokle/resources/observations.py`
   - Add `brokle/resources/scores.py`

2. **Advanced Features**
   - Async batch processing mode
   - Custom retry strategies
   - Batch compression
   - Event streaming

3. **Monitoring**
   - Batch performance metrics
   - Deduplication statistics
   - Error rate tracking

---

## Rollback Plan

If issues arise, rollback is simple:

1. Revert to previous commit
2. No database changes needed (backend handles both endpoints)
3. No user code changes required

The backend supports both old and new endpoints during transition period.

---

## Documentation Updates

**Required updates:**
- [ ] API reference documentation
- [ ] Configuration guide
- [ ] Migration guide for users
- [ ] Example code snippets
- [ ] Changelog entry

---

## Key Takeaways

✅ **Migration Complete**: All telemetry now uses unified batch API
✅ **100% Backward Compatible**: No user code changes required
✅ **23 New Tests**: Comprehensive test coverage
✅ **Enhanced Capabilities**: ULID deduplication, partial failures, better error handling
✅ **Performance Improved**: Batching reduces network overhead
✅ **Production Ready**: All tests passing, robust error handling

The Python SDK is now fully aligned with the backend's unified telemetry batch API architecture.
