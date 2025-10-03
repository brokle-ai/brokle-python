# Configuration Cleanup: Removed Redundant Telemetry Settings

## Summary

Removed redundant legacy configuration fields in favor of cleaner batch API naming. Since there are no existing SDK users, this was done as a breaking change for long-term API clarity.

## Changes Made

### ❌ Removed (Legacy Config)

| Field | Type | Default | Replaced By |
|-------|------|---------|-------------|
| `telemetry_batch_size` | int | 100 | `batch_max_size` |
| `telemetry_flush_interval` | int | 10000ms | `batch_flush_interval` |
| `BROKLE_TELEMETRY_BATCH_SIZE` | env var | - | `BROKLE_BATCH_MAX_SIZE` |
| `BROKLE_TELEMETRY_FLUSH_INTERVAL` | env var | - | `BROKLE_BATCH_FLUSH_INTERVAL` |

### ✅ Kept (Clean Batch API Config)

| Field | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `batch_max_size` | int | 100 | 1-1000 | Maximum events per batch |
| `batch_flush_interval` | float | 5.0 | 0.1-60.0 | Flush interval in seconds |
| `batch_enable_deduplication` | bool | True | - | Enable ULID deduplication |
| `batch_deduplication_ttl` | int | 3600 | 60-86400 | Dedup cache TTL (seconds) |
| `batch_use_redis_cache` | bool | True | - | Use Redis for distributed dedup |
| `batch_fail_on_duplicate` | bool | False | - | Fail batch on duplicates |

---

## Files Modified

### 1. `brokle/config.py`
**Removed:**
- `telemetry_batch_size` field definition
- `telemetry_flush_interval` field definition
- `BROKLE_TELEMETRY_BATCH_SIZE` environment variable loading
- `BROKLE_TELEMETRY_FLUSH_INTERVAL` environment variable loading

**Result:**
- Only `batch_*` configuration fields remain
- Cleaner, more consistent API

### 2. `brokle/observability/context.py`
**Updated function signature:**
```python
# BEFORE
def set_brokle_context(
    telemetry_batch_size: Optional[int] = None,
    telemetry_flush_interval: Optional[int] = None,
    ...
)

# AFTER
def set_brokle_context(
    batch_max_size: Optional[int] = None,
    batch_flush_interval: Optional[float] = None,
    ...
)
```

**Updated 4 locations:**
1. Function parameters (2 fields)
2. Docstring descriptions (2 fields)
3. Explicit config check (2 references)
4. Client kwargs mapping (2 references)

### 3. `tests/test_task_manager.py`
**Updated 9 test references:**
```python
# BEFORE
config.telemetry_batch_size = 10

# AFTER
config.batch_max_size = 10
```

All tests updated to use new configuration.

### 4. `MIGRATION_BATCH_API.md`
**Added configuration table** with all batch settings and their ranges.

---

## Configuration Before → After

### Before (Confusing - Two Overlapping Systems)
```python
Brokle(
    # Legacy names (from /api/v1/telemetry/bulk)
    telemetry_batch_size=100,      # ❌ Which one to use?
    telemetry_flush_interval=5000, # ❌ Milliseconds? Seconds?

    # New names (from /v1/telemetry/batch)
    batch_max_size=200,            # ❌ Which one to use?
    batch_flush_interval=10.0,     # ❌ Which one to use?
)
```

### After (Clean - Single Consistent System)
```python
Brokle(
    # Clear batch API configuration
    batch_max_size=200,           # ✅ Events per batch (1-1000)
    batch_flush_interval=10.0,    # ✅ Flush interval in seconds (0.1-60.0)
    batch_enable_deduplication=True,
    batch_deduplication_ttl=3600,
)
```

---

## Benefits

### 1. **Clarity**
- ✅ Only one way to configure batch size
- ✅ Only one way to configure flush interval
- ✅ No confusion about which setting to use

### 2. **Better Naming**
- ✅ `batch_max_size` is more descriptive than `telemetry_batch_size`
- ✅ `batch_flush_interval` (seconds) is clearer than `telemetry_flush_interval` (milliseconds)
- ✅ Consistent `batch_*` prefix for all batch settings

### 3. **Consistency**
- ✅ All batch settings use the same naming convention
- ✅ All batch settings documented in one place
- ✅ Aligned with batch API architecture

### 4. **Simplicity**
- ✅ Fewer configuration fields to understand
- ✅ Less documentation to maintain
- ✅ Cleaner API surface

### 5. **Future-Proof**
- ✅ Clean foundation for future batch features
- ✅ No legacy baggage
- ✅ Better long-term maintainability

---

## Test Results

### Before Cleanup
- **63 tests** passing (batch + task_manager + client)

### After Cleanup
- **240 tests** passing (full test suite)
- ✅ All configuration tests updated
- ✅ All integration tests working
- ✅ No regressions

---

## Migration Guide (For Future Users)

If we ever have users with the old config (we don't), the migration is simple:

### Old Code (Would Have Been)
```python
client = Brokle(
    telemetry_batch_size=150,
    telemetry_flush_interval=8000,  # milliseconds
)
```

### New Code
```python
client = Brokle(
    batch_max_size=150,
    batch_flush_interval=8.0,  # seconds
)
```

**Key Difference:** Flush interval changed from **milliseconds to seconds** for better clarity.

---

## Implementation Details

### Processor Usage
The `BackgroundProcessor` now uses `batch_max_size` directly:

```python
# In _worker_loop():
batch_size = self.config.batch_max_size  # Uses config value (1-1000)
```

No more hard-coded `min(..., 100)` limit.

### Environment Variables
```bash
# Old (removed)
export BROKLE_TELEMETRY_BATCH_SIZE=150
export BROKLE_TELEMETRY_FLUSH_INTERVAL=8000

# New (recommended)
export BROKLE_BATCH_MAX_SIZE=150
export BROKLE_BATCH_FLUSH_INTERVAL=8.0
```

---

## Validation

All configuration fields have proper validation:

```python
# batch_max_size
ge=1, le=1000  # Must be between 1 and 1000

# batch_flush_interval
ge=0.1, le=60.0  # Must be between 0.1 and 60.0 seconds

# batch_deduplication_ttl
ge=60, le=86400  # Must be between 60 and 86400 seconds (1 min to 24 hours)
```

---

## Summary

This cleanup removed 2 redundant configuration fields in favor of cleaner batch API naming:

- **Removed:** `telemetry_batch_size` → Use `batch_max_size`
- **Removed:** `telemetry_flush_interval` → Use `batch_flush_interval`

**Result:**
- ✅ Simpler, clearer API
- ✅ Better long-term maintainability
- ✅ Aligned with batch API architecture
- ✅ No breaking changes (no users yet)
- ✅ 240 tests passing

The SDK now has a clean, consistent configuration API built for the long term.
