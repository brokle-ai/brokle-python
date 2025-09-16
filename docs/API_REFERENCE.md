# Auto-Instrumentation API Reference

Complete API reference for Brokle's Auto-Instrumentation system.

## Core Functions

### `auto_instrument(libraries=None, exclude=None) -> Dict[str, bool]`

Automatically instrument available LLM libraries.

**Parameters:**
- `libraries` (Optional[List[str]]): Specific libraries to instrument. If None, instruments all available.
- `exclude` (Optional[List[str]]): Libraries to exclude from instrumentation.

**Returns:**
- `Dict[str, bool]`: Mapping of library names to instrumentation success status.

**Example:**
```python
# Instrument all available libraries
results = brokle_ai.auto_instrument()

# Instrument specific libraries only
results = brokle_ai.auto_instrument(libraries=["openai", "anthropic"])

# Instrument all except specific libraries
results = brokle_ai.auto_instrument(exclude=["langchain"])
```

### `instrument(library: str) -> bool`

Instrument a specific library.

**Parameters:**
- `library` (str): Name of the library to instrument ("openai", "anthropic", "langchain").

**Returns:**
- `bool`: True if instrumentation succeeded, False otherwise.

**Example:**
```python
success = brokle_ai.instrument("openai")
if success:
    print("OpenAI instrumentation enabled")
```

### `uninstrument(library: str) -> bool`

Remove instrumentation from a specific library.

**Parameters:**
- `library` (str): Name of the library to uninstrument.

**Returns:**
- `bool`: True if uninstrumentation succeeded, False otherwise.

**Example:**
```python
success = brokle_ai.uninstrument("openai")
```

## Status & Health Functions

### `print_status() -> None`

Print visual instrumentation status for all libraries.

**Example:**
```python
brokle_ai.print_status()
"""
=== Brokle Auto-Instrumentation Status ===
✅ openai (auto): instrumented
⚪ anthropic (auto): available
❌ langchain (auto): not_available

📊 Health Summary:
   Overall Health: 100%
   Libraries: 1/2 instrumented, 3/3 healthy
"""
```

### `get_status() -> Dict[str, InstrumentationStatus]`

Get programmatic instrumentation status.

**Returns:**
- `Dict[str, InstrumentationStatus]`: Mapping of library names to their status.

**Status Values:**
- `InstrumentationStatus.INSTRUMENTED`: Library is actively instrumented
- `InstrumentationStatus.AVAILABLE`: Library is available but not instrumented
- `InstrumentationStatus.NOT_AVAILABLE`: Library is not installed
- `InstrumentationStatus.FAILED`: Instrumentation failed

**Example:**
```python
status = brokle_ai.get_status()
for library, lib_status in status.items():
    print(f"{library}: {lib_status.value}")
```

### `print_health_report() -> None`

Print detailed health report with error information and circuit breaker states.

**Example:**
```python
brokle_ai.print_health_report()
"""
=== Brokle Auto-Instrumentation Health Report ===

🏥 Overall Health Score: 85%
   📊 Libraries: 3 total, 2 available
   ✅ Status: 2 instrumented, 2 healthy

⚡ Circuit Breaker Status:
   🟢 openai: closed
   🔴 anthropic: open
   🟢 langchain: closed

❗ Recent Errors:
   anthropic.instrument: 3 errors
"""
```

### `get_health_report() -> Dict[str, Any]`

Get comprehensive health report programmatically.

**Returns:**
- `Dict[str, Any]`: Health report with overall metrics, library details, and error summary.

**Report Structure:**
```python
{
    "overall_health": {
        "score": float,  # 0-100 health score
        "total_libraries": int,
        "healthy_libraries": int,
        "instrumented_libraries": int,
        "available_libraries": int
    },
    "library_details": {
        "library_name": {
            "status": str,
            "description": str,
            "auto_instrument": bool,
            "available": bool,
            "instrumented": bool,
            "healthy": bool,
            "error_summary": dict
        }
    },
    "error_summary": {
        "error_counts": dict,
        "circuit_breaker_states": dict,
        "last_errors": dict
    },
    "circuit_breaker_states": dict
}
```

**Example:**
```python
health = brokle_ai.get_health_report()
print(f"Overall health: {health['overall_health']['score']}%")

for lib, details in health['library_details'].items():
    if not details['healthy']:
        print(f"⚠️ {lib} has issues")
```

## Error Management Functions

### `reset_all_errors() -> None`

Reset all error tracking and circuit breakers.

**Example:**
```python
# Reset all errors to clean state
brokle_ai.reset_all_errors()

# Verify health improved
health = brokle_ai.get_health_report()
print(f"Health after reset: {health['overall_health']['score']}%")
```

### `get_error_handler() -> InstrumentationErrorHandler`

Get the global error handler for advanced error management.

**Returns:**
- `InstrumentationErrorHandler`: Error handler instance for advanced operations.

**Example:**
```python
error_handler = brokle_ai.get_error_handler()

# Check if specific operation is healthy
is_healthy = error_handler.is_operation_healthy("openai", "instrument")

# Get detailed error summary
error_summary = error_handler.get_error_summary()

# Reset specific library errors
error_handler.reset_errors("openai")
```

## Registry Functions

### `get_registry() -> InstrumentationRegistry`

Get the global instrumentation registry.

**Returns:**
- `InstrumentationRegistry`: Registry instance for advanced operations.

**Example:**
```python
registry = brokle_ai.get_registry()

# Get available libraries
available = registry.get_available_libraries()
print(f"Available libraries: {available}")

# Get instrumented libraries
instrumented = registry.get_instrumented_libraries()
print(f"Instrumented libraries: {instrumented}")
```

## Classes

### `InstrumentationRegistry`

Central registry for managing library instrumentation.

#### Methods

##### `list_libraries() -> List[str]`
List all registered library names.

##### `get_available_libraries() -> List[str]`
Get list of available (installed) libraries.

##### `get_instrumented_libraries() -> List[str]`
Get list of currently instrumented libraries.

##### `instrument_library(name: str) -> bool`
Instrument a specific library by name.

##### `uninstrument_library(name: str) -> bool`
Remove instrumentation from a library by name.

##### `get_instrumentation_summary() -> Dict[str, Dict[str, Any]]`
Get detailed summary of all library instrumentation states.

**Example:**
```python
registry = brokle_ai.get_registry()

# List all libraries
all_libs = registry.list_libraries()

# Get summary
summary = registry.get_instrumentation_summary()
for lib, details in summary.items():
    print(f"{lib}: {details['status']} - {details['description']}")
```

### `InstrumentationErrorHandler`

Handles errors and circuit breaker functionality.

#### Methods

##### `is_operation_healthy(library: str, operation: str) -> bool`
Check if a specific operation is healthy (circuit breaker not open).

##### `get_error_summary(library: Optional[str] = None) -> Dict[str, Any]`
Get error summary for all operations or specific library.

##### `reset_errors(library: str, operation: Optional[str] = None) -> None`
Reset error tracking for library or specific operation.

##### `get_circuit_breaker(operation: str) -> CircuitBreaker`
Get circuit breaker instance for an operation.

**Example:**
```python
error_handler = brokle_ai.get_error_handler()

# Check health
if error_handler.is_operation_healthy("openai", "instrument"):
    print("OpenAI instrumentation is healthy")

# Get circuit breaker
cb = error_handler.get_circuit_breaker("openai.instrument")
print(f"Circuit breaker state: {cb.state}")
```

## Error Classes

### `InstrumentationError`

Base exception for instrumentation-related errors.

**Attributes:**
- `message` (str): Error message
- `severity` (ErrorSeverity): Error severity level
- `library` (Optional[str]): Library name
- `operation` (Optional[str]): Operation name
- `original_error` (Optional[Exception]): Original exception that caused this error
- `timestamp` (datetime): When the error occurred

### `LibraryNotAvailableError`

Raised when a library is not available for instrumentation.

### `ObservabilityError`

Raised when observability operations fail.

### `ConfigurationError`

Raised when configuration is invalid.

### `ErrorSeverity`

Enumeration of error severity levels.

**Values:**
- `ErrorSeverity.CRITICAL`: Complete instrumentation failure
- `ErrorSeverity.HIGH`: Partial failure, fallback to basic instrumentation
- `ErrorSeverity.MEDIUM`: Recoverable error, retry with exponential backoff
- `ErrorSeverity.LOW`: Minor issue, log and continue
- `ErrorSeverity.DEBUG`: Development/debugging information

**Example:**
```python
from brokle.auto_instrumentation import InstrumentationError, ErrorSeverity

try:
    brokle_ai.instrument("nonexistent_library")
except InstrumentationError as e:
    print(f"Error: {e.message}")
    print(f"Severity: {e.severity.value}")
    print(f"Library: {e.library}")
    print(f"Operation: {e.operation}")
    print(f"Timestamp: {e.timestamp}")
```

## Decorators

### `@safe_operation(library: str, operation: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM)`

Decorator for safe instrumentation operations with error handling and circuit breaker protection.

**Parameters:**
- `library` (str): Library name for error tracking
- `operation` (str): Operation name for error tracking
- `severity` (ErrorSeverity): Error severity level

**Example:**
```python
from brokle.auto_instrumentation.error_handlers import safe_operation, ErrorSeverity

@safe_operation("my_lib", "custom_operation", ErrorSeverity.LOW)
def my_instrumentation_function():
    # Your instrumentation code here
    # Errors will be handled gracefully
    return True
```

### `@safe_async_operation(library: str, operation: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM)`

Async version of `@safe_operation`.

**Example:**
```python
from brokle.auto_instrumentation.error_handlers import safe_async_operation

@safe_async_operation("my_lib", "async_operation")
async def my_async_function():
    # Async instrumentation code
    await some_async_operation()
    return True
```

## Context Managers

### `instrumentation_context(library: str, operation: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM)`

Context manager for safe instrumentation operations.

**Parameters:**
- `library` (str): Library name
- `operation` (str): Operation name
- `severity` (ErrorSeverity): Error severity level

**Example:**
```python
from brokle.auto_instrumentation.error_handlers import instrumentation_context, ErrorSeverity

with instrumentation_context("openai", "setup", ErrorSeverity.HIGH):
    # Setup code that might fail
    setup_openai_instrumentation()
    # Errors are handled based on severity level
```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `BROKLE_PUBLIC_KEY` | Brokle public key | None | Yes |
| `BROKLE_BASE_URL` | API base URL | `https://api.brokle.ai` | No |
| `BROKLE_ORGANIZATION_ID` | Organization ID | None | Yes |
| `BROKLE_SECRET_KEY` | Secret key | None | Yes |
| `BROKLE_ENVIRONMENT` | Environment name | `production` | No |
| `BROKLE_AUTO_INSTRUMENT` | Auto-instrument on import | `true` | No |
| `BROKLE_CIRCUIT_BREAKER_ENABLED` | Enable circuit breakers | `true` | No |

### Programmatic Configuration

```python
from brokle.config import BrokleConfig

config = BrokleConfig(
    public_key="your-public-key",
    base_url="https://api.brokle.ai",
    organization_id="org_123",
    secret_key="sk_your_secret_key",
    environment="production",
    auto_instrument=True,
    circuit_breaker_enabled=True,
    max_retries=3,
    timeout_seconds=30
)

# Apply configuration
import brokle.auto_instrumentation as brokle_ai
brokle_ai.configure(config)
```

## Constants

### Library Names
```python
SUPPORTED_LIBRARIES = [
    "openai",      # OpenAI Python library
    "anthropic",   # Anthropic Python library
    "langchain"    # LangChain framework
]
```

### Default Settings
```python
DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD = 3
DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 30  # seconds
DEFAULT_RETRY_MAX_ATTEMPTS = 2
DEFAULT_RETRY_BASE_DELAY = 1.0  # seconds
DEFAULT_RETRY_MAX_DELAY = 10.0  # seconds
```

## Type Hints

```python
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime

# Status type
class InstrumentationStatus(Enum):
    NOT_AVAILABLE = "not_available"
    AVAILABLE = "available"
    INSTRUMENTED = "instrumented"
    FAILED = "failed"

# Function signatures
def auto_instrument(
    libraries: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None
) -> Dict[str, bool]: ...

def get_status() -> Dict[str, InstrumentationStatus]: ...

def get_health_report() -> Dict[str, Any]: ...
```

## Best Practices

### 1. **Initialize Early**
```python
# At the top of your main application file
import brokle.auto_instrumentation as brokle_ai
brokle_ai.auto_instrument()

# Then import your LLM libraries
import openai
import anthropic
```

### 2. **Check Health Regularly**
```python
def periodic_health_check():
    health = brokle_ai.get_health_report()
    if health["overall_health"]["score"] < 80:
        logger.warning(f"Instrumentation health degraded: {health}")
        # Consider alerting or recovery actions
```

### 3. **Handle Errors Gracefully**
```python
try:
    results = brokle_ai.auto_instrument()
    failed_libraries = [lib for lib, success in results.items() if not success]
    if failed_libraries:
        logger.warning(f"Failed to instrument: {failed_libraries}")
except Exception as e:
    logger.error(f"Auto-instrumentation failed: {e}")
    # Application continues without instrumentation
```

### 4. **Use Appropriate Error Handling**
```python
from brokle.auto_instrumentation import LibraryNotAvailableError

try:
    brokle_ai.instrument("optional_library")
except LibraryNotAvailableError:
    logger.info("Optional library not available, continuing...")
except Exception as e:
    logger.error(f"Unexpected instrumentation error: {e}")
```

---

*Complete API reference for Brokle Auto-Instrumentation v1.0+*