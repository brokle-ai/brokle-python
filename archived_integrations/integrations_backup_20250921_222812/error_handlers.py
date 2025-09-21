"""
Comprehensive error handling utilities for auto-instrumentation.

This module provides robust error handling patterns including retry mechanisms,
circuit breakers, and graceful degradation for instrumentation failures.
"""

import functools
import logging
import time
from contextlib import contextmanager
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for instrumentation failures."""
    CRITICAL = "critical"    # Complete instrumentation failure - disable instrumentation
    HIGH = "high"           # Partial failure - fallback to basic instrumentation
    MEDIUM = "medium"       # Recoverable error - retry with exponential backoff
    LOW = "low"             # Minor issue - log and continue
    DEBUG = "debug"         # Development/debugging information


class InstrumentationError(Exception):
    """Base exception for instrumentation-related errors."""

    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 library: Optional[str] = None, operation: Optional[str] = None,
                 original_error: Optional[Exception] = None):
        self.message = message
        self.severity = severity
        self.library = library
        self.operation = operation
        self.original_error = original_error
        self.timestamp = datetime.now(timezone.utc)

        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the error message with context."""
        parts = []
        if self.library:
            parts.append(f"[{self.library}]")
        if self.operation:
            parts.append(f"({self.operation})")
        parts.append(self.message)

        if self.original_error:
            parts.append(f"- Original error: {self.original_error}")

        return " ".join(parts)


class ConfigurationError(InstrumentationError):
    """Error in instrumentation configuration."""
    pass


class LibraryNotAvailableError(InstrumentationError):
    """Library not available for instrumentation."""
    pass


class ObservabilityError(InstrumentationError):
    """Error in observability data collection."""
    pass


class CircuitBreakerError(InstrumentationError):
    """Circuit breaker triggered due to repeated failures."""
    pass


class CircuitBreaker:
    """Circuit breaker pattern for instrumentation operations."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60,
                 expected_exceptions: tuple = (Exception,)):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exceptions = expected_exceptions

        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half-open

    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self.state == "open"

    def is_half_open(self) -> bool:
        """Check if circuit breaker is half-open."""
        return self.state == "half-open"

    def should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker."""
        if self.state != "open":
            return False

        if self.last_failure_time is None:
            return True

        return datetime.now(timezone.utc) - self.last_failure_time > timedelta(seconds=self.recovery_timeout)

    def record_success(self):
        """Record a successful operation."""
        self.failure_count = 0
        self.state = "closed"
        self.last_failure_time = None

    def record_failure(self):
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)

        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures. "
                f"Will retry after {self.recovery_timeout} seconds."
            )

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call a function with circuit breaker protection."""
        if self.is_open():
            if self.should_attempt_reset():
                self.state = "half-open"
                logger.info("Circuit breaker half-open, attempting recovery...")
            else:
                raise CircuitBreakerError(
                    f"Circuit breaker is open. Last failure: {self.last_failure_time}",
                    severity=ErrorSeverity.HIGH
                )

        try:
            result = func(*args, **kwargs)
            if self.is_half_open():
                self.record_success()
                logger.info("Circuit breaker recovered successfully")
            return result

        except self.expected_exceptions as e:
            self.record_failure()
            if self.is_half_open():
                self.state = "open"
            raise


class RetryPolicy:
    """Retry policy with exponential backoff and jitter."""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0,
                 max_delay: float = 60.0, backoff_factor: float = 2.0,
                 jitter: bool = True):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt."""
        delay = min(self.base_delay * (self.backoff_factor ** attempt), self.max_delay)

        if self.jitter:
            # Add jitter to prevent thundering herd
            import random
            delay *= (0.5 + random.random() * 0.5)

        return delay

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry policy."""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                if attempt == self.max_retries:
                    break

                delay = self.calculate_delay(attempt)
                logger.debug(f"Retry attempt {attempt + 1}/{self.max_retries} "
                           f"after {delay:.2f}s delay: {e}")
                time.sleep(delay)

        # All retries failed
        raise InstrumentationError(
            f"Failed after {self.max_retries} retries",
            severity=ErrorSeverity.HIGH,
            original_error=last_exception
        )


class InstrumentationErrorHandler:
    """Centralized error handling for instrumentation operations."""

    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_counts: Dict[str, int] = {}
        self.last_errors: Dict[str, datetime] = {}
        self.retry_policies: Dict[str, RetryPolicy] = {}

    def get_circuit_breaker(self, operation: str) -> CircuitBreaker:
        """Get or create circuit breaker for operation."""
        if operation not in self.circuit_breakers:
            self.circuit_breakers[operation] = CircuitBreaker(
                failure_threshold=3,
                recovery_timeout=30
            )
        return self.circuit_breakers[operation]

    def get_retry_policy(self, operation: str) -> RetryPolicy:
        """Get or create retry policy for operation."""
        if operation not in self.retry_policies:
            self.retry_policies[operation] = RetryPolicy(
                max_retries=2,
                base_delay=1.0,
                max_delay=10.0
            )
        return self.retry_policies[operation]

    def handle_error(self, error: Exception, library: str, operation: str,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> None:
        """Handle instrumentation error with appropriate logging and tracking."""
        error_key = f"{library}.{operation}"

        # Update error tracking
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        self.last_errors[error_key] = datetime.now(timezone.utc)

        # Create instrumentation error if not already one
        if isinstance(error, InstrumentationError):
            instrumentation_error = error
        else:
            instrumentation_error = InstrumentationError(
                f"Operation failed: {str(error)}",
                severity=severity,
                library=library,
                operation=operation,
                original_error=error
            )

        # Log based on severity
        self._log_error(instrumentation_error, error_key)

        # Handle circuit breaker
        circuit_breaker = self.get_circuit_breaker(f"{library}.{operation}")
        circuit_breaker.record_failure()

    def _log_error(self, error: InstrumentationError, error_key: str) -> None:
        """Log error with appropriate level based on severity."""
        error_count = self.error_counts.get(error_key, 1)
        context = {
            "library": error.library,
            "operation": error.operation,
            "severity": error.severity.value,
            "error_count": error_count,
            "timestamp": error.timestamp.isoformat()
        }

        message = f"{error.message} (count: {error_count})"

        if error.severity == ErrorSeverity.CRITICAL:
            logger.error(message, extra=context)
        elif error.severity == ErrorSeverity.HIGH:
            logger.warning(message, extra=context)
        elif error.severity == ErrorSeverity.MEDIUM:
            logger.info(message, extra=context)
        elif error.severity == ErrorSeverity.LOW:
            logger.debug(message, extra=context)
        else:  # DEBUG
            logger.debug(message, extra=context)

    def is_operation_healthy(self, library: str, operation: str) -> bool:
        """Check if operation is healthy (circuit breaker not open)."""
        circuit_breaker = self.get_circuit_breaker(f"{library}.{operation}")
        return not circuit_breaker.is_open()

    def reset_errors(self, library: str, operation: Optional[str] = None) -> None:
        """Reset error tracking for library or specific operation."""
        if operation:
            error_key = f"{library}.{operation}"
            self.error_counts.pop(error_key, None)
            self.last_errors.pop(error_key, None)
            circuit_breaker = self.get_circuit_breaker(error_key)
            circuit_breaker.record_success()
        else:
            # Reset all operations for library
            keys_to_remove = [key for key in self.error_counts.keys() if key.startswith(f"{library}.")]
            for key in keys_to_remove:
                self.error_counts.pop(key, None)
                self.last_errors.pop(key, None)
                if key in self.circuit_breakers:
                    self.circuit_breakers[key].record_success()

    def get_error_summary(self, library: Optional[str] = None) -> Dict[str, Any]:
        """Get error summary for debugging."""
        summary = {
            "error_counts": {},
            "circuit_breaker_states": {},
            "last_errors": {}
        }

        for key, count in self.error_counts.items():
            if library is None or key.startswith(f"{library}."):
                summary["error_counts"][key] = count

                if key in self.circuit_breakers:
                    summary["circuit_breaker_states"][key] = self.circuit_breakers[key].state

                if key in self.last_errors:
                    summary["last_errors"][key] = self.last_errors[key].isoformat()

        return summary


# Global error handler instance
_error_handler = InstrumentationErrorHandler()


def get_error_handler() -> InstrumentationErrorHandler:
    """Get the global error handler instance."""
    return _error_handler


def safe_operation(library: str, operation: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
    """Decorator for safe instrumentation operations with error handling."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = get_error_handler()

            # Check if operation is healthy
            if not error_handler.is_operation_healthy(library, operation):
                logger.debug(f"Skipping {library}.{operation} - circuit breaker open")
                return None

            try:
                circuit_breaker = error_handler.get_circuit_breaker(f"{library}.{operation}")
                return circuit_breaker.call(func, *args, **kwargs)

            except Exception as e:
                error_handler.handle_error(e, library, operation, severity)

                # Re-raise critical errors
                if severity == ErrorSeverity.CRITICAL:
                    raise

                # Return None for other errors (graceful degradation)
                return None

        return wrapper
    return decorator


def safe_async_operation(library: str, operation: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
    """Decorator for safe async instrumentation operations with error handling."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            error_handler = get_error_handler()

            # Check if operation is healthy
            if not error_handler.is_operation_healthy(library, operation):
                logger.debug(f"Skipping {library}.{operation} - circuit breaker open")
                return None

            try:
                # For async, we'll use a simpler approach without circuit breaker call
                # since circuit breaker doesn't support async natively
                return await func(*args, **kwargs)

            except Exception as e:
                error_handler.handle_error(e, library, operation, severity)

                # Re-raise critical errors
                if severity == ErrorSeverity.CRITICAL:
                    raise

                # Return None for other errors (graceful degradation)
                return None

        return async_wrapper
    return decorator


@contextmanager
def instrumentation_context(library: str, operation: str,
                          severity: ErrorSeverity = ErrorSeverity.MEDIUM):
    """Context manager for safe instrumentation operations."""
    error_handler = get_error_handler()

    try:
        yield
    except Exception as e:
        error_handler.handle_error(e, library, operation, severity)

        # Re-raise critical errors
        if severity == ErrorSeverity.CRITICAL:
            raise


def validate_config(config: Any, required_fields: List[str],
                   operation: str = "config_validation") -> bool:
    """Validate configuration with comprehensive error handling."""
    try:
        if config is None:
            raise ConfigurationError(
                "Configuration is None",
                severity=ErrorSeverity.HIGH,
                operation=operation
            )

        missing_fields = []
        for field in required_fields:
            if not hasattr(config, field) or getattr(config, field) is None:
                missing_fields.append(field)

        if missing_fields:
            raise ConfigurationError(
                f"Missing required configuration fields: {', '.join(missing_fields)}",
                severity=ErrorSeverity.HIGH,
                operation=operation
            )

        return True

    except ConfigurationError:
        raise
    except Exception as e:
        raise ConfigurationError(
            f"Configuration validation failed: {e}",
            severity=ErrorSeverity.HIGH,
            operation=operation,
            original_error=e
        )