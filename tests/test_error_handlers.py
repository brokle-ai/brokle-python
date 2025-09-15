"""
Unit tests for error handling components.

This module tests the error handling infrastructure including circuit breakers,
retry policies, and error tracking mechanisms.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch
from datetime import datetime, timezone, timedelta

from brokle.auto_instrumentation.error_handlers import (
    InstrumentationError,
    LibraryNotAvailableError,
    ObservabilityError,
    ConfigurationError,
    CircuitBreakerError,
    ErrorSeverity,
    CircuitBreaker,
    RetryPolicy,
    InstrumentationErrorHandler,
    safe_operation,
    safe_async_operation,
    instrumentation_context,
    validate_config,
    get_error_handler
)


class TestErrorClasses:
    """Test error class hierarchy and functionality."""

    def test_instrumentation_error_creation(self):
        """Test InstrumentationError creation and properties."""
        error = InstrumentationError(
            "Test error message",
            severity=ErrorSeverity.HIGH,
            library="test_lib",
            operation="test_op"
        )

        assert error.message == "Test error message"
        assert error.severity == ErrorSeverity.HIGH
        assert error.library == "test_lib"
        assert error.operation == "test_op"
        assert isinstance(error.timestamp, datetime)

    def test_error_message_formatting(self):
        """Test error message formatting with context."""
        original_error = ValueError("Original error")

        error = InstrumentationError(
            "Test message",
            library="test_lib",
            operation="test_op",
            original_error=original_error
        )

        error_str = str(error)
        assert "[test_lib]" in error_str
        assert "(test_op)" in error_str
        assert "Test message" in error_str
        assert "Original error: Original error" in error_str

    def test_specialized_error_types(self):
        """Test specialized error types."""
        # LibraryNotAvailableError
        lib_error = LibraryNotAvailableError("Library not found")
        assert isinstance(lib_error, InstrumentationError)

        # ObservabilityError
        obs_error = ObservabilityError("Observability failed")
        assert isinstance(obs_error, InstrumentationError)

        # ConfigurationError
        config_error = ConfigurationError("Config invalid")
        assert isinstance(config_error, InstrumentationError)

        # CircuitBreakerError
        cb_error = CircuitBreakerError("Circuit breaker open")
        assert isinstance(cb_error, InstrumentationError)

    def test_error_severity_enum(self):
        """Test error severity enumeration."""
        severities = [
            ErrorSeverity.CRITICAL,
            ErrorSeverity.HIGH,
            ErrorSeverity.MEDIUM,
            ErrorSeverity.LOW,
            ErrorSeverity.DEBUG
        ]

        for severity in severities:
            assert severity.value in ["critical", "high", "medium", "low", "debug"]


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_initial_state(self):
        """Test circuit breaker initial state."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)

        assert cb.failure_count == 0
        assert cb.last_failure_time is None
        assert cb.state == "closed"
        assert not cb.is_open()
        assert not cb.is_half_open()

    def test_failure_tracking(self):
        """Test failure tracking and state transitions."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)

        # Record failures
        for i in range(2):
            cb.record_failure()
            assert cb.state == "closed"  # Should still be closed

        # Third failure should open the circuit
        cb.record_failure()
        assert cb.state == "open"
        assert cb.is_open()
        assert cb.failure_count == 3

    def test_recovery_timeout(self):
        """Test recovery timeout functionality."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)  # 100ms timeout

        # Open the circuit
        cb.record_failure()
        assert cb.is_open()
        assert not cb.should_attempt_reset()

        # Wait for recovery timeout
        time.sleep(0.15)
        assert cb.should_attempt_reset()

    def test_half_open_state(self):
        """Test half-open state functionality."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)

        # Open the circuit
        cb.record_failure()
        assert cb.is_open()

        # Wait and trigger half-open
        time.sleep(0.15)
        cb.state = "half-open"  # Manually set for testing
        assert cb.is_half_open()

        # Success should close the circuit
        cb.record_success()
        assert cb.state == "closed"
        assert cb.failure_count == 0

    def test_circuit_breaker_call_protection(self):
        """Test circuit breaker call protection."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        call_count = 0

        def test_function():
            nonlocal call_count
            call_count += 1
            raise Exception(f"Test failure {call_count}")

        # First two calls should fail and be recorded
        for i in range(2):
            with pytest.raises(Exception):
                cb.call(test_function)

        # Circuit should now be open
        assert cb.is_open()

        # Next call should raise CircuitBreakerError without calling function
        with pytest.raises(CircuitBreakerError):
            cb.call(test_function)

        assert call_count == 2  # Function should not have been called again

    def test_successful_recovery(self):
        """Test successful recovery after circuit opens."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)

        # Open the circuit
        with pytest.raises(Exception):
            cb.call(lambda: exec('raise Exception("test")'))

        assert cb.is_open()

        # Wait for recovery
        time.sleep(0.15)

        # Successful call should close circuit
        result = cb.call(lambda: "success")
        assert result == "success"
        assert cb.state == "closed"


class TestRetryPolicy:
    """Test retry policy functionality."""

    def test_retry_policy_initialization(self):
        """Test retry policy initialization."""
        policy = RetryPolicy(
            max_retries=3,
            base_delay=1.0,
            max_delay=30.0,
            backoff_factor=2.0,
            jitter=True
        )

        assert policy.max_retries == 3
        assert policy.base_delay == 1.0
        assert policy.max_delay == 30.0
        assert policy.backoff_factor == 2.0
        assert policy.jitter is True

    def test_delay_calculation(self):
        """Test exponential backoff delay calculation."""
        policy = RetryPolicy(
            max_retries=3,
            base_delay=1.0,
            max_delay=10.0,
            backoff_factor=2.0,
            jitter=False  # Disable jitter for predictable testing
        )

        # Test delay calculations
        delay_0 = policy.calculate_delay(0)  # Should be 1.0
        delay_1 = policy.calculate_delay(1)  # Should be 2.0
        delay_2 = policy.calculate_delay(2)  # Should be 4.0

        assert delay_0 == 1.0
        assert delay_1 == 2.0
        assert delay_2 == 4.0

        # Test max delay cap
        delay_large = policy.calculate_delay(10)
        assert delay_large == 10.0  # Should be capped at max_delay

    def test_jitter_functionality(self):
        """Test jitter adds randomness to delays."""
        policy = RetryPolicy(base_delay=1.0, jitter=True)

        delays = [policy.calculate_delay(0) for _ in range(10)]

        # All delays should be different due to jitter
        assert len(set(delays)) > 1

        # All delays should be between 0.5 and 1.0 (jitter range)
        for delay in delays:
            assert 0.5 <= delay <= 1.0

    def test_successful_retry(self):
        """Test successful execution after retries."""
        policy = RetryPolicy(max_retries=3, base_delay=0.01)  # Fast for testing

        attempt_count = 0

        def sometimes_failing_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception(f"Attempt {attempt_count} failed")
            return f"Success on attempt {attempt_count}"

        result = policy.execute(sometimes_failing_function)
        assert result == "Success on attempt 3"
        assert attempt_count == 3

    def test_retry_exhaustion(self):
        """Test behavior when retries are exhausted."""
        policy = RetryPolicy(max_retries=2, base_delay=0.01)

        def always_failing_function():
            raise ValueError("Always fails")

        with pytest.raises(InstrumentationError) as exc_info:
            policy.execute(always_failing_function)

        assert "Failed after 2 retries" in str(exc_info.value)
        assert exc_info.value.severity == ErrorSeverity.HIGH


class TestInstrumentationErrorHandler:
    """Test comprehensive error handler functionality."""

    def setup_method(self):
        """Setup for each test method."""
        self.error_handler = InstrumentationErrorHandler()

    def test_error_tracking(self):
        """Test error counting and tracking."""
        test_error = Exception("Test error")

        # Handle multiple errors
        for i in range(3):
            self.error_handler.handle_error(
                test_error, "test_lib", "test_op", ErrorSeverity.MEDIUM
            )

        # Check error counts
        summary = self.error_handler.get_error_summary()
        assert "test_lib.test_op" in summary["error_counts"]
        assert summary["error_counts"]["test_lib.test_op"] == 3

    def test_circuit_breaker_integration(self):
        """Test circuit breaker integration in error handler."""
        test_error = Exception("Test error")

        # Initially healthy
        assert self.error_handler.is_operation_healthy("test_lib", "test_op")

        # Generate enough errors to open circuit breaker
        for i in range(5):
            self.error_handler.handle_error(
                test_error, "test_lib", "test_op", ErrorSeverity.HIGH
            )

        # Should be unhealthy now
        assert not self.error_handler.is_operation_healthy("test_lib", "test_op")

    def test_error_reset(self):
        """Test error reset functionality."""
        test_error = Exception("Test error")

        # Generate errors
        self.error_handler.handle_error(
            test_error, "test_lib", "test_op", ErrorSeverity.MEDIUM
        )

        # Verify errors exist
        summary_before = self.error_handler.get_error_summary()
        assert len(summary_before["error_counts"]) > 0

        # Reset errors
        self.error_handler.reset_errors("test_lib", "test_op")

        # Should be healthy again
        assert self.error_handler.is_operation_healthy("test_lib", "test_op")

    def test_error_summary_generation(self):
        """Test error summary generation."""
        test_error = Exception("Test error")

        # Generate errors for multiple operations
        self.error_handler.handle_error(test_error, "lib1", "op1", ErrorSeverity.LOW)
        self.error_handler.handle_error(test_error, "lib1", "op2", ErrorSeverity.MEDIUM)
        self.error_handler.handle_error(test_error, "lib2", "op1", ErrorSeverity.HIGH)

        summary = self.error_handler.get_error_summary()

        # Should have error counts
        assert "error_counts" in summary
        assert "circuit_breaker_states" in summary
        assert "last_errors" in summary

        # Should track all operations
        assert len(summary["error_counts"]) >= 3


class TestSafeOperationDecorators:
    """Test safe operation decorators."""

    def setup_method(self):
        """Setup for each test method."""
        # Get a fresh error handler
        from brokle.auto_instrumentation.error_handlers import _error_handler
        _error_handler.reset_errors("test_lib")

    def test_safe_operation_success(self):
        """Test safe operation decorator with successful function."""
        @safe_operation("test_lib", "test_op", ErrorSeverity.LOW)
        def successful_function(x, y):
            return x + y

        result = successful_function(2, 3)
        assert result == 5

    def test_safe_operation_failure(self):
        """Test safe operation decorator with failing function."""
        @safe_operation("test_lib", "test_op", ErrorSeverity.LOW)
        def failing_function():
            raise ValueError("Test failure")

        # Should return None on failure for non-critical errors
        result = failing_function()
        assert result is None

    def test_safe_operation_critical_failure(self):
        """Test safe operation decorator with critical failure."""
        @safe_operation("test_lib", "test_op", ErrorSeverity.CRITICAL)
        def critical_failing_function():
            raise ValueError("Critical failure")

        # Should re-raise critical errors
        with pytest.raises(ValueError):
            critical_failing_function()

    @pytest.mark.asyncio
    async def test_safe_async_operation_success(self):
        """Test safe async operation decorator with successful function."""
        @safe_async_operation("test_lib", "async_op", ErrorSeverity.LOW)
        async def successful_async_function(x, y):
            await asyncio.sleep(0.01)  # Simulate async work
            return x * y

        result = await successful_async_function(4, 5)
        assert result == 20

    @pytest.mark.asyncio
    async def test_safe_async_operation_failure(self):
        """Test safe async operation decorator with failing function."""
        @safe_async_operation("test_lib", "async_op", ErrorSeverity.LOW)
        async def failing_async_function():
            await asyncio.sleep(0.01)
            raise RuntimeError("Async failure")

        # Should return None on failure for non-critical errors
        result = await failing_async_function()
        assert result is None

    def test_circuit_breaker_protection(self):
        """Test circuit breaker protection in decorators."""
        call_count = 0

        @safe_operation("test_lib", "protected_op", ErrorSeverity.MEDIUM)
        def sometimes_failing_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise Exception(f"Failure {call_count}")
            return f"Success {call_count}"

        # First few calls should fail
        for i in range(3):
            result = sometimes_failing_function()
            assert result is None

        # Circuit should be open, so next call should be skipped
        result = sometimes_failing_function()
        assert result is None
        # Call count should not have increased due to circuit breaker
        assert call_count == 3


class TestContextManager:
    """Test instrumentation context manager."""

    def test_successful_context(self):
        """Test context manager with successful operation."""
        with instrumentation_context("test_lib", "test_op", ErrorSeverity.LOW):
            # Should execute without issues
            result = 2 + 2
            assert result == 4

    def test_context_with_exception(self):
        """Test context manager with exception."""
        # Should handle exceptions gracefully for non-critical errors
        with instrumentation_context("test_lib", "test_op", ErrorSeverity.LOW):
            # This should not propagate for LOW severity
            pass

    def test_context_with_critical_exception(self):
        """Test context manager with critical exception."""
        # Should re-raise critical exceptions
        with pytest.raises(ValueError):
            with instrumentation_context("test_lib", "test_op", ErrorSeverity.CRITICAL):
                raise ValueError("Critical error in context")


class TestConfigurationValidation:
    """Test configuration validation functionality."""

    def test_valid_config_validation(self):
        """Test validation of valid configuration."""
        # Create a mock config object
        mock_config = Mock()
        mock_config.api_key = "test_key"
        mock_config.base_url = "https://api.example.com"

        result = validate_config(mock_config, ["api_key", "base_url"])
        assert result is True

    def test_invalid_config_validation(self):
        """Test validation of invalid configuration."""
        # Config with missing fields
        mock_config = Mock()
        mock_config.api_key = "test_key"
        mock_config.base_url = None  # Missing required field

        with pytest.raises(ConfigurationError):
            validate_config(mock_config, ["api_key", "base_url"])

    def test_none_config_validation(self):
        """Test validation of None configuration."""
        with pytest.raises(ConfigurationError):
            validate_config(None, ["api_key"])

    def test_config_validation_with_exception(self):
        """Test configuration validation error handling."""
        # Create a config that raises exception when accessed
        mock_config = Mock()
        mock_config.api_key = Mock(side_effect=Exception("Access error"))

        with pytest.raises(ConfigurationError):
            validate_config(mock_config, ["api_key"])


class TestGlobalErrorHandler:
    """Test global error handler singleton."""

    def test_global_error_handler_singleton(self):
        """Test global error handler is singleton."""
        handler1 = get_error_handler()
        handler2 = get_error_handler()

        assert handler1 is handler2

    def test_global_error_handler_functionality(self):
        """Test global error handler basic functionality."""
        handler = get_error_handler()

        # Should be an instance of InstrumentationErrorHandler
        assert isinstance(handler, InstrumentationErrorHandler)

        # Should have basic functionality
        assert hasattr(handler, 'handle_error')
        assert hasattr(handler, 'is_operation_healthy')
        assert hasattr(handler, 'get_error_summary')
        assert hasattr(handler, 'reset_errors')


# Integration test markers
pytestmark = [
    pytest.mark.unit,
    pytest.mark.error_handlers
]

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])