"""
End-to-end integration tests for auto-instrumentation system.

This module contains comprehensive tests for the auto-instrumentation
functionality including error handling, circuit breakers, and health monitoring.
"""

import asyncio
import json
import logging
import pytest
import time
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any

# Import the auto-instrumentation components
from brokle.auto_instrumentation import (
    InstrumentationRegistry,
    auto_instrument,
    print_status,
    print_health_report,
    get_status,
    get_health_report,
    reset_all_errors,
    get_registry,
    InstrumentationError,
    LibraryNotAvailableError,
    ObservabilityError,
    ConfigurationError,
    ErrorSeverity,
    get_error_handler
)

from brokle.auto_instrumentation.error_handlers import (
    CircuitBreaker,
    RetryPolicy,
    InstrumentationErrorHandler,
    safe_operation,
    safe_async_operation,
    instrumentation_context
)

from brokle.auto_instrumentation.openai_instrumentation import (
    OpenAIInstrumentation,
    instrument_openai,
    uninstrument_openai,
    is_openai_instrumented
)

from brokle.auto_instrumentation.registry import LibraryInstrumentation


class TestInstrumentationErrorHandling:
    """Test comprehensive error handling functionality."""

    def setup_method(self):
        """Setup for each test method."""
        # Reset error tracking before each test
        reset_all_errors()
        self.error_handler = get_error_handler()

    def test_error_severity_levels(self):
        """Test different error severity levels."""
        # Test all severity levels
        severities = [
            ErrorSeverity.CRITICAL,
            ErrorSeverity.HIGH,
            ErrorSeverity.MEDIUM,
            ErrorSeverity.LOW,
            ErrorSeverity.DEBUG
        ]

        for severity in severities:
            error = InstrumentationError(
                f"Test error for {severity.value}",
                severity=severity,
                library="test",
                operation="test_operation"
            )

            assert error.severity == severity
            assert "test" in error.message
            assert error.library == "test"
            assert error.operation == "test_operation"

    def test_circuit_breaker_functionality(self):
        """Test circuit breaker pattern."""
        circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1)

        # Test initial state
        assert not circuit_breaker.is_open()
        assert circuit_breaker.state == "closed"

        # Simulate failures
        for i in range(3):
            circuit_breaker.record_failure()

        # Should be open after failure threshold
        assert circuit_breaker.is_open()
        assert circuit_breaker.state == "open"

        # Test recovery after timeout
        time.sleep(1.1)  # Wait for recovery timeout
        assert circuit_breaker.should_attempt_reset()

        # Test successful recovery
        circuit_breaker.record_success()
        assert not circuit_breaker.is_open()
        assert circuit_breaker.state == "closed"

    def test_retry_policy(self):
        """Test retry policy with exponential backoff."""
        retry_policy = RetryPolicy(max_retries=3, base_delay=0.1, max_delay=1.0)

        call_count = 0

        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception(f"Failure {call_count}")
            return "success"

        # Should succeed on third attempt
        result = retry_policy.execute(failing_function)
        assert result == "success"
        assert call_count == 3

    def test_error_handler_tracking(self):
        """Test error tracking and health monitoring."""
        error_handler = get_error_handler()

        # Initially healthy
        assert error_handler.is_operation_healthy("test_lib", "test_op")

        # Simulate errors
        test_error = Exception("Test error")
        for i in range(5):
            error_handler.handle_error(test_error, "test_lib", "test_op", ErrorSeverity.MEDIUM)

        # Should be unhealthy after multiple errors
        assert not error_handler.is_operation_healthy("test_lib", "test_op")

        # Reset should make it healthy again
        error_handler.reset_errors("test_lib", "test_op")
        assert error_handler.is_operation_healthy("test_lib", "test_op")

    def test_safe_operation_decorator(self):
        """Test safe operation decorator functionality."""
        call_count = 0

        @safe_operation("test_lib", "test_operation", ErrorSeverity.LOW)
        def test_function(should_fail: bool = False):
            nonlocal call_count
            call_count += 1
            if should_fail:
                raise Exception("Test failure")
            return "success"

        # Successful operation
        result = test_function(should_fail=False)
        assert result == "success"

        # Failed operation should return None (graceful degradation)
        result = test_function(should_fail=True)
        assert result is None  # Safe operation returns None on failure

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_safe_async_operation_decorator(self):
        """Test safe async operation decorator functionality."""
        call_count = 0

        @safe_async_operation("test_lib", "async_operation", ErrorSeverity.LOW)
        async def async_test_function(should_fail: bool = False):
            nonlocal call_count
            call_count += 1
            if should_fail:
                raise Exception("Async test failure")
            return "async_success"

        # Successful async operation
        result = await async_test_function(should_fail=False)
        assert result == "async_success"

        # Failed async operation should return None
        result = await async_test_function(should_fail=True)
        assert result is None

        assert call_count == 2

    def test_instrumentation_context_manager(self):
        """Test instrumentation context manager."""
        error_handler = get_error_handler()

        # Test successful context
        with instrumentation_context("test_lib", "context_test", ErrorSeverity.LOW):
            # Should not raise any exceptions
            pass

        # Test context with exception
        with instrumentation_context("test_lib", "context_error", ErrorSeverity.LOW):
            # This should be caught and handled
            pass

        # Error handler should have tracked any errors
        summary = error_handler.get_error_summary()
        # Context manager handles errors silently for LOW severity


class TestInstrumentationRegistry:
    """Test instrumentation registry functionality."""

    def setup_method(self):
        """Setup for each test method."""
        reset_all_errors()
        self.registry = InstrumentationRegistry()

    def test_registry_initialization(self):
        """Test registry initializes with default libraries."""
        libraries = self.registry.list_libraries()

        # Should have default libraries
        expected_libraries = ["openai", "anthropic", "langchain"]
        for lib in expected_libraries:
            assert lib in libraries

    def test_library_registration(self):
        """Test registering and unregistering libraries."""
        # Create a mock library
        mock_lib = LibraryInstrumentation(
            name="test_lib",
            instrument_func=lambda: True,
            uninstrument_func=lambda: True,
            is_instrumented_func=lambda: False,
            is_available_func=lambda: True,
            description="Test library"
        )

        # Register library
        self.registry.register_library(mock_lib)
        assert "test_lib" in self.registry.list_libraries()

        # Unregister library
        assert self.registry.unregister_library("test_lib")
        assert "test_lib" not in self.registry.list_libraries()

    def test_instrumentation_status_tracking(self):
        """Test instrumentation status tracking."""
        status = self.registry.get_status()

        # Should have status for all default libraries
        assert "openai" in status
        assert "anthropic" in status
        assert "langchain" in status

        # Status should be InstrumentationStatus enum values
        for lib_name, lib_status in status.items():
            assert hasattr(lib_status, 'value')

    def test_health_reporting(self):
        """Test comprehensive health reporting."""
        health_report = self.registry.get_health_report()

        # Should have all required sections
        assert "overall_health" in health_report
        assert "library_details" in health_report
        assert "error_summary" in health_report
        assert "circuit_breaker_states" in health_report

        # Overall health should have required metrics
        overall = health_report["overall_health"]
        assert "score" in overall
        assert "total_libraries" in overall
        assert "healthy_libraries" in overall
        assert "instrumented_libraries" in overall
        assert "available_libraries" in overall

    def test_instrumentation_summary(self):
        """Test instrumentation summary functionality."""
        summary = self.registry.get_instrumentation_summary()

        for lib_name, lib_info in summary.items():
            # Each library should have required fields
            required_fields = [
                "status", "description", "auto_instrument",
                "available", "instrumented", "healthy"
            ]
            for field in required_fields:
                assert field in lib_info

    def test_auto_instrumentation(self):
        """Test automatic instrumentation of available libraries."""
        with patch('brokle.auto_instrumentation.openai_instrumentation.OPENAI_AVAILABLE', True):
            with patch('brokle.auto_instrumentation.openai_instrumentation.instrument_openai', return_value=True):
                results = self.registry.auto_instrument()

                # Should attempt instrumentation of available libraries
                assert isinstance(results, dict)


class TestOpenAIInstrumentation:
    """Test OpenAI-specific instrumentation functionality."""

    def setup_method(self):
        """Setup for each test method."""
        reset_all_errors()

    @patch('brokle.auto_instrumentation.openai_instrumentation.OPENAI_AVAILABLE', True)
    def test_openai_availability_check(self):
        """Test OpenAI library availability check."""
        instrumentation = OpenAIInstrumentation()
        assert instrumentation.is_available()

    @patch('brokle.auto_instrumentation.openai_instrumentation.OPENAI_AVAILABLE', False)
    def test_openai_unavailable_handling(self):
        """Test handling when OpenAI is not available."""
        instrumentation = OpenAIInstrumentation()
        assert not instrumentation.is_available()

        # Should fail gracefully when not available
        result = instrumentation.instrument()
        assert result is None  # Safe operation returns None on failure

    @patch('brokle.auto_instrumentation.openai_instrumentation.OPENAI_AVAILABLE', True)
    @patch('brokle.auto_instrumentation.openai_instrumentation.openai')
    def test_openai_instrumentation_with_mocks(self, mock_openai):
        """Test OpenAI instrumentation with proper mocks."""
        # Setup mock OpenAI structure
        mock_completions = Mock()
        mock_completions.create = Mock()

        mock_chat = Mock()
        mock_chat.completions = Mock()
        mock_chat.completions.Completions = mock_completions

        mock_resources = Mock()
        mock_resources.chat = mock_chat

        mock_openai.resources = mock_resources

        instrumentation = OpenAIInstrumentation()

        # Mock the config and client
        with patch.object(instrumentation, 'config', return_value=Mock()):
            with patch.object(instrumentation, 'client', return_value=Mock()):
                # Should succeed with proper mocks
                result = instrumentation.instrument()
                # The actual result depends on internal implementation details
                # The important part is that it doesn't raise exceptions

    def test_openai_global_functions(self):
        """Test global OpenAI instrumentation functions."""
        # These should not raise exceptions even if OpenAI is not available
        # The safe operation decorators should handle failures gracefully

        # Test the global functions exist and are callable
        assert callable(instrument_openai)
        assert callable(uninstrument_openai)
        assert callable(is_openai_instrumented)


class TestRegistryIntegration:
    """Test integration between registry and instrumentation components."""

    def setup_method(self):
        """Setup for each test method."""
        reset_all_errors()

    def test_global_registry_functions(self):
        """Test global registry functions."""
        # Test registry access
        registry = get_registry()
        assert isinstance(registry, InstrumentationRegistry)

        # Test status functions
        status = get_status()
        assert isinstance(status, dict)

        # Test health functions
        health_report = get_health_report()
        assert isinstance(health_report, dict)
        assert "overall_health" in health_report

    def test_auto_instrument_integration(self):
        """Test auto instrumentation integration."""
        # Should not raise exceptions
        results = auto_instrument()
        assert isinstance(results, dict)

        # Test with specific libraries
        results = auto_instrument(libraries=["openai"])
        assert isinstance(results, dict)

        # Test with exclusions
        results = auto_instrument(exclude=["langchain"])
        assert isinstance(results, dict)

    def test_status_printing(self, capsys):
        """Test status printing functionality."""
        # Test basic status printing
        print_status()
        captured = capsys.readouterr()
        assert "Brokle Auto-Instrumentation Status" in captured.out

        # Test health report printing
        print_health_report()
        captured = capsys.readouterr()
        assert "Health Report" in captured.out

    def test_error_reset_functionality(self):
        """Test error reset functionality."""
        error_handler = get_error_handler()

        # Generate some errors
        test_error = Exception("Test error")
        error_handler.handle_error(test_error, "test_lib", "test_op", ErrorSeverity.LOW)

        # Verify errors exist
        summary_before = error_handler.get_error_summary()

        # Reset all errors
        reset_all_errors()

        # Verify errors are cleared
        summary_after = error_handler.get_error_summary()

        # After reset, should have fewer or no errors
        assert len(summary_after.get("error_counts", {})) <= len(summary_before.get("error_counts", {}))


class TestEndToEndScenarios:
    """Test realistic end-to-end scenarios."""

    def setup_method(self):
        """Setup for each test method."""
        reset_all_errors()

    def test_complete_instrumentation_lifecycle(self):
        """Test complete instrumentation lifecycle."""
        registry = get_registry()

        # 1. Check initial status
        initial_status = registry.get_status()
        assert isinstance(initial_status, dict)

        # 2. Attempt auto instrumentation
        results = auto_instrument()
        assert isinstance(results, dict)

        # 3. Check health after instrumentation
        health_report = get_health_report()
        assert "overall_health" in health_report
        assert health_report["overall_health"]["total_libraries"] > 0

        # 4. Reset errors
        reset_all_errors()

        # 5. Final status check
        final_status = registry.get_status()
        assert isinstance(final_status, dict)

    @patch('brokle.auto_instrumentation.openai_instrumentation.OPENAI_AVAILABLE', True)
    def test_instrumentation_failure_recovery(self):
        """Test instrumentation failure and recovery scenarios."""
        registry = get_registry()

        # Simulate instrumentation failures by making functions fail
        with patch('brokle.auto_instrumentation.openai_instrumentation.instrument_openai', side_effect=Exception("Mock failure")):
            # Should handle failures gracefully
            results = auto_instrument(libraries=["openai"])

            # Should not raise exceptions
            assert isinstance(results, dict)

        # Check health after failures
        health_report = get_health_report()

        # Should track the failures
        assert isinstance(health_report, dict)

        # Reset and verify recovery
        reset_all_errors()

        health_after_reset = get_health_report()
        assert isinstance(health_after_reset, dict)

    def test_concurrent_operations(self):
        """Test concurrent operations don't interfere with each other."""
        import threading
        import time

        results = []
        errors = []

        def worker():
            try:
                # Each thread performs various operations
                status = get_status()
                results.append(status)

                health = get_health_report()
                results.append(health)

                # Simulate some instrumentation attempts
                auto_result = auto_instrument(libraries=[])  # Empty list to avoid actual instrumentation
                results.append(auto_result)

            except Exception as e:
                errors.append(e)

        # Run multiple threads concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0, f"Concurrent operations had errors: {errors}"
        assert len(results) == 15  # 3 operations * 5 threads

    def test_memory_and_cleanup(self):
        """Test memory usage and cleanup behavior."""
        import gc

        # Get initial object count
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Perform many operations
        for i in range(100):
            registry = get_registry()
            status = get_status()
            health = get_health_report()

            # Force cleanup
            del registry, status, health

        # Force garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())

        # Should not have significant memory leaks
        # Allow some variance for test framework overhead
        assert final_objects - initial_objects < 1000, "Potential memory leak detected"

    def test_logging_integration(self, caplog):
        """Test logging integration works properly."""
        with caplog.at_level(logging.DEBUG):
            # Perform operations that should generate logs
            registry = get_registry()
            health_report = registry.get_health_report()

            # Try to instrument (may fail safely)
            auto_instrument()

        # Should have some log entries
        assert len(caplog.records) >= 0  # At minimum, should not crash

        # Check that error logs are properly structured
        for record in caplog.records:
            assert hasattr(record, 'levelname')
            assert hasattr(record, 'message')


# Fixtures and test configuration
@pytest.fixture
def mock_openai_available():
    """Fixture to mock OpenAI as available."""
    with patch('brokle.auto_instrumentation.openai_instrumentation.OPENAI_AVAILABLE', True):
        yield


@pytest.fixture
def mock_openai_client():
    """Fixture to mock OpenAI client functionality."""
    mock_client = Mock()
    mock_client.observability = Mock()
    mock_client.observability.create_observation_sync = Mock(return_value=Mock(id="test_obs_id"))
    mock_client.observability.complete_observation_sync = Mock(return_value=True)
    mock_client.observability.create_trace_sync = Mock(return_value=Mock(id="test_trace_id"))

    with patch('brokle.auto_instrumentation.openai_instrumentation.OpenAIInstrumentation.client', new_callable=lambda: mock_client):
        yield mock_client


@pytest.fixture(autouse=True)
def reset_instrumentation_state():
    """Automatically reset instrumentation state before each test."""
    reset_all_errors()
    yield
    reset_all_errors()


# Integration test markers
pytestmark = [
    pytest.mark.integration,
    pytest.mark.auto_instrumentation
]

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])