"""
Clean Decorator Tests

Tests the actual @observe decorator functionality without deprecated mocking patterns.
"""

from unittest.mock import MagicMock, patch

import pytest

from brokle import observe
from brokle.decorators import trace_workflow


class TestObserveDecorator:
    """Test @observe decorator functionality."""

    def test_observe_basic_functionality(self):
        """Test basic @observe decorator usage."""

        @observe(name="test-function")
        def sample_function(x, y):
            return x + y

        # Function should execute correctly
        result = sample_function(2, 3)
        assert result == 5

    def test_observe_with_capture_options(self):
        """Test @observe decorator with capture options."""

        @observe(name="test-function", capture_inputs=False, capture_outputs=False)
        def sample_function(x, y):
            return x + y

        # Function should execute correctly even with capture disabled
        result = sample_function(2, 3)
        assert result == 5

    def test_observe_async_function(self):
        """Test @observe decorator with async functions."""

        @observe(name="test-async")
        async def async_function(x, y):
            return x * y

        # Note: This would require an async test context in real usage
        # For now, just verify the decorator can be applied
        assert callable(async_function)


class TestTraceWorkflow:
    """Test trace_workflow context manager."""

    def test_trace_workflow_returns_span(self):
        """Test that trace_workflow returns a BrokleSpan object."""
        with trace_workflow("test-workflow") as span:
            # Should return a BrokleSpan object, not a mock
            assert hasattr(span, "span_id")
            assert hasattr(span, "name")
            assert hasattr(span, "attributes")
            assert span.name == "workflow.test-workflow"

    def test_trace_workflow_with_metadata(self):
        """Test trace_workflow with metadata."""
        metadata = {"user_id": "123", "session": "abc"}

        with trace_workflow("test-workflow", metadata=metadata) as span:
            # Verify span exists and has expected structure
            assert span is not None
            assert hasattr(span, "attributes")

    def test_trace_workflow_error_handling(self):
        """Test trace_workflow handles exceptions properly."""
        try:
            with trace_workflow("test-error-workflow") as span:
                assert span is not None
                raise ValueError("Test error")
        except ValueError:
            # Exception should be re-raised
            pass
        else:
            pytest.fail("Exception should have been re-raised")


class TestPatternIntegration:
    """Test decorator patterns with real functionality."""

    def test_decorator_with_client_creation(self):
        """Test decorator works with client creation."""

        @observe(name="client-test")
        def function_that_uses_client():
            # This would normally create spans, but we're just testing
            # that the decorator doesn't interfere with normal functionality
            return "success"

        result = function_that_uses_client()
        assert result == "success"
