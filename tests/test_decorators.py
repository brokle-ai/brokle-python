"""Tests for @observe decorator functionality."""

import pytest
from unittest.mock import patch, MagicMock
import asyncio

from brokle.decorators import observe, trace_workflow, observe_llm, observe_retrieval


class TestObserveDecorator:
    """Test essential @observe decorator functionality."""

    def test_observe_decorator_basic(self):
        """Test basic @observe decorator usage."""
        with patch('brokle.client.get_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.config.telemetry_enabled = True
            mock_get_client.return_value = mock_client

            @observe()
            def test_function(x, y):
                return x + y

            result = test_function(2, 3)
            assert result == 5

    def test_observe_decorator_with_configuration(self):
        """Test @observe decorator with name and metadata."""
        with patch('brokle.client.get_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.config.telemetry_enabled = True
            mock_get_client.return_value = mock_client

            @observe(
                name="custom-operation",
                metadata={"user_id": "user123"},
                tags=["test"],
                capture_inputs=True,
                capture_outputs=True
            )
            def test_function(data):
                return f"processed-{data}"

            result = test_function("input")
            assert result == "processed-input"

    def test_observe_decorator_error_handling(self):
        """Test @observe decorator error handling."""
        with patch('brokle.client.get_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.config.telemetry_enabled = True
            mock_get_client.return_value = mock_client

            @observe()
            def failing_function():
                raise ValueError("Test error")

            with pytest.raises(ValueError, match="Test error"):
                failing_function()

    def test_observe_decorator_without_client(self):
        """Test @observe decorator when client not available."""
        with patch('brokle.client.get_client') as mock_get_client:
            mock_get_client.side_effect = Exception("No client")

            @observe()
            def test_function():
                return "test"

            result = test_function()
            assert result == "test"

    def test_observe_decorator_disabled_telemetry(self):
        """Test @observe decorator when telemetry disabled."""
        with patch('brokle.client.get_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.config.telemetry_enabled = False
            mock_get_client.return_value = mock_client

            @observe()
            def test_function():
                return "test"

            result = test_function()
            assert result == "test"

    @pytest.mark.asyncio
    async def test_observe_decorator_async(self):
        """Test @observe decorator with async functions."""
        with patch('brokle.client.get_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.config.telemetry_enabled = True
            mock_get_client.return_value = mock_client

            @observe()
            async def async_function(delay):
                await asyncio.sleep(delay)
                return f"completed after {delay}s"

            result = await async_function(0.01)
            assert result == "completed after 0.01s"


class TestTraceWorkflow:
    """Test trace_workflow context manager."""

    def test_trace_workflow_basic(self):
        """Test basic trace_workflow usage."""
        with patch('brokle.decorators.create_span') as mock_create_span:
            mock_span = MagicMock()
            mock_create_span.return_value = mock_span

            with trace_workflow("test-workflow") as span:
                assert span == mock_span

            mock_create_span.assert_called_once()
            mock_span.end.assert_called_once()

    def test_trace_workflow_error_handling(self):
        """Test trace_workflow error handling."""
        with patch('brokle.decorators.create_span') as mock_create_span:
            mock_span = MagicMock()
            mock_create_span.return_value = mock_span

            with pytest.raises(ValueError, match="Test error"):
                with trace_workflow("test-workflow"):
                    raise ValueError("Test error")

            mock_span.end.assert_called_once()


class TestSpecializedDecorators:
    """Test specialized LLM and retrieval decorators."""

    def test_observe_llm_decorator(self):
        """Test @observe_llm decorator."""
        with patch('brokle.client.get_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.config.telemetry_enabled = True
            mock_get_client.return_value = mock_client

            @observe_llm(model="gpt-4")
            def llm_call(prompt):
                return f"Response to: {prompt}"

            result = llm_call("Hello")
            assert result == "Response to: Hello"

    def test_observe_retrieval_decorator(self):
        """Test @observe_retrieval decorator."""
        with patch('brokle.client.get_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.config.telemetry_enabled = True
            mock_get_client.return_value = mock_client

            @observe_retrieval(index_name="documents")
            def search_documents(query):
                return f"Found documents for: {query}"

            result = search_documents("test query")
            assert result == "Found documents for: test query"