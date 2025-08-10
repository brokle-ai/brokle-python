"""Tests for the decorators module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from brokle.decorators import observe
from brokle.telemetry import TelemetryManager


class TestObserveDecorator:
    """Test @observe decorator functionality."""
    
    @pytest.fixture
    def mock_telemetry(self):
        """Create mock telemetry manager."""
        with patch('brokle.decorators.TelemetryManager') as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance
            yield mock_instance
    
    def test_observe_sync_function(self, mock_telemetry):
        """Test @observe decorator on sync function."""
        mock_telemetry.start_span.return_value.__enter__ = MagicMock()
        mock_telemetry.start_span.return_value.__exit__ = MagicMock()
        
        @observe(name="test_function")
        def test_function(x, y):
            return x + y
        
        result = test_function(1, 2)
        
        assert result == 3
        mock_telemetry.start_span.assert_called_once()
        call_args = mock_telemetry.start_span.call_args
        assert call_args[0][0] == "test_function"
    
    def test_observe_sync_function_with_error(self, mock_telemetry):
        """Test @observe decorator on sync function with error."""
        mock_span = MagicMock()
        mock_telemetry.start_span.return_value = mock_span
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock()
        
        @observe(name="test_function")
        def test_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            test_function()
        
        mock_telemetry.start_span.assert_called_once()
        mock_span.record_exception.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_observe_async_function(self, mock_telemetry):
        """Test @observe decorator on async function."""
        mock_span = AsyncMock()
        mock_telemetry.start_span.return_value = mock_span
        mock_span.__aenter__ = AsyncMock(return_value=mock_span)
        mock_span.__aexit__ = AsyncMock()
        
        @observe(name="test_async_function")
        async def test_async_function(x, y):
            await asyncio.sleep(0.001)  # Small delay
            return x * y
        
        result = await test_async_function(3, 4)
        
        assert result == 12
        mock_telemetry.start_span.assert_called_once()
        call_args = mock_telemetry.start_span.call_args
        assert call_args[0][0] == "test_async_function"
    
    @pytest.mark.asyncio
    async def test_observe_async_function_with_error(self, mock_telemetry):
        """Test @observe decorator on async function with error."""
        mock_span = AsyncMock()
        mock_telemetry.start_span.return_value = mock_span
        mock_span.__aenter__ = AsyncMock(return_value=mock_span)
        mock_span.__aexit__ = AsyncMock()
        
        @observe(name="test_async_function")
        async def test_async_function():
            await asyncio.sleep(0.001)
            raise RuntimeError("Async test error")
        
        with pytest.raises(RuntimeError, match="Async test error"):
            await test_async_function()
        
        mock_telemetry.start_span.assert_called_once()
        mock_span.record_exception.assert_called_once()
    
    def test_observe_with_metadata(self, mock_telemetry):
        """Test @observe decorator with metadata."""
        mock_span = MagicMock()
        mock_telemetry.start_span.return_value = mock_span
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock()
        
        @observe(
            name="test_function",
            metadata={"user_id": "123", "session_id": "abc"}
        )
        def test_function(value):
            return value * 2
        
        result = test_function(5)
        
        assert result == 10
        mock_telemetry.start_span.assert_called_once()
        mock_span.set_attributes.assert_called()
    
    def test_observe_with_tags(self, mock_telemetry):
        """Test @observe decorator with tags."""
        mock_span = MagicMock()
        mock_telemetry.start_span.return_value = mock_span
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock()
        
        @observe(
            name="test_function",
            tags=["api", "user-facing", "critical"]
        )
        def test_function():
            return "success"
        
        result = test_function()
        
        assert result == "success"
        mock_telemetry.start_span.assert_called_once()
        mock_span.set_attributes.assert_called()
    
    def test_observe_default_name(self, mock_telemetry):
        """Test @observe decorator with default name."""
        mock_span = MagicMock()
        mock_telemetry.start_span.return_value = mock_span
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock()
        
        @observe()
        def my_function():
            return "default_name_test"
        
        result = my_function()
        
        assert result == "default_name_test"
        mock_telemetry.start_span.assert_called_once()
        call_args = mock_telemetry.start_span.call_args
        assert call_args[0][0] == "my_function"
    
    def test_observe_preserves_function_attributes(self, mock_telemetry):
        """Test that @observe preserves function attributes."""
        mock_span = MagicMock()
        mock_telemetry.start_span.return_value = mock_span
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock()
        
        @observe(name="test_function")
        def original_function():
            """Original function docstring."""
            return "test"
        
        # Check that function attributes are preserved
        assert original_function.__name__ == "original_function"
        assert original_function.__doc__ == "Original function docstring."
    
    def test_observe_with_session_id(self, mock_telemetry):
        """Test @observe decorator with session_id."""
        mock_span = MagicMock()
        mock_telemetry.start_span.return_value = mock_span
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock()
        
        @observe(name="test_function", session_id="session_123")
        def test_function():
            return "session_test"
        
        result = test_function()
        
        assert result == "session_test"
        mock_telemetry.start_span.assert_called_once()
        mock_span.set_attributes.assert_called()
    
    def test_observe_with_user_id(self, mock_telemetry):
        """Test @observe decorator with user_id."""
        mock_span = MagicMock()
        mock_telemetry.start_span.return_value = mock_span
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock()
        
        @observe(name="test_function", user_id="user_456")
        def test_function():
            return "user_test"
        
        result = test_function()
        
        assert result == "user_test"
        mock_telemetry.start_span.assert_called_once()
        mock_span.set_attributes.assert_called()