"""Pytest configuration and fixtures for Brokle SDK tests."""

import pytest
from unittest.mock import MagicMock, AsyncMock
import asyncio

from brokle.config import Config


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config():
    """Create a test configuration."""
    return Config(
        api_key="bk_test_secret",
        host="https://test.brokle.com",
        environment="test"
    )


@pytest.fixture
def mock_http_client():
    """Create a mock HTTP client."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "success"}
    mock_client.request = AsyncMock(return_value=mock_response)
    mock_client.aclose = AsyncMock()
    return mock_client


@pytest.fixture
def mock_telemetry_manager():
    """Create a mock telemetry manager."""
    mock_manager = MagicMock()
    mock_span = MagicMock()
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock()
    mock_span.__aenter__ = AsyncMock(return_value=mock_span)
    mock_span.__aexit__ = AsyncMock()
    mock_manager.start_span.return_value = mock_span
    return mock_manager


@pytest.fixture
def sample_openai_response():
    """Create a sample OpenAI-compatible response."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-3.5-turbo",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 9,
            "total_tokens": 21
        }
    }


@pytest.fixture
def sample_streaming_chunks():
    """Create sample streaming response chunks."""
    return [
        {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-3.5-turbo",
            "choices": [{
                "index": 0,
                "delta": {"content": "Hello"},
                "finish_reason": None
            }]
        },
        {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-3.5-turbo",
            "choices": [{
                "index": 0,
                "delta": {"content": " there!"},
                "finish_reason": None
            }]
        },
        {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-3.5-turbo",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
    ]


@pytest.fixture
def sample_embeddings_response():
    """Create a sample embeddings response."""
    return {
        "object": "list",
        "data": [{
            "object": "embedding",
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
            "index": 0
        }],
        "model": "text-embedding-ada-002",
        "usage": {
            "prompt_tokens": 5,
            "total_tokens": 5
        }
    }


@pytest.fixture
def sample_analytics_response():
    """Create a sample analytics response."""
    return {
        "total_requests": 1500,
        "total_cost": 12.50,
        "average_latency": 250.5,
        "requests_by_model": {
            "gpt-3.5-turbo": 1200,
            "gpt-4": 300
        },
        "cost_by_provider": {
            "openai": 8.75,
            "anthropic": 3.75
        },
        "cache_hit_rate": 0.85
    }


@pytest.fixture
def sample_evaluation_response():
    """Create a sample evaluation response."""
    return {
        "evaluation_id": "eval_123",
        "scores": {
            "relevance": 0.95,
            "accuracy": 0.88,
            "helpfulness": 0.92,
            "safety": 1.0
        },
        "feedback": "High quality response with accurate information",
        "recommendations": [
            "Consider adding more specific examples",
            "Response could be more concise"
        ]
    }


# Async test utilities
@pytest.fixture
async def async_mock_client(mock_http_client):
    """Create an async mock client."""
    yield mock_http_client


# Test data generators
@pytest.fixture
def generate_chat_messages():
    """Generate chat messages for testing."""
    def _generate(count=3):
        messages = []
        for i in range(count):
            role = "user" if i % 2 == 0 else "assistant"
            content = f"Test message {i + 1}"
            messages.append({"role": role, "content": content})
        return messages
    return _generate


@pytest.fixture
def generate_test_metadata():
    """Generate test metadata."""
    def _generate(user_id="user_123", session_id="session_456"):
        return {
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": "2024-01-15T10:30:00Z",
            "environment": "test",
            "version": "1.0.0"
        }
    return _generate


# Cleanup fixtures
@pytest.fixture(autouse=True)
def reset_client_state():
    """Reset client state before and after each test."""
    # Clear singleton instance
    import brokle.client
    brokle.client._client_singleton = None

    # Clear observability context
    from brokle.observability.context import clear_context
    clear_context()

    yield

    # Cleanup after test
    brokle.client._client_singleton = None
    clear_context()


@pytest.fixture(autouse=True)
def reset_telemetry():
    """Reset telemetry state before each test."""
    # This would reset any global telemetry state if needed
    yield
    # Cleanup telemetry resources
