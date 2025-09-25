"""
Tests for new Brokle client architecture.

Tests sync/async clients, resource organization, and OpenAI compatibility.
"""

import pytest
import httpx
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from brokle.client import Brokle, AsyncBrokle, get_client
from brokle.exceptions import NetworkError, AuthenticationError


class TestBrokleClient:
    """Test sync Brokle client."""

    def test_init_with_parameters(self):
        """Test initialization with explicit parameters."""
        client = Brokle(
            api_key="ak_test123",
            host="http://localhost:8080",
            project_id="proj_test",
            environment="test"
        )

        assert client.config.api_key == "ak_test123"
        assert client.config.host == "http://localhost:8080"
        assert client.config.project_id == "proj_test"
        assert client.config.environment == "test"

        # Check resources are initialized
        assert hasattr(client, "chat")
        assert hasattr(client, "embeddings")
        assert hasattr(client, "models")
        assert hasattr(client.chat, "completions")

    def test_context_manager(self):
        """Test context manager functionality."""
        with patch.dict("os.environ", {
            "BROKLE_API_KEY": "ak_test",
            "BROKLE_PROJECT_ID": "proj_test"
        }):
            with Brokle() as client:
                assert client._client is None  # Not created until first use
                # Use client to trigger HTTP client creation
                assert client._get_client() is not None

            # After context exit, client should be closed
            assert client._client is None or client._client.is_closed

    def test_explicit_close(self):
        """Test explicit client cleanup."""
        with patch.dict("os.environ", {
            "BROKLE_API_KEY": "ak_test",
            "BROKLE_PROJECT_ID": "proj_test"
        }):
            client = Brokle()
            http_client = client._get_client()
            assert http_client is not None

            client.close()
            assert client._client is None

    @patch('httpx.Client')
    def test_request_success(self, mock_httpx_client):
        """Test successful HTTP request."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello!"}}],
            "brokle_metadata": {"provider": "openai", "request_id": "req_123", "latency_ms": 150}
        }

        # Mock client
        mock_client_instance = Mock()
        mock_client_instance.request.return_value = mock_response
        mock_httpx_client.return_value = mock_client_instance

        # Test request
        with patch.dict("os.environ", {
            "BROKLE_API_KEY": "ak_test",
            "BROKLE_PROJECT_ID": "proj_test"
        }):
            client = Brokle()
            result = client.request("POST", "/v1/chat/completions", json={"model": "gpt-4"})

            assert result["choices"][0]["message"]["content"] == "Hello!"
            assert result["brokle_metadata"]["provider"] == "openai"

    @patch('httpx.Client')
    def test_request_network_error(self, mock_httpx_client):
        """Test network error handling."""
        # Mock network error
        mock_client_instance = Mock()
        mock_client_instance.request.side_effect = httpx.ConnectError("Connection failed")
        mock_httpx_client.return_value = mock_client_instance

        # Test error handling
        with patch.dict("os.environ", {
            "BROKLE_API_KEY": "ak_test",
            "BROKLE_PROJECT_ID": "proj_test"
        }):
            client = Brokle()

            with pytest.raises(NetworkError, match="Failed to connect"):
                client.request("POST", "/v1/chat/completions")

    def test_chat_completions_create(self):
        """Test chat completions creation."""
        with patch.dict("os.environ", {
            "BROKLE_API_KEY": "ak_test",
            "BROKLE_PROJECT_ID": "proj_test"
        }):
            client = Brokle()

            # Mock the request method
            mock_response = {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15
                },
                "brokle_metadata": {
                    "provider": "openai",
                    "request_id": "req_123",
                    "latency_ms": 150
                }
            }

            with patch.object(client, 'request', return_value=mock_response):
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Hello!"}],
                    routing_strategy="cost_optimized"
                )

                assert response.model == "gpt-4"
                assert response.choices[0].message.content == "Hello!"
                assert response.brokle_metadata.provider == "openai"

    def test_get_client_function(self):
        """Test get_client function."""
        with patch.dict("os.environ", {
            "BROKLE_API_KEY": "ak_test",
            "BROKLE_PROJECT_ID": "proj_test"
        }):
            client = get_client()
            assert isinstance(client, Brokle)
            assert client.config.api_key == "ak_test"


class TestAsyncBrokleClient:
    """Test async Brokle client."""

    def test_init_with_parameters(self):
        """Test async client initialization."""
        client = AsyncBrokle(
            api_key="ak_test123",
            host="http://localhost:8080",
            project_id="proj_test",
            environment="test"
        )

        assert client.config.api_key == "ak_test123"
        assert client.config.host == "http://localhost:8080"
        assert client.config.project_id == "proj_test"
        assert client.config.environment == "test"

        # Check async resources are initialized
        assert hasattr(client, "chat")
        assert hasattr(client, "embeddings")
        assert hasattr(client, "models")
        assert hasattr(client.chat, "completions")

        # Check HTTP client is initialized immediately
        assert client._client is not None

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context manager functionality."""
        with patch.dict("os.environ", {
            "BROKLE_API_KEY": "ak_test",
            "BROKLE_PROJECT_ID": "proj_test"
        }):
            async with AsyncBrokle() as client:
                assert client._client is not None
                assert not client._client.is_closed

    @pytest.mark.asyncio
    async def test_explicit_close(self):
        """Test explicit async client cleanup."""
        with patch.dict("os.environ", {
            "BROKLE_API_KEY": "ak_test",
            "BROKLE_PROJECT_ID": "proj_test"
        }):
            client = AsyncBrokle()
            assert client._client is not None

            await client.close()

    @pytest.mark.asyncio
    async def test_async_request_success(self):
        """Test successful async HTTP request."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello async!"}}],
            "brokle_metadata": {"provider": "openai", "request_id": "req_async", "latency_ms": 120}
        }

        with patch.dict("os.environ", {
            "BROKLE_API_KEY": "ak_test",
            "BROKLE_PROJECT_ID": "proj_test"
        }):
            client = AsyncBrokle()

            # Mock the async client request
            with patch.object(client._client, 'request', new_callable=AsyncMock) as mock_request:
                mock_request.return_value = mock_response

                result = await client.request("POST", "/v1/chat/completions", json={"model": "gpt-4"})

                assert result["choices"][0]["message"]["content"] == "Hello async!"
                assert result["brokle_metadata"]["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_async_chat_completions_create(self):
        """Test async chat completions creation."""
        with patch.dict("os.environ", {
            "BROKLE_API_KEY": "ak_test",
            "BROKLE_PROJECT_ID": "proj_test"
        }):
            client = AsyncBrokle()

            # Mock response
            mock_response = {
                "id": "chatcmpl-async-123",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello async!"},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15
                },
                "brokle_metadata": {
                    "provider": "openai",
                    "request_id": "req_async",
                    "latency_ms": 120
                }
            }

            with patch.object(client, 'request', new_callable=AsyncMock) as mock_request:
                mock_request.return_value = mock_response

                response = await client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Hello async!"}],
                    routing_strategy="quality_optimized"
                )

                assert response.model == "gpt-4"
                assert response.choices[0].message.content == "Hello async!"
                assert response.brokle_metadata.provider == "openai"


class TestEmbeddingsResource:
    """Test embeddings resource."""

    def test_embeddings_create(self):
        """Test embeddings creation."""
        with patch.dict("os.environ", {
            "BROKLE_API_KEY": "ak_test",
            "BROKLE_PROJECT_ID": "proj_test"
        }):
            client = Brokle()

            # Mock response
            mock_response = {
                "object": "list",
                "data": [{
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1, 0.2, 0.3]
                }],
                "model": "text-embedding-3-small",
                "usage": {
                    "prompt_tokens": 5,
                    "total_tokens": 5
                },
                "brokle_metadata": {
                    "provider": "openai",
                    "request_id": "req_emb_123",
                    "latency_ms": 80
                }
            }

            with patch.object(client, 'request', return_value=mock_response):
                response = client.embeddings.create(
                    input="Hello world",
                    model="text-embedding-3-small",
                    cache_strategy="semantic"
                )

                assert response.model == "text-embedding-3-small"
                assert len(response.data) == 1
                assert response.data[0].embedding == [0.1, 0.2, 0.3]
                assert response.brokle_metadata.provider == "openai"


class TestModelsResource:
    """Test models resource."""

    def test_models_list(self):
        """Test models listing."""
        with patch.dict("os.environ", {
            "BROKLE_API_KEY": "ak_test",
            "BROKLE_PROJECT_ID": "proj_test"
        }):
            client = Brokle()

            # Mock response
            mock_response = {
                "object": "list",
                "data": [
                    {
                        "id": "gpt-4",
                        "object": "model",
                        "created": 1234567890,
                        "owned_by": "openai",
                        "provider": "openai",
                        "category": "chat",
                        "capabilities": ["chat"],
                        "cost_per_token": 0.00003,
                        "context_length": 8192,
                        "availability": "available"
                    }
                ]
            }

            with patch.object(client, 'request', return_value=mock_response):
                response = client.models.list(provider="openai", category="chat")

                assert len(response.data) == 1
                assert response.data[0].id == "gpt-4"
                assert response.data[0].provider == "openai"
                assert response.data[0].category == "chat"

    def test_models_retrieve(self):
        """Test model retrieval."""
        with patch.dict("os.environ", {
            "BROKLE_API_KEY": "ak_test",
            "BROKLE_PROJECT_ID": "proj_test"
        }):
            client = Brokle()

            # Mock response
            mock_response = {
                "id": "gpt-4",
                "object": "model",
                "created": 1234567890,
                "owned_by": "openai",
                "provider": "openai",
                "category": "chat",
                "capabilities": ["chat"],
                "cost_per_token": 0.00003,
                "context_length": 8192,
                "availability": "available"
            }

            with patch.object(client, 'request', return_value=mock_response):
                model = client.models.retrieve("gpt-4")

                assert model.id == "gpt-4"
                assert model.provider == "openai"
                assert model.category == "chat"