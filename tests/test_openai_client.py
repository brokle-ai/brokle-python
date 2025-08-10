"""Tests for OpenAI client integration."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from brokle.openai.client import OpenAI
from brokle.config import Config


class TestOpenAIClient:
    """Test OpenAI client functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config(
            api_key="ak_test_key",
            project_id="proj_test",
            host="https://test.example.com"
        )
    
    @pytest.fixture
    def client(self, config):
        """Create test OpenAI client."""
        return OpenAI(config=config)
    
    def test_init_with_config(self, config):
        """Test OpenAI client initialization with config."""
        client = OpenAI(config=config)
        assert client.config == config
        assert client.config.api_key == "ak_test_key"
    
    def test_init_with_kwargs(self):
        """Test OpenAI client initialization with kwargs."""
        client = OpenAI(
            api_key="ak_test_key",
            project_id="proj_test",
            host="https://test.example.com"
        )
        assert client.config.api_key == "ak_test_key"
        assert client.config.project_id == "proj_test"
        assert client.config.host == "https://test.example.com"
    
    def test_init_with_openai_api_key(self):
        """Test OpenAI client initialization with OpenAI API key."""
        # Should convert OpenAI API key to Brokle format
        client = OpenAI(api_key="sk-openai-key")
        assert client.config.api_key == "sk-openai-key"
    
    @pytest.mark.asyncio
    async def test_chat_completions_create(self, client):
        """Test chat completions creation."""
        with patch.object(client, '_brokle_client') as mock_brokle:
            mock_response = {
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
            mock_brokle.completions.create = AsyncMock(return_value=mock_response)
            
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello!"}]
            )
            
            assert response["id"] == "chatcmpl-123"
            assert response["choices"][0]["message"]["content"] == "Hello! How can I help you today?"
            mock_brokle.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_chat_completions_create_with_stream(self, client):
        """Test chat completions creation with streaming."""
        with patch.object(client, '_brokle_client') as mock_brokle:
            # Mock streaming response
            mock_chunks = [
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
                        "delta": {"content": "!"},
                        "finish_reason": "stop"
                    }]
                }
            ]
            
            async def mock_stream():
                for chunk in mock_chunks:
                    yield chunk
            
            mock_brokle.completions.create = AsyncMock(return_value=mock_stream())
            
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello!"}],
                stream=True
            )
            
            # Collect streaming chunks
            chunks = []
            async for chunk in response:
                chunks.append(chunk)
            
            assert len(chunks) == 2
            assert chunks[0]["choices"][0]["delta"]["content"] == "Hello"
            assert chunks[1]["choices"][0]["delta"]["content"] == "!"
            mock_brokle.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_completions_create(self, client):
        """Test text completions creation."""
        with patch.object(client, '_brokle_client') as mock_brokle:
            mock_response = {
                "id": "cmpl-123",
                "object": "text_completion",
                "created": 1677652288,
                "model": "gpt-3.5-turbo-instruct",
                "choices": [{
                    "text": "The capital of France is Paris.",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 7,
                    "total_tokens": 12
                }
            }
            mock_brokle.completions.create = AsyncMock(return_value=mock_response)
            
            response = await client.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt="The capital of France is"
            )
            
            assert response["id"] == "cmpl-123"
            assert response["choices"][0]["text"] == "The capital of France is Paris."
            mock_brokle.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_embeddings_create(self, client):
        """Test embeddings creation."""
        with patch.object(client, '_brokle_client') as mock_brokle:
            mock_response = {
                "object": "list",
                "data": [{
                    "object": "embedding",
                    "embedding": [0.1, 0.2, 0.3],
                    "index": 0
                }],
                "model": "text-embedding-ada-002",
                "usage": {
                    "prompt_tokens": 5,
                    "total_tokens": 5
                }
            }
            mock_brokle.embeddings.create = AsyncMock(return_value=mock_response)
            
            response = await client.embeddings.create(
                model="text-embedding-ada-002",
                input="Hello world"
            )
            
            assert response["object"] == "list"
            assert len(response["data"]) == 1
            assert response["data"][0]["embedding"] == [0.1, 0.2, 0.3]
            mock_brokle.embeddings.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_with_brokle_params(self, client):
        """Test OpenAI client with Brokle specific parameters."""
        with patch.object(client, '_brokle_client') as mock_brokle:
            mock_response = {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "gpt-3.5-turbo",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello!"
                    },
                    "finish_reason": "stop"
                }]
            }
            mock_brokle.completions.create = AsyncMock(return_value=mock_response)
            
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello!"}],
                # Brokle specific parameters
                routing_strategy="cost_optimized",
                enable_cache=True,
                max_cost=0.10,
                tags=["test", "api"]
            )
            
            assert response["id"] == "chatcmpl-123"
            mock_brokle.completions.create.assert_called_once()
            
            # Check that Brokle parameters were passed
            call_args = mock_brokle.completions.create.call_args
            assert call_args[1]["routing_strategy"] == "cost_optimized"
            assert call_args[1]["enable_cache"] is True
            assert call_args[1]["max_cost"] == 0.10
            assert call_args[1]["tags"] == ["test", "api"]
    
    @pytest.mark.asyncio
    async def test_error_handling(self, client):
        """Test error handling in OpenAI client."""
        with patch.object(client, '_brokle_client') as mock_brokle:
            from brokle.exceptions import AuthenticationError
            mock_brokle.completions.create = AsyncMock(side_effect=AuthenticationError("Invalid API key"))
            
            with pytest.raises(AuthenticationError):
                await client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello!"}]
                )
    
    @pytest.mark.asyncio
    async def test_context_manager(self, config):
        """Test OpenAI client as async context manager."""
        async with OpenAI(config=config) as client:
            assert isinstance(client, OpenAI)
            # Client should be properly initialized
            assert client.config == config
        # Should not raise - client should be properly closed
    
    def test_sync_context_manager(self, config):
        """Test OpenAI client as sync context manager."""
        with OpenAI(config=config) as client:
            assert isinstance(client, OpenAI)
            assert client.config == config
        # Should not raise - client should be properly closed