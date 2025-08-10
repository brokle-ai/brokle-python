"""Tests for the core client module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from brokle.client import Brokle, get_client
from brokle.config import Config
from brokle.exceptions import BrokleError, AuthenticationError, RateLimitError


class TestBrokle:
    """Test Brokle client functionality."""
    
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
        """Create test client."""
        return Brokle(config)
    
    def test_init_with_config(self, config):
        """Test client initialization with config."""
        client = Brokle(config)
        assert client.config == config
        assert client.config.api_key == "ak_test_key"
    
    def test_init_with_kwargs(self):
        """Test client initialization with kwargs."""
        client = Brokle(
            api_key="ak_test_key",
            project_id="proj_test",
            host="https://test.example.com"
        )
        assert client.config.api_key == "ak_test_key"
        assert client.config.project_id == "proj_test"
        assert client.config.host == "https://test.example.com"
    
    @pytest.mark.asyncio
    async def test_http_request_success(self, client):
        """Test successful HTTP request."""
        with patch.object(client, '_client') as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": "test"}
            mock_client.request = AsyncMock(return_value=mock_response)
            
            result = await client._request("POST", "/test", {"key": "value"})
            
            assert result == {"data": "test"}
            mock_client.request.assert_called_once_with(
                "POST",
                "https://test.example.com/test",
                json={"key": "value"},
                headers=client.config.get_headers()
            )
    
    @pytest.mark.asyncio
    async def test_http_request_auth_error(self, client):
        """Test authentication error handling."""
        with patch.object(client, '_client') as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_response.json.return_value = {"error": "Unauthorized"}
            mock_client.request = AsyncMock(return_value=mock_response)
            
            with pytest.raises(AuthenticationError):
                await client._request("POST", "/test", {"key": "value"})
    
    @pytest.mark.asyncio
    async def test_http_request_rate_limit_error(self, client):
        """Test rate limit error handling."""
        with patch.object(client, '_client') as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.json.return_value = {"error": "Rate limit exceeded"}
            mock_client.request = AsyncMock(return_value=mock_response)
            
            with pytest.raises(RateLimitError):
                await client._request("POST", "/test", {"key": "value"})
    
    @pytest.mark.asyncio
    async def test_http_request_generic_error(self, client):
        """Test generic error handling."""
        with patch.object(client, '_client') as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.json.return_value = {"error": "Internal server error"}
            mock_client.request = AsyncMock(return_value=mock_response)
            
            with pytest.raises(BrokleError):
                await client._request("POST", "/test", {"key": "value"})
    
    @pytest.mark.asyncio
    async def test_http_request_network_error(self, client):
        """Test network error handling."""
        with patch.object(client, '_client') as mock_client:
            mock_client.request = AsyncMock(side_effect=httpx.NetworkError("Connection failed"))
            
            with pytest.raises(BrokleError, match="Network error"):
                await client._request("POST", "/test", {"key": "value"})
    
    @pytest.mark.asyncio
    async def test_close(self, client):
        """Test client cleanup."""
        with patch.object(client, '_client') as mock_client:
            mock_client.aclose = AsyncMock()
            
            await client.close()
            mock_client.aclose.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_context_manager(self, config):
        """Test async context manager."""
        async with Brokle(config) as client:
            assert isinstance(client, Brokle)
        # Should not raise - client should be properly closed


class TestGetClient:
    """Test get_client function."""
    
    def test_get_client_with_config(self):
        """Test get_client with config object."""
        config = Config(api_key="ak_test_key", project_id="proj_test")
        client = get_client(config)
        
        assert isinstance(client, Brokle)
        assert client.config == config
    
    def test_get_client_with_kwargs(self):
        """Test get_client with kwargs."""
        client = get_client(
            api_key="ak_test_key",
            project_id="proj_test",
            host="https://test.example.com"
        )
        
        assert isinstance(client, Brokle)
        assert client.config.api_key == "ak_test_key"
        assert client.config.project_id == "proj_test"
        assert client.config.host == "https://test.example.com"
    
    def test_get_client_no_args(self):
        """Test get_client with no arguments."""
        with patch('brokle.client.get_config') as mock_get_config:
            mock_config = Config(api_key="ak_global", project_id="proj_global")
            mock_get_config.return_value = mock_config
            
            client = get_client()
            
            assert isinstance(client, Brokle)
            assert client.config == mock_config
            mock_get_config.assert_called_once()
    
    def test_get_client_mixed_args(self):
        """Test get_client with mixed arguments."""
        config = Config(api_key="ak_test_key", project_id="proj_test")
        
        with pytest.raises(ValueError, match="Cannot provide both config and keyword arguments"):
            get_client(config, api_key="ak_override")