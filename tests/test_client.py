"""Tests for core Brokle client functionality."""

import pytest
from unittest.mock import patch, MagicMock
import asyncio

from brokle import Brokle
from brokle.config import Config
from brokle._client.span import BrokleSpan, BrokleGeneration


class TestBrokleClient:
    """Test essential Brokle client functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config(
            api_key="ak_test_key",
            project_id="proj_test",
            host="https://test.example.com",
            otel_enabled=False
        )

    def test_client_initialization_with_config(self, config):
        """Test client initialization with config object."""
        client = Brokle(config=config)

        assert client.config.api_key == "ak_test_key"
        assert client.config.project_id == "proj_test"
        assert client.config.otel_enabled is False

    def test_client_initialization_with_kwargs(self):
        """Test client initialization with keyword arguments."""
        client = Brokle(
            api_key="ak_kwargs_key",
            project_id="proj_kwargs",
            environment="staging",
            otel_enabled=False
        )

        assert client.config.api_key == "ak_kwargs_key"
        assert client.config.project_id == "proj_kwargs"
        assert client.config.environment == "staging"

    def test_client_initialization_missing_credentials(self):
        """Test client initialization with missing credentials."""
        client = Brokle(otel_enabled=False)

        assert client.config.api_key == "ak_fake"
        assert client.config.project_id == "fake"
        assert client.config.otel_enabled is False

    def test_client_has_required_components(self, config):
        """Test client has required components."""
        client = Brokle(config=config)

        assert hasattr(client, 'auth_manager')
        assert hasattr(client, '_http_client')
        assert client.auth_manager is not None

    def test_client_span_creation(self, config):
        """Test basic span creation."""
        client = Brokle(config=config)

        span = client.span("test-span")
        assert isinstance(span, BrokleSpan)
        assert span.name == "test-span"

        # Test with metadata
        span_with_meta = client.span(
            "meta-span",
            metadata={"user_id": "user123"},
            tags=["test"]
        )
        assert span_with_meta.metadata["user_id"] == "user123"
        assert "test" in span_with_meta.tags

    def test_client_generation_creation(self, config):
        """Test generation creation."""
        client = Brokle(config=config)

        generation = client.generation(
            "test-generation",
            model="gpt-4",
            provider="openai"
        )

        assert isinstance(generation, BrokleGeneration)
        assert generation.name == "test-generation"
        assert generation.model == "gpt-4"
        assert generation.provider == "openai"

    @pytest.mark.asyncio
    async def test_client_http_requests(self, config):
        """Test client HTTP request functionality."""
        with patch('httpx.AsyncClient.request') as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {"status": "success"}
            mock_response.raise_for_status.return_value = None
            mock_request.return_value = mock_response

            client = Brokle(config=config)
            result = await client._make_request("POST", "/api/test", data={"test": "data"})

            assert result == {"status": "success"}

            # Verify auth headers were included
            call_args = mock_request.call_args
            headers = call_args.kwargs['headers']
            assert "X-API-Key" in headers
            assert "X-Project-ID" in headers
            assert headers["X-API-Key"] == "ak_test_key"

    def test_client_lifecycle_operations(self, config):
        """Test client lifecycle operations."""
        client = Brokle(config=config)

        # Should not raise errors
        client.flush()

        # Test tracer property
        assert client.tracer is not None

    @pytest.mark.asyncio
    async def test_client_async_operations(self, config):
        """Test client async operations."""
        client = Brokle(config=config)

        # Async shutdown should not raise
        await client.shutdown()

    def test_client_environment_configuration(self):
        """Test client with different environment configurations."""
        client = Brokle(
            api_key="ak_env_test",
            project_id="proj_env",
            environment="production",
            host="https://custom.brokle.com",
            otel_enabled=False
        )

        assert client.config.environment == "production"
        assert client.config.host == "https://custom.brokle.com"

        headers = client.auth_manager.get_auth_headers()
        assert headers["X-Environment"] == "production"

    def test_client_error_handling(self):
        """Test client error handling scenarios."""
        # Invalid API key format should raise error
        with pytest.raises(Exception):
            Brokle(
                api_key="invalid_key_format",
                project_id="proj_test",
                otel_enabled=False
            )

    def test_client_string_representation(self, config):
        """Test client string representation."""
        client = Brokle(config=config)

        repr_str = repr(client)
        assert "Brokle" in repr_str
        assert "proj_test" in repr_str
        # Should not expose sensitive API key
        assert "ak_test_key" not in repr_str