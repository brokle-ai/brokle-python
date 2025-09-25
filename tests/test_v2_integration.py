"""
Clean v2.0 Integration Tests

Tests the actual public v2.0 API without deprecated internal methods.
"""

import pytest
import os

from brokle import Brokle, get_client
from brokle.config import Config
from brokle.exceptions import AuthenticationError


class TestV2Integration:
    """Test v2.0 integration patterns."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config(
            api_key="ak_test_key",
            project_id="proj_test",
            host="https://api.brokle.ai",
            otel_enabled=False
        )

    def test_pattern_3_native_sdk(self, config):
        """Test Pattern 3: Native SDK usage."""
        # Direct instantiation with config
        client = Brokle(config=config)

        # Verify client has expected resources
        assert hasattr(client, 'chat')
        assert hasattr(client, 'embeddings')
        assert hasattr(client, 'models')

        # Verify configuration
        assert client.config.api_key == "ak_test_key"
        assert client.config.project_id == "proj_test"

    def test_pattern_3_with_kwargs(self):
        """Test Pattern 3: Native SDK with kwargs."""
        client = Brokle(
            api_key="ak_kwargs_key",
            project_id="proj_kwargs",
            environment="staging",
            otel_enabled=False
        )

        assert client.config.api_key == "ak_kwargs_key"
        assert client.config.project_id == "proj_kwargs"
        assert client.config.environment == "staging"

    def test_pattern_1_2_get_client(self, monkeypatch):
        """Test Pattern 1/2: get_client() from environment."""
        # Set environment variables
        monkeypatch.setenv("BROKLE_API_KEY", "ak_env_key")
        monkeypatch.setenv("BROKLE_PROJECT_ID", "proj_env")
        monkeypatch.setenv("BROKLE_HOST", "https://api.brokle.ai")

        # get_client() should use environment variables
        client = get_client()

        assert client.config.api_key == "ak_env_key"
        assert client.config.project_id == "proj_env"
        assert client.config.host == "https://api.brokle.ai"

    def test_client_lifecycle(self, config):
        """Test client lifecycle operations."""
        client = Brokle(config=config)

        # Context manager usage
        with client:
            assert isinstance(client, Brokle)

        # Explicit close (should not raise errors)
        client.close()

    def test_client_http_preparation(self, config):
        """Test client HTTP preparation (public interface only)."""
        client = Brokle(config=config)

        # Test URL preparation (if it's a public method)
        if hasattr(client, '_prepare_url'):
            url = client._prepare_url('/v1/chat/completions')
            assert url.endswith('/v1/chat/completions')

    def test_environment_configuration_handling(self):
        """Test various environment configurations."""
        # Test with environment name
        client = Brokle(
            api_key="ak_test",
            project_id="proj_test",
            environment="production",
            otel_enabled=False
        )

        assert client.config.environment == "production"

        # Test with custom host
        client2 = Brokle(
            api_key="ak_test",
            project_id="proj_test",
            host="https://custom.brokle.ai",
            otel_enabled=False
        )

        assert client2.config.host == "https://custom.brokle.ai"

    def test_error_handling_patterns(self, monkeypatch):
        """Test error handling in v2.0."""
        # Clear environment variables
        monkeypatch.delenv("BROKLE_API_KEY", raising=False)
        monkeypatch.delenv("BROKLE_PROJECT_ID", raising=False)

        # Should raise AuthenticationError when no credentials
        with pytest.raises(AuthenticationError, match="API key is required"):
            Brokle(otel_enabled=False)

    def test_configuration_precedence(self, monkeypatch):
        """Test configuration precedence (explicit > env vars)."""
        # Set environment variables
        monkeypatch.setenv("BROKLE_API_KEY", "ak_env_key")
        monkeypatch.setenv("BROKLE_PROJECT_ID", "proj_env")

        # Explicit parameters should override environment
        client = Brokle(
            api_key="ak_explicit_key",
            project_id="proj_explicit",
            otel_enabled=False
        )

        assert client.config.api_key == "ak_explicit_key"
        assert client.config.project_id == "proj_explicit"

    def test_client_string_representation(self, config):
        """Test client has reasonable string representation."""
        client = Brokle(config=config)
        repr_str = repr(client)

        # Should contain some identifying information
        assert "Brokle" in repr_str or "brokle" in repr_str.lower()