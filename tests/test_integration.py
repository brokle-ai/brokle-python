"""
Clean Integration Tests

Tests the actual public API without deprecated internal methods.
"""

import os

import pytest

from brokle import Brokle, get_client
from brokle.config import Config
from brokle.exceptions import AuthenticationError


class TestV2Integration:
    """Test integration patterns."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config(
            api_key="bk_test_secret", host="https://api.brokle.ai", otel_enabled=False
        )

    def test_pattern_3_native_sdk(self, config):
        """Test Pattern 3: Native SDK usage."""
        # Direct instantiation with config
        client = Brokle(config=config)

        # Verify client has expected resources
        assert hasattr(client, "chat")
        assert hasattr(client, "embeddings")
        assert hasattr(client, "models")

        # Verify configuration
        assert client.config.api_key == "bk_test_secret"

    def test_pattern_3_with_kwargs(self):
        """Test Pattern 3: Native SDK with kwargs."""
        client = Brokle(
            api_key="bk_kwargs_secret", environment="staging", otel_enabled=False
        )

        assert client.config.api_key == "bk_kwargs_secret"
        assert client.config.environment == "staging"

    def test_pattern_1_2_get_client(self, monkeypatch):
        """Test Pattern 1/2: get_client() from environment."""
        # Set environment variables
        monkeypatch.setenv("BROKLE_API_KEY", "bk_env_secret")
        monkeypatch.setenv("BROKLE_HOST", "https://api.brokle.ai")

        # get_client() should use environment variables
        client = get_client()

        assert client.config.api_key == "bk_env_secret"
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
        if hasattr(client, "_prepare_url"):
            url = client._prepare_url("/v1/chat/completions")
            assert url.endswith("/v1/chat/completions")

    def test_environment_configuration_handling(self):
        """Test various environment configurations."""
        # Test with environment name
        client = Brokle(
            api_key="bk_test_secret", environment="production", otel_enabled=False
        )

        assert client.config.environment == "production"

        # Test with custom host
        client2 = Brokle(
            api_key="bk_test_secret",
            host="https://custom.brokle.ai",
            otel_enabled=False,
        )

        assert client2.config.host == "https://custom.brokle.ai"

    def test_error_handling_patterns(self, monkeypatch, caplog):
        """Test error handling patterns."""
        import logging

        # Clear environment variables
        monkeypatch.delenv("BROKLE_API_KEY", raising=False)

        # Should create disabled client and log warning when no credentials
        with caplog.at_level(logging.WARNING, logger="brokle"):
            client = Brokle(otel_enabled=False)

        # Should be disabled and log warning
        assert client.is_disabled is True
        assert "Authentication error:" in caplog.text
        assert "initialized without api_key" in caplog.text
        assert "Client will be disabled" in caplog.text

    def test_configuration_precedence(self, monkeypatch):
        """Test configuration precedence (explicit > env vars)."""
        # Set environment variables
        monkeypatch.setenv("BROKLE_API_KEY", "bk_env_secret")

        # Explicit parameters should override environment
        client = Brokle(api_key="bk_explicit_secret", otel_enabled=False)

        assert client.config.api_key == "bk_explicit_secret"

    def test_client_string_representation(self, config):
        """Test client has reasonable string representation."""
        client = Brokle(config=config)
        repr_str = repr(client)

        # Should contain some identifying information
        assert "Brokle" in repr_str or "brokle" in repr_str.lower()
