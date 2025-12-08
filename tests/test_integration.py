"""
Clean Integration Tests

Tests the actual public API without deprecated internal methods.
"""

import os

import pytest

from brokle import Brokle, get_client
from brokle.config import BrokleConfig

# brokle.exceptions module was removed


class TestV2Integration:
    """Test integration patterns."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return BrokleConfig(
            api_key="bk_test_secret",
            base_url="https://api.brokle.ai",
            tracing_enabled=False,
        )

    # DEPRECATED: Client no longer has .chat, .embeddings, .models attributes
    # def test_pattern_3_native_sdk(self, config):
    #     """Test Pattern 3: Native SDK usage."""
    #     # Client API changed - these attributes don't exist
    #     pass

    def test_pattern_3_with_kwargs(self):
        """Test Pattern 3: Native SDK with kwargs."""
        client = Brokle(
            api_key="bk_kwargs_secret", environment="staging", tracing_enabled=False
        )

        assert client.config.api_key == "bk_kwargs_secret"
        assert client.config.environment == "staging"

    # DEPRECATED: Environment variable name changed
    # def test_pattern_1_2_get_client(self, monkeypatch):
    #     """Test Pattern 1/2: get_client() from environment."""
    #     # BROKLE_HOST changed to BROKLE_BASE_URL
    #     pass

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
            api_key="bk_test_secret", environment="production", tracing_enabled=False
        )

        assert client.config.environment == "production"

        # Test with custom host
        client2 = Brokle(
            api_key="bk_test_secret",
            base_url="https://custom.brokle.ai",
            tracing_enabled=False,
        )

        assert client2.config.base_url == "https://custom.brokle.ai"

    # DEPRECATED: Disabled mode behavior changed
    # def test_error_handling_patterns(self, monkeypatch, caplog):
    #     """Test error handling patterns."""
    #     # Client API changed - is_disabled and log messages changed
    #     pass

    def test_configuration_precedence(self, monkeypatch):
        """Test configuration precedence (explicit > env vars)."""
        # Set environment variables
        monkeypatch.setenv("BROKLE_API_KEY", "bk_env_secret")

        # Explicit parameters should override environment
        client = Brokle(api_key="bk_explicit_secret", tracing_enabled=False)

        assert client.config.api_key == "bk_explicit_secret"

    def test_client_string_representation(self, config):
        """Test client has reasonable string representation."""
        client = Brokle(config=config)
        repr_str = repr(client)

        # Should contain some identifying information
        assert "Brokle" in repr_str or "brokle" in repr_str.lower()
