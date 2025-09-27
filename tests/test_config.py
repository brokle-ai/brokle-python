"""Tests for configuration module."""

import os
from unittest.mock import patch

import pytest

from brokle.config import Config, sanitize_environment_name, validate_environment_name


class TestConfig:
    """Test configuration functionality."""

    def test_config_from_env(self):
        """Test configuration from environment variables."""
        with patch.dict(
            os.environ,
            {
                "BROKLE_API_KEY": "bk_test_secret",
                "BROKLE_HOST": "https://test.example.com",
                "BROKLE_ENVIRONMENT": "test",
            },
        ):
            config = Config.from_env()
            assert config.api_key == "bk_test_secret"
            assert config.host == "https://test.example.com"
            assert config.environment == "test"

    def test_config_validation(self):
        """Test configuration validation."""
        config = Config(api_key="bk_test_secret")

        # Should not raise
        config.validate()

        # Missing API key should raise
        config.api_key = None
        with pytest.raises(ValueError, match="API key is required"):
            config.validate()

    def test_api_key_validation(self):
        """Test API key format validation."""
        # Valid API key
        config = Config(api_key="bk_test_secret")
        assert config.api_key == "bk_test_secret"

        # Invalid API key format
        with pytest.raises(ValueError, match='API key must start with "bk_"'):
            Config(api_key="invalid_key")

    def test_host_validation(self):
        """Test host URL validation."""
        # Valid hosts
        config = Config(host="http://localhost:8080")
        assert config.host == "http://localhost:8080"

        config = Config(host="https://example.com/")
        assert config.host == "https://example.com"  # Trailing slash removed

        # Invalid host format
        with pytest.raises(
            ValueError, match="Host must start with http:// or https://"
        ):
            Config(host="invalid_host")

    def test_get_headers(self):
        """Test getting HTTP headers."""
        config = Config(api_key="bk_test_secret", environment="test")

        headers = config.get_headers()
        assert headers["X-API-Key"] == "bk_test_secret"
        assert headers["X-Environment"] == "test"
        assert headers["Content-Type"] == "application/json"
        assert "brokle-python" in headers["User-Agent"]

    def test_environment_validation(self):
        """Test environment name validation."""
        # Valid environments should work
        Config(environment="test")
        Config(environment="staging")
        Config(environment="default")

        # Invalid environments should raise errors
        with pytest.raises(ValueError, match="Environment name must be lowercase"):
            Config(environment="PRODUCTION")

        with pytest.raises(ValueError, match="Environment name too long"):
            Config(environment="a" * 41)

        with pytest.raises(
            ValueError, match="Environment name cannot start with 'brokle' prefix"
        ):
            Config(environment="brokle-test")

        with pytest.raises(ValueError, match="Environment name cannot be empty"):
            Config(environment="")

    def test_validate_environment_name_function(self):
        """Test the validate_environment_name function directly."""
        # Valid names should not raise
        validate_environment_name("test")
        validate_environment_name("staging")
        validate_environment_name("default")
        validate_environment_name("dev")

        # Invalid names should raise
        with pytest.raises(ValueError, match="must be lowercase"):
            validate_environment_name("TEST")

        with pytest.raises(ValueError, match="too long"):
            validate_environment_name("a" * 41)

        with pytest.raises(ValueError, match="brokle"):
            validate_environment_name("brokle-env")

        with pytest.raises(ValueError, match="cannot be empty"):
            validate_environment_name("")

    def test_sanitize_environment_name_function(self):
        """Test the sanitize_environment_name function."""
        assert sanitize_environment_name("TEST") == "test"
        assert sanitize_environment_name("Production") == "production"
        assert sanitize_environment_name("a" * 50) == "a" * 40
        # Test that brokle prefix throws error in sanitization too
        with pytest.raises(ValueError, match="brokle"):
            sanitize_environment_name("brokle-test")
        with pytest.raises(ValueError, match="brokle"):
            sanitize_environment_name("brokle")
        assert sanitize_environment_name("") == "default"
        assert sanitize_environment_name("valid_env") == "valid_env"


def test_from_env_respects_flags():
    """Ensure boolean/int environment values convert correctly."""
    with patch.dict(
        os.environ,
        {
            "BROKLE_TELEMETRY_ENABLED": "false",
            "BROKLE_MAX_RETRIES": "5",
        },
    ):
        config = Config.from_env()
        assert config.telemetry_enabled is False
        assert config.max_retries == 5
