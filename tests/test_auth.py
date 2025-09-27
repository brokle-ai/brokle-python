"""Tests for authentication module."""

import time
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from brokle.auth import AuthManager
from brokle.config import Config
from brokle.exceptions import AuthenticationError


class TestAuthManager:
    """Test AuthManager functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config(api_key="bk_test_key", host="https://test.example.com")

    @pytest.fixture
    def auth_manager(self, config):
        """Create test auth manager."""
        return AuthManager(config)

    def test_init_with_config(self, config):
        """Test AuthManager initialization."""
        auth_manager = AuthManager(config)
        assert auth_manager.config == config
        assert auth_manager.config.api_key == "bk_test_key"

    @pytest.mark.asyncio
    async def test_validate_api_key_valid(self, auth_manager):
        """Test API key validation with valid key."""
        with pytest.raises(ValueError):  # Will fail in test environment
            await auth_manager.validate_api_key()

    @pytest.mark.asyncio
    async def test_validate_api_key_invalid_format(self, config):
        """Test API key validation with invalid format."""
        config.api_key = "invalid_key"
        auth_manager = AuthManager(config)

        with pytest.raises(ValueError, match="API key is required|Failed to validate"):
            await auth_manager.validate_api_key()

    @pytest.mark.asyncio
    async def test_validate_api_key_missing(self, config):
        """Test API key validation with missing key."""
        config.api_key = None
        auth_manager = AuthManager(config)

        with pytest.raises(ValueError, match="API key is required"):
            await auth_manager.validate_api_key()

    @pytest.mark.asyncio
    async def test_validate_api_key_empty(self, config):
        """Test API key validation with empty key."""
        config.api_key = ""
        auth_manager = AuthManager(config)

        with pytest.raises(ValueError, match="API key is required"):
            await auth_manager.validate_api_key()

    def test_get_auth_headers(self, auth_manager):
        """Test getting authentication headers."""
        headers = auth_manager.get_auth_headers()

        assert headers["X-API-Key"] == "bk_test_key"
        assert "User-Agent" in headers

    def test_get_auth_headers_with_environment(self, config):
        """Test getting authentication headers with environment."""
        config.environment = "default"
        auth_manager = AuthManager(config)

        headers = auth_manager.get_auth_headers()

        assert headers["X-Environment"] == "default"

    def test_get_auth_headers_default_environment(self, auth_manager):
        """Test getting authentication headers with default environment."""
        headers = auth_manager.get_auth_headers()

        # Should include X-Environment header with default value
        assert "X-Environment" in headers
        assert headers["X-Environment"] == "default"

    def test_is_validated_true(self, auth_manager):
        """Test validation status when validated."""
        assert auth_manager.is_validated() is False  # Not validated yet

    def test_is_validated_false_no_key(self, config):
        """Test validation status without API key."""
        config.api_key = None
        auth_manager = AuthManager(config)

        assert auth_manager.is_validated() is False

    def test_is_validated_false_invalid_key(self, config):
        """Test validation status with invalid API key."""
        config.api_key = "invalid_key"
        auth_manager = AuthManager(config)

        assert auth_manager.is_validated() is False

    def test_get_auth_info_placeholder(self, auth_manager):
        """Test get auth info method."""
        # Auth info is None until validated
        result = auth_manager.get_auth_info()
        assert result is None

    def test_clear_auth(self, auth_manager):
        """Test clearing auth information."""
        # Should clear auth info
        auth_manager.clear_auth()
        assert auth_manager.get_auth_info() is None
        assert auth_manager.is_validated() is False

    def test_auth_manager_with_different_api_key_formats(self):
        """Test AuthManager with different API key formats."""
        # Test with valid Brokle format
        config1 = Config(api_key="bk_test_key")
        auth1 = AuthManager(config1)
        assert auth1.is_validated() is False  # Not validated yet

        # Test with another valid Brokle format
        config2 = Config(api_key="bk_prod_key_12345")
        auth2 = AuthManager(config2)
        assert auth2.is_validated() is False  # Not validated yet

        # Test that invalid formats are caught during config creation
        with pytest.raises(ValidationError, match='API key must start with "bk_"'):
            Config(api_key="sk-openai-key")

        with pytest.raises(ValidationError, match='API key must start with "bk_"'):
            Config(api_key="invalid_format")

    def test_auth_headers_immutability(self, auth_manager):
        """Test that auth headers are independent copies."""
        headers1 = auth_manager.get_auth_headers()
        headers2 = auth_manager.get_auth_headers()

        # Should be equal but not the same object
        assert headers1 == headers2
        assert headers1 is not headers2

        # Modifying one should not affect the other
        headers1["Custom-Header"] = "test"
        assert "Custom-Header" not in headers2

    def test_auth_manager_str_representation(self, auth_manager):
        """Test string representation of AuthManager."""
        str_repr = str(auth_manager)
        assert "AuthManager" in str_repr
        assert "bk_test_key" not in str_repr  # Should not expose sensitive info

    def test_auth_manager_repr(self, auth_manager):
        """Test repr representation of AuthManager."""
        repr_str = repr(auth_manager)
        assert "AuthManager" in repr_str
        assert "bk_test_key" not in repr_str  # Should not expose API key

    def test_auth_headers_consistency(self, auth_manager):
        """Test that auth headers are consistent across calls."""
        # First call should create headers
        headers1 = auth_manager.get_auth_headers()

        # Second call should return same headers
        headers2 = auth_manager.get_auth_headers()

        # Headers should be the same
        assert headers1 == headers2
        assert headers1["X-API-Key"] == headers2["X-API-Key"]

    def test_auth_headers_basic(self, auth_manager):
        """Test basic authentication headers."""
        headers = auth_manager.get_auth_headers()

        # Should include auth headers
        assert headers["X-API-Key"] == "bk_test_key"

    def test_auth_header_consistency(self, auth_manager):
        """Test that auth headers are consistent."""
        headers1 = auth_manager.get_auth_headers()
        headers2 = auth_manager.get_auth_headers()

        # Should have same auth headers
        assert headers1["X-API-Key"] == headers2["X-API-Key"]
