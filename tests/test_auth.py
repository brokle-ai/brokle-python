"""Tests for authentication module."""

import pytest
from unittest.mock import patch, MagicMock
import time

from brokle.auth import AuthManager
from brokle.config import Config
from brokle.exceptions import AuthenticationError


class TestAuthManager:
    """Test AuthManager functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config(
            api_key="ak_test_key",
            project_id="proj_test",
            host="https://test.example.com"
        )
    
    @pytest.fixture
    def auth_manager(self, config):
        """Create test auth manager."""
        return AuthManager(config)
    
    def test_init_with_config(self, config):
        """Test AuthManager initialization."""
        auth_manager = AuthManager(config)
        assert auth_manager.config == config
        assert auth_manager.api_key == "ak_test_key"
    
    def test_validate_api_key_valid(self, auth_manager):
        """Test API key validation with valid key."""
        # Should not raise
        auth_manager.validate_api_key()
    
    def test_validate_api_key_invalid_format(self, config):
        """Test API key validation with invalid format."""
        config.api_key = "invalid_key"
        auth_manager = AuthManager(config)
        
        with pytest.raises(AuthenticationError, match="Invalid API key format"):
            auth_manager.validate_api_key()
    
    def test_validate_api_key_missing(self, config):
        """Test API key validation with missing key."""
        config.api_key = None
        auth_manager = AuthManager(config)
        
        with pytest.raises(AuthenticationError, match="API key is required"):
            auth_manager.validate_api_key()
    
    def test_validate_api_key_empty(self, config):
        """Test API key validation with empty key."""
        config.api_key = ""
        auth_manager = AuthManager(config)
        
        with pytest.raises(AuthenticationError, match="API key is required"):
            auth_manager.validate_api_key()
    
    def test_get_auth_headers(self, auth_manager):
        """Test getting authentication headers."""
        headers = auth_manager.get_auth_headers()
        
        assert headers["X-API-Key"] == "ak_test_key"
        assert headers["X-Project-ID"] == "proj_test"
        assert "User-Agent" in headers
        assert "brokle-python" in headers["User-Agent"]
    
    def test_get_auth_headers_with_environment(self, config):
        """Test getting authentication headers with environment."""
        config.environment = "default"
        auth_manager = AuthManager(config)
        
        headers = auth_manager.get_auth_headers()
        
        assert headers["X-Environment"] == "default"
    
    def test_get_auth_headers_without_environment(self, auth_manager):
        """Test getting authentication headers without environment."""
        headers = auth_manager.get_auth_headers()
        
        # Should not include X-Environment header if not set
        assert "X-Environment" not in headers
    
    def test_is_authenticated_true(self, auth_manager):
        """Test authentication status when authenticated."""
        assert auth_manager.is_authenticated() is True
    
    def test_is_authenticated_false_no_key(self, config):
        """Test authentication status without API key."""
        config.api_key = None
        auth_manager = AuthManager(config)
        
        assert auth_manager.is_authenticated() is False
    
    def test_is_authenticated_false_invalid_key(self, config):
        """Test authentication status with invalid API key."""
        config.api_key = "invalid_key"
        auth_manager = AuthManager(config)
        
        assert auth_manager.is_authenticated() is False
    
    def test_refresh_token_placeholder(self, auth_manager):
        """Test refresh token method (placeholder)."""
        # This is a placeholder method for future token refresh functionality
        result = auth_manager.refresh_token()
        assert result is None
    
    def test_get_bearer_token_format(self, auth_manager):
        """Test getting bearer token format."""
        # If bearer token format is needed in the future
        token = auth_manager.get_bearer_token()
        assert token is None  # Currently returns None as it's not implemented
    
    def test_auth_manager_with_different_api_key_formats(self):
        """Test AuthManager with different API key formats."""
        # Test with Brokle format
        config1 = Config(api_key="ak_test_key", project_id="proj_test")
        auth1 = AuthManager(config1)
        assert auth1.is_authenticated() is True
        
        # Test with OpenAI format (should also work)
        config2 = Config(api_key="sk-openai-key", project_id="proj_test")
        auth2 = AuthManager(config2)
        assert auth2.is_authenticated() is True
        
        # Test with invalid format
        config3 = Config(api_key="invalid_format", project_id="proj_test")
        auth3 = AuthManager(config3)
        assert auth3.is_authenticated() is False
    
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
        assert "ak_test_key" not in str_repr  # Should not expose sensitive info
        assert "proj_test" in str_repr  # Project ID is not sensitive
    
    def test_auth_manager_repr(self, auth_manager):
        """Test repr representation of AuthManager."""
        repr_str = repr(auth_manager)
        assert "AuthManager" in repr_str
        assert "project_id" in repr_str
        assert "ak_test_key" not in repr_str  # Should not expose API key
    
    @patch('brokle.auth.time.time')
    def test_auth_headers_caching(self, mock_time, auth_manager):
        """Test that auth headers are cached appropriately."""
        # Mock time to control caching behavior
        mock_time.return_value = 1000
        
        # First call should create headers
        headers1 = auth_manager.get_auth_headers()
        
        # Second call should return same headers (if cached)
        headers2 = auth_manager.get_auth_headers()
        
        # Headers should be the same
        assert headers1 == headers2
    
    def test_auth_with_additional_headers(self, auth_manager):
        """Test authentication with additional custom headers."""
        additional_headers = {
            "X-Custom-Header": "custom_value",
            "X-Request-ID": "req_123"
        }
        
        headers = auth_manager.get_auth_headers(additional_headers)
        
        # Should include both auth headers and additional headers
        assert headers["X-API-Key"] == "ak_test_key"
        assert headers["X-Project-ID"] == "proj_test"
        assert headers["X-Custom-Header"] == "custom_value"
        assert headers["X-Request-ID"] == "req_123"
    
    def test_auth_header_override_protection(self, auth_manager):
        """Test that auth headers cannot be overridden."""
        malicious_headers = {
            "X-API-Key": "malicious_key",
            "X-Project-ID": "malicious_project"
        }
        
        headers = auth_manager.get_auth_headers(malicious_headers)
        
        # Should preserve original auth headers
        assert headers["X-API-Key"] == "ak_test_key"
        assert headers["X-Project-ID"] == "proj_test"