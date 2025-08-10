"""Tests for configuration module."""

import os
import pytest
from unittest.mock import patch

from brokle.config import Config, configure, get_config, reset_config


class TestConfig:
    """Test configuration functionality."""
    
    def test_config_from_env(self):
        """Test configuration from environment variables."""
        with patch.dict(os.environ, {
            'BROKLE_API_KEY': 'ak_test_key',
            'BROKLE_HOST': 'https://test.example.com',
            'BROKLE_PROJECT_ID': 'proj_test',
            'BROKLE_ENVIRONMENT': 'test',
        }):
            config = Config.from_env()
            assert config.api_key == 'ak_test_key'
            assert config.host == 'https://test.example.com'
            assert config.project_id == 'proj_test'
            assert config.environment == 'test'
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = Config(
            api_key='ak_test_key',
            project_id='proj_test'
        )
        
        # Should not raise
        config.validate()
        
        # Missing API key should raise
        config.api_key = None
        with pytest.raises(ValueError, match="API key is required"):
            config.validate()
        
        # Missing project ID should raise
        config.api_key = 'ak_test_key'
        config.project_id = None
        with pytest.raises(ValueError, match="Project ID is required"):
            config.validate()
    
    def test_api_key_validation(self):
        """Test API key format validation."""
        # Valid API key
        config = Config(api_key='ak_test_key')
        assert config.api_key == 'ak_test_key'
        
        # Invalid API key format
        with pytest.raises(ValueError, match='API key must start with "ak_"'):
            Config(api_key='invalid_key')
    
    def test_host_validation(self):
        """Test host URL validation."""
        # Valid hosts
        config = Config(host='http://localhost:8000')
        assert config.host == 'http://localhost:8000'
        
        config = Config(host='https://example.com/')
        assert config.host == 'https://example.com'  # Trailing slash removed
        
        # Invalid host format
        with pytest.raises(ValueError, match='Host must start with http:// or https://'):
            Config(host='invalid_host')
    
    def test_get_headers(self):
        """Test getting HTTP headers."""
        config = Config(
            api_key='ak_test_key',
            project_id='proj_test',
            environment='test'
        )
        
        headers = config.get_headers()
        assert headers['X-API-Key'] == 'ak_test_key'
        assert headers['X-Project-ID'] == 'proj_test'
        assert headers['X-Environment'] == 'test'
        assert headers['Content-Type'] == 'application/json'
        assert 'brokle-python' in headers['User-Agent']


class TestGlobalConfig:
    """Test global configuration management."""
    
    def setup_method(self):
        """Reset config before each test."""
        reset_config()
    
    def teardown_method(self):
        """Clean up after each test."""
        reset_config()
    
    def test_configure_and_get_config(self):
        """Test global configuration."""
        configure(
            api_key='ak_global_test',
            host='https://global.example.com',
            project_id='proj_global'
        )
        
        config = get_config()
        assert config.api_key == 'ak_global_test'
        assert config.host == 'https://global.example.com'
        assert config.project_id == 'proj_global'
    
    def test_configure_updates_existing(self):
        """Test that configure updates existing configuration."""
        configure(api_key='ak_initial')
        
        config1 = get_config()
        assert config1.api_key == 'ak_initial'
        
        configure(api_key='ak_updated')
        
        config2 = get_config()
        assert config2.api_key == 'ak_updated'
        assert config1 is config2  # Same instance
    
    def test_reset_config(self):
        """Test resetting configuration."""
        configure(api_key='ak_test')
        config1 = get_config()
        
        reset_config()
        config2 = get_config()
        
        assert config1 is not config2  # Different instances
        assert config2.api_key is None  # Reset to default