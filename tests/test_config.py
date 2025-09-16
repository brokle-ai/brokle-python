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
            'BROKLE_PUBLIC_KEY': 'pk_test_key',
            'BROKLE_HOST': 'https://test.example.com',
            'BROKLE_SECRET_KEY': 'proj_test',
            'BROKLE_ENVIRONMENT': 'test',
        }):
            config = Config.from_env()
            assert config.public_key == 'pk_test_key'
            assert config.host == 'https://test.example.com'
            assert config.secret_key == 'proj_test'
            assert config.environment == 'test'
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = Config(
            public_key='pk_test_key',
            secret_key='proj_test'
        )
        
        # Should not raise
        config.validate()
        
        # Missing public key should raise
        config.public_key = None
        with pytest.raises(ValueError, match="Public key is required"):
            config.validate()

        # Missing secret key should raise
        config.public_key = 'pk_test_key'
        config.secret_key = None
        with pytest.raises(ValueError, match="Secret key is required"):
            config.validate()
    
    def test_public_key_validation(self):
        """Test public key format validation."""
        # Valid public key
        config = Config(public_key='pk_test_key')
        assert config.public_key == 'pk_test_key'

        # Invalid public key format
        with pytest.raises(ValueError, match='Public key must start with "pk_"'):
            Config(public_key='invalid_key')
    
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
            public_key='pk_test_key',
            secret_key='proj_test',
            environment='test'
        )
        
        headers = config.get_headers()
        assert headers['X-Public-Key'] == 'pk_test_key'
        assert headers['X-Secret-Key'] == 'proj_test'
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
            public_key='pk_global_test',
            host='https://global.example.com',
            secret_key='proj_global'
        )
        
        config = get_config()
        assert config.public_key == 'pk_global_test'
        assert config.host == 'https://global.example.com'
        assert config.secret_key == 'proj_global'
    
    def test_configure_updates_existing(self):
        """Test that configure updates existing configuration."""
        configure(public_key='pk_initial')
        
        config1 = get_config()
        assert config1.public_key == 'pk_initial'
        
        configure(public_key='pk_updated')
        
        config2 = get_config()
        assert config2.public_key == 'pk_updated'
        assert config1 is config2  # Same instance
    
    def test_reset_config(self):
        """Test resetting configuration."""
        configure(public_key='pk_test')
        config1 = get_config()
        
        reset_config()
        config2 = get_config()
        
        assert config1 is not config2  # Different instances
        assert config2.public_key is None  # Reset to default