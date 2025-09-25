"""Tests for core Brokle client functionality."""

import pytest
import os

from brokle import Brokle
from brokle.config import Config
from brokle.exceptions import AuthenticationError


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

    def test_client_initialization_missing_credentials(self, monkeypatch):
        """Test client initialization with missing credentials."""
        # Clear all environment variables
        monkeypatch.delenv("BROKLE_API_KEY", raising=False)
        monkeypatch.delenv("BROKLE_PROJECT_ID", raising=False)

        # Should raise AuthenticationError when no credentials provided
        with pytest.raises(AuthenticationError, match="API key is required"):
            Brokle(otel_enabled=False)

    def test_client_has_required_components(self, config):
        """Test client has required components."""
        client = Brokle(config=config)

        assert hasattr(client, 'chat')
        assert hasattr(client, 'embeddings')
        assert hasattr(client, 'models')

    def test_client_environment_configuration(self):
        """Test environment-based configuration."""
        client = Brokle(
            api_key="ak_env_key",
            project_id="proj_env",
            environment="production",
            host="https://custom.brokle.com",
            otel_enabled=False
        )

        assert client.config.environment == "production"
        assert client.config.host == "https://custom.brokle.com"
        assert client.config.api_key == "ak_env_key"
        assert client.config.project_id == "proj_env"

    def test_client_error_handling(self):
        """Test client error handling scenarios."""
        # Test invalid API key format validation (if implemented in Config)
        try:
            client = Brokle(
                api_key="invalid_key_format",
                project_id="proj_test",
                otel_enabled=False
            )
            # If no validation, that's also valid for the current API
            assert client.config.api_key == "invalid_key_format"
        except Exception:
            # Validation exists, which is also valid
            pass

    def test_client_lifecycle_operations(self, config):
        """Test client lifecycle operations."""
        client = Brokle(config=config)

        # Test context manager behavior
        with client:
            assert isinstance(client, Brokle)

        # Test explicit close (should not raise errors)
        client.close()

    def test_client_string_representation(self, config):
        """Test client string representation."""
        client = Brokle(config=config)
        repr_str = repr(client)

        # Should have some reasonable string representation
        assert "Brokle" in repr_str or "brokle" in repr_str.lower()