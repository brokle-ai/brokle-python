"""Core integration tests for the Brokle SDK."""

import os
import pytest
from unittest.mock import patch

# Note: Client management now handled by context manager

from brokle import (
    Brokle, get_client, Config,
    AuthManager, BrokleError, AuthenticationError, ConfigurationError
)
from pydantic import ValidationError


@pytest.fixture(autouse=True)
def clear_client_instances():
    """Ensure context manager cache is reset across tests."""
    from brokle._client.context import _context_manager
    _context_manager._active_contexts.clear()
    yield
    _context_manager._active_contexts.clear()


class TestSDKImports:
    """Test that all core SDK components can be imported."""

    def test_core_imports(self):
        """Test core imports work."""
        # These should not raise
        assert Brokle is not None
        assert get_client is not None
        assert Config is not None
        assert AuthManager is not None

    def test_exception_imports(self):
        """Test exception imports work."""
        assert BrokleError is not None
        assert AuthenticationError is not None
        assert ConfigurationError is not None


class TestConfigIntegration:
    """Test configuration integration with other components."""

    def test_config_creation_minimal(self):
        """Test config can be created with minimal parameters."""
        config = Config(
            api_key="ak_test_key",
            project_id="proj_test"
        )
        assert config.api_key == "ak_test_key"
        assert config.project_id == "proj_test"
        assert config.environment == "default"

    def test_config_with_environment(self):
        """Test config with custom environment."""
        config = Config(
            api_key="ak_test_key",
            project_id="proj_test",
            environment="staging"
        )
        assert config.environment == "staging"

    def test_config_from_env_defaults(self):
        """Config.from_env should pick up environment variables."""
        with patch.dict(os.environ, {
            "BROKLE_API_KEY": "ak_env_key",
            "BROKLE_PROJECT_ID": "proj_env",
            "BROKLE_HOST": "https://env.example.com",
        }, clear=True):
            config = Config.from_env()
        assert config.api_key == "ak_env_key"
        assert config.project_id == "proj_env"
        assert config.host == "https://env.example.com"


class TestAuthManagerIntegration:
    """Test authentication manager integration."""

    def test_auth_manager_creation(self):
        """Test auth manager can be created."""
        config = Config(
            api_key="ak_test_key",
            project_id="proj_test"
        )
        auth = AuthManager(config)
        assert auth is not None
        assert str(auth) == "AuthManager(project_id=proj_test)"

    def test_auth_headers_generation(self):
        """Test auth headers generation."""
        config = Config(
            api_key="ak_test_key",
            project_id="proj_test",
            environment="staging"
        )
        auth = AuthManager(config)
        headers = auth.get_auth_headers()

        assert "X-API-Key" in headers
        assert "X-Project-ID" in headers
        assert "X-Environment" in headers
        assert headers["X-Project-ID"] == "proj_test"
        assert headers["X-Environment"] == "staging"

    def test_auth_manager_with_default_environment(self):
        """Test auth manager with default environment."""
        config = Config(
            api_key="ak_test_key",
            project_id="proj_test"
        )
        auth = AuthManager(config)
        headers = auth.get_auth_headers()

        assert headers["X-Environment"] == "default"


class TestClientIntegration:
    """Test Brokle client integration."""

    def test_client_creation_with_config(self):
        """Test client creation with config object."""
        config = Config(
            api_key="ak_test_key",
            project_id="proj_test"
        )
        # Disable OTEL for testing
        config.otel_enabled = False

        client = Brokle(config=config)
        assert client.config.api_key == "ak_test_key"
        assert client.config.project_id == "proj_test"

    def test_client_creation_with_kwargs(self):
        """Test client creation with keyword arguments."""
        client = Brokle(
            api_key="ak_test_key",
            project_id="proj_test",
            host="https://test.example.com",
            otel_enabled=False
        )
        assert client.config.api_key == "ak_test_key"
        assert client.config.project_id == "proj_test"
        assert client.config.host == "https://test.example.com"

    def test_client_has_auth_manager(self):
        """Test client has auth manager."""
        config = Config(
            api_key="ak_test_key",
            project_id="proj_test",
            otel_enabled=False
        )
        client = Brokle(config=config)
        assert hasattr(client, 'auth_manager')
        assert isinstance(client.auth_manager, AuthManager)

    def test_client_auth_integration(self):
        """Test client integrates properly with auth manager."""
        config = Config(
            api_key="ak_test_key",
            project_id="proj_test",
            environment="staging",
            otel_enabled=False
        )
        client = Brokle(config=config)

        headers = client.auth_manager.get_auth_headers()
        assert headers["X-Project-ID"] == "proj_test"
        assert headers["X-Environment"] == "staging"

    def test_client_span_creation(self):
        """Test client can create spans."""
        config = Config(
            api_key="ak_test_key",
            project_id="proj_test",
            otel_enabled=False
        )
        client = Brokle(config=config)

        # This should not raise even with OTEL disabled
        span = client.span("test-span")
        assert span is not None
        assert hasattr(span, 'name')

    def test_client_generation_creation(self):
        """Test client can create generation spans."""
        config = Config(
            api_key="ak_test_key",
            project_id="proj_test",
            otel_enabled=False
        )
        client = Brokle(config=config)

        # This should not raise even with OTEL disabled
        generation = client.generation("test-generation", model="gpt-4")
        assert generation is not None
        assert hasattr(generation, 'name')

    @pytest.mark.asyncio
    async def test_client_shutdown(self):
        """Test client shutdown doesn't raise errors."""
        config = Config(
            api_key="ak_test_key",
            project_id="proj_test",
            otel_enabled=False
        )
        client = Brokle(config=config)

        # Should not raise
        await client.shutdown()

    def test_client_flush(self):
        """Test client flush doesn't raise errors."""
        config = Config(
            api_key="ak_test_key",
            project_id="proj_test",
            otel_enabled=False
        )
        client = Brokle(config=config)

        # Should not raise
        client.flush()


class TestGetClientIntegration:
    """Test get_client function integration."""

    def test_get_client_with_explicit_config(self):
        """Explicit kwargs should build a client without touching environment."""
        client = get_client(
            api_key="ak_test_key",
            project_id="proj_test",
            otel_enabled=False
        )

        assert isinstance(client, Brokle)
        assert client.config.api_key == "ak_test_key"
        assert client.config.project_id == "proj_test"

    def test_get_client_env_fallback(self):
        """Environment variables should seed the default singleton."""
        with patch.dict(os.environ, {
            "BROKLE_API_KEY": "ak_env",
            "BROKLE_PROJECT_ID": "proj_env",
            "BROKLE_HOST": "https://env.example.com",
        }, clear=True):
            client = get_client(otel_enabled=False)

        assert client.config.api_key == "ak_env"
        assert client.config.project_id == "proj_env"
        assert client.config.host == "https://env.example.com"

    def test_get_client_singleton_default(self):
        """Calling get_client() repeatedly returns the same default instance."""
        with patch.dict(os.environ, {
            "BROKLE_API_KEY": "ak_singleton",
            "BROKLE_PROJECT_ID": "proj_singleton",
        }, clear=True):
            first = get_client(otel_enabled=False)
            second = get_client()

        assert first is second

    def test_get_client_multi_project(self):
        """Different api keys should result in distinct cached clients."""
        client_a = get_client(api_key="ak_project_a", project_id="proj_a", otel_enabled=False)
        client_b = get_client(api_key="ak_project_b", project_id="proj_b", otel_enabled=False)

        assert client_a is not client_b
        assert client_a.config.project_id == "proj_a"
        assert client_b.config.project_id == "proj_b"

    def test_get_client_without_key_when_multiple_exist(self):
        """No api_key with multiple cached clients should return disabled instance."""
        get_client(api_key="ak_project_a", project_id="proj_a", otel_enabled=True)
        get_client(api_key="ak_project_b", project_id="proj_b", otel_enabled=True)

        with patch.dict(os.environ, {}, clear=True):
            default = get_client()

        assert default.config.otel_enabled is False
        assert default.config.api_key == "ak_fake"


class TestExceptionIntegration:
    """Test exception integration with other components."""

    def test_config_validation_exceptions(self):
        """Test config validation raises proper exceptions."""
        # Invalid environment should raise ValidationError (from Pydantic)
        with pytest.raises(ValidationError, match="Environment name cannot start with 'brokle' prefix"):
            Config(
                api_key="ak_test_key",
                project_id="proj_test",
                environment="brokle-test"
            )

    def test_exception_hierarchy(self):
        """Test exception hierarchy works correctly."""
        error = AuthenticationError("Invalid API key")

        # Should be instance of both specific and base exception
        assert isinstance(error, AuthenticationError)
        assert isinstance(error, BrokleError)

        # Should have proper attributes
        assert str(error) == "Invalid API key"
        assert error.status_code == 401
        assert error.error_code == "authentication_error"

    def test_exception_with_details(self):
        """Test exception with additional details."""
        details = {"provider": "openai", "model": "gpt-4"}
        error = BrokleError("Provider error", status_code=500, details=details)

        assert error.details == details
        assert error.status_code == 500


class TestEndToEndWorkflow:
    """Test end-to-end workflow scenarios."""

    def test_complete_client_workflow(self):
        """Test complete client initialization and usage workflow."""
        # 1. Create config
        config = Config(
            api_key="ak_test_key",
            project_id="proj_test",
            environment="production",
            otel_enabled=False
        )

        # 2. Create client
        client = Brokle(config=config)

        # 3. Verify auth integration
        headers = client.auth_manager.get_auth_headers()
        assert headers["X-Project-ID"] == "proj_test"
        assert headers["X-Environment"] == "production"

        # 4. Create spans
        span = client.span("workflow-test")
        generation = client.generation("llm-call", model="gpt-4")

        # 5. Verify objects created
        assert span is not None
        assert generation is not None

        # 6. Cleanup should not raise
        client.flush()

    @pytest.mark.asyncio
    async def test_async_workflow(self):
        """Test async workflow doesn't cause issues."""
        config = Config(
            api_key="ak_test_key",
            project_id="proj_test",
            otel_enabled=False
        )

        client = Brokle(config=config)

        # Should handle async operations gracefully
        await client.shutdown()

    def test_env_based_workflow(self):
        """End-to-end workflow backed by environment variables."""
        with patch.dict(os.environ, {
            "BROKLE_API_KEY": "ak_global_key",
            "BROKLE_PROJECT_ID": "proj_global",
            "BROKLE_ENVIRONMENT": "staging",
        }, clear=True):
            client = get_client(otel_enabled=False)

        assert client.config.api_key == "ak_global_key"
        assert client.config.project_id == "proj_global"
        assert client.config.environment == "staging"

        headers = client.auth_manager.get_auth_headers()
        assert headers["X-Project-ID"] == "proj_global"
        assert headers["X-Environment"] == "staging"
