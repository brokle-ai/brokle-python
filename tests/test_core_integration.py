"""Core integration tests for the Brokle SDK."""

import pytest
from unittest.mock import patch

from brokle import (
    Brokle, get_client, Config, configure, get_config, reset_config,
    AuthManager, BrokleError, AuthenticationError, ConfigurationError
)
from pydantic import ValidationError


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

    def test_global_config_integration(self):
        """Test global configuration management."""
        # Reset first
        reset_config()

        # Configure globally
        configure(
            api_key="ak_global_key",
            project_id="proj_global"
        )

        config = get_config()
        assert config.api_key == "ak_global_key"
        assert config.project_id == "proj_global"

        # Cleanup
        reset_config()


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

    def teardown_method(self):
        """Reset global client after each test."""
        import brokle._client.client as client_module
        client_module._client = None

    def test_get_client_with_kwargs(self):
        """Test get_client with keyword arguments."""
        client = get_client(
            api_key="ak_test_key",
            project_id="proj_test",
            otel_enabled=False
        )

        assert isinstance(client, Brokle)
        assert client.config.api_key == "ak_test_key"
        assert client.config.project_id == "proj_test"

    def test_get_client_singleton(self):
        """Test get_client returns singleton instance."""
        client1 = get_client(
            api_key="ak_test_key",
            project_id="proj_test",
            otel_enabled=False
        )
        client2 = get_client()

        # Should be the same instance
        assert client1 is client2

    def test_get_client_with_config(self):
        """Test get_client with config object."""
        config = Config(
            api_key="ak_test_key",
            project_id="proj_test",
            otel_enabled=False
        )

        client = get_client(config=config)
        assert isinstance(client, Brokle)
        assert client.config.api_key == "ak_test_key"

    def test_get_client_integration_with_global_config(self):
        """Test get_client works with global configuration."""
        # Reset global config
        reset_config()

        # Set global config
        configure(
            api_key="ak_global_key",
            project_id="proj_global",
            otel_enabled=False
        )

        # get_client should use global config when no params provided
        client = get_client()
        assert client.config.api_key == "ak_global_key"
        assert client.config.project_id == "proj_global"

        # Cleanup
        reset_config()

    def test_client_preserves_global_config_with_overrides(self):
        """Test that client preserves global config when only some params are overridden."""
        # Reset global config
        reset_config()

        # Set global config with custom host and otel settings
        configure(
            api_key="ak_global_key",
            project_id="proj_global",
            host="https://custom.api.com",
            environment="production",
            otel_enabled=False
        )

        # Create client with only api_key override - should preserve other settings
        client = Brokle(api_key="ak_override_key")

        # Should have the overridden api_key
        assert client.config.api_key == "ak_override_key"

        # Should preserve all other global config settings
        assert client.config.project_id == "proj_global"
        assert client.config.host == "https://custom.api.com"
        assert client.config.environment == "production"
        assert client.config.otel_enabled == False

        # Cleanup
        reset_config()

    def test_client_kwargs_override_global_config(self):
        """Test that kwargs properly override global config settings."""
        # Reset global config
        reset_config()

        # Set global config
        configure(
            api_key="ak_global_key",
            project_id="proj_global",
            host="https://global.api.com",
            otel_enabled=True
        )

        # Create client with kwargs overrides
        client = Brokle(
            api_key="ak_new_key",
            host="https://override.api.com",
            otel_enabled=False
        )

        # Should have overridden values
        assert client.config.api_key == "ak_new_key"
        assert client.config.host == "https://override.api.com"
        assert client.config.otel_enabled == False

        # Should preserve non-overridden values
        assert client.config.project_id == "proj_global"

        # Cleanup
        reset_config()

    def test_client_does_not_mutate_global_config(self):
        """Test that client creation with overrides doesn't mutate the global config."""
        # Reset global config
        reset_config()

        # Set global config
        configure(
            api_key="ak_global_key",
            project_id="proj_global",
            host="https://global.api.com",
            environment="production",
            otel_enabled=True
        )

        # Get reference to global config state before client creation
        global_config_before = get_config()
        original_api_key = global_config_before.api_key
        original_host = global_config_before.host
        original_otel = global_config_before.otel_enabled

        # Create client with overrides
        client = Brokle(
            api_key="ak_override_key",
            host="https://override.api.com",
            otel_enabled=False
        )

        # Verify client has overridden values
        assert client.config.api_key == "ak_override_key"
        assert client.config.host == "https://override.api.com"
        assert client.config.otel_enabled == False

        # CRITICAL: Verify global config was NOT mutated
        global_config_after = get_config()
        assert global_config_after.api_key == original_api_key  # Should be "ak_global_key"
        assert global_config_after.host == original_host  # Should be "https://global.api.com"
        assert global_config_after.otel_enabled == original_otel  # Should be True

        # Verify it's the same singleton object
        assert global_config_before is global_config_after

        # Create another client to ensure it gets the original global config
        client2 = Brokle()
        assert client2.config.api_key == "ak_global_key"
        assert client2.config.host == "https://global.api.com"
        assert client2.config.otel_enabled == True

        # Cleanup
        reset_config()


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

    def test_global_config_workflow(self):
        """Test workflow using global configuration."""
        # Reset
        reset_config()

        # 1. Configure globally
        configure(
            api_key="ak_global_key",
            project_id="proj_global",
            environment="staging",
            otel_enabled=False
        )

        # 2. Get client using global config
        client = get_client()

        # 3. Verify configuration
        assert client.config.api_key == "ak_global_key"
        assert client.config.project_id == "proj_global"
        assert client.config.environment == "staging"

        # 4. Verify auth headers
        headers = client.auth_manager.get_auth_headers()
        assert headers["X-Project-ID"] == "proj_global"
        assert headers["X-Environment"] == "staging"

        # Cleanup
        reset_config()