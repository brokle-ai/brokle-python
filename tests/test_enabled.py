"""Tests for BROKLE_ENABLED master switch functionality."""

import pytest
from unittest.mock import MagicMock

from brokle.config import BrokleConfig
from brokle import Brokle, observe
from brokle.wrappers import wrap_openai


class TestConfigEnabled:
    """Tests for BrokleConfig.enabled field."""

    def test_config_enabled_default_true(self):
        """Default enabled should be True."""
        config = BrokleConfig(api_key="bk_test_secret_key")
        assert config.enabled is True

    def test_config_disabled_skips_validation(self):
        """Disabled config should not validate API key."""
        config = BrokleConfig(api_key="invalid", enabled=False)
        assert config.enabled is False
        assert config.api_key == "invalid"

    def test_config_disabled_empty_api_key(self):
        """Disabled config should accept empty API key."""
        config = BrokleConfig(api_key="", enabled=False)
        assert config.enabled is False

    def test_config_repr_shows_enabled(self):
        """__repr__ should include enabled state."""
        config = BrokleConfig(api_key="bk_test_secret_key", enabled=False)
        repr_str = repr(config)
        assert "enabled=False" in repr_str

    def test_config_repr_shows_enabled_true(self):
        """__repr__ should show enabled=True when enabled."""
        config = BrokleConfig(api_key="bk_test_secret_key", enabled=True)
        repr_str = repr(config)
        assert "enabled=True" in repr_str


class TestConfigFromEnv:
    """Tests for BrokleConfig.from_env() with enabled."""

    def test_from_env_enabled_default_true(self, monkeypatch):
        """from_env() should default enabled to True."""
        monkeypatch.setenv("BROKLE_API_KEY", "bk_test_secret_key")
        monkeypatch.delenv("BROKLE_ENABLED", raising=False)
        config = BrokleConfig.from_env()
        assert config.enabled is True

    def test_from_env_disabled_no_api_key_required(self, monkeypatch):
        """from_env() with BROKLE_ENABLED=false should not require API key."""
        monkeypatch.setenv("BROKLE_ENABLED", "false")
        monkeypatch.delenv("BROKLE_API_KEY", raising=False)
        config = BrokleConfig.from_env()
        assert config.enabled is False
        assert config.api_key == "bk_disabled_placeholder"

    def test_from_env_enabled_requires_api_key(self, monkeypatch):
        """from_env() with enabled=True should require API key."""
        monkeypatch.delenv("BROKLE_API_KEY", raising=False)
        monkeypatch.delenv("BROKLE_ENABLED", raising=False)
        with pytest.raises(ValueError, match="BROKLE_API_KEY"):
            BrokleConfig.from_env()

    @pytest.mark.parametrize("value", ["false", "False", "FALSE", "0", "no", "off"])
    def test_from_env_disabled_various_false_values(self, monkeypatch, value):
        """from_env() should accept various false values."""
        monkeypatch.setenv("BROKLE_ENABLED", value)
        monkeypatch.delenv("BROKLE_API_KEY", raising=False)
        config = BrokleConfig.from_env()
        assert config.enabled is False

    @pytest.mark.parametrize("value", ["true", "True", "TRUE", "1", "yes", "on"])
    def test_from_env_enabled_various_true_values(self, monkeypatch, value):
        """from_env() should accept various true values."""
        monkeypatch.setenv("BROKLE_ENABLED", value)
        monkeypatch.setenv("BROKLE_API_KEY", "bk_test_secret_key")
        config = BrokleConfig.from_env()
        assert config.enabled is True


class TestClientEnabled:
    """Tests for Brokle client with enabled parameter."""

    def test_client_enabled_param(self):
        """Brokle() should accept enabled parameter directly."""
        client = Brokle(api_key="bk_test_secret_key", enabled=False)
        assert client.config.enabled is False

    def test_client_enabled_default_true(self):
        """Brokle() should default enabled to True."""
        client = Brokle(api_key="bk_test_secret_key")
        assert client.config.enabled is True

    def test_client_disabled_no_otel_resources(self):
        """Disabled client should not create OTEL resources."""
        client = Brokle(api_key="bk_test_secret_key", enabled=False)
        assert client._provider is None
        assert client._processor is None
        assert client._meter_provider is None

    def test_client_disabled_invalid_api_key_ok(self):
        """Disabled client should accept invalid API key."""
        client = Brokle(api_key="invalid", enabled=False)
        assert client.config.enabled is False


class TestDecoratorEnabled:
    """Tests for @observe decorator with disabled SDK."""

    def test_observe_passthrough_when_disabled(self, monkeypatch):
        """@observe should pass through when SDK disabled."""
        monkeypatch.setenv("BROKLE_ENABLED", "false")

        import brokle._client
        brokle._client._global_client = None

        @observe()
        def my_func(x):
            return x * 2

        result = my_func(5)
        assert result == 10

    def test_observe_passthrough_preserves_return_value(self, monkeypatch):
        """@observe should preserve return values when disabled."""
        monkeypatch.setenv("BROKLE_ENABLED", "false")

        import brokle._client
        brokle._client._global_client = None

        @observe()
        def get_dict():
            return {"key": "value", "count": 42}

        result = get_dict()
        assert result == {"key": "value", "count": 42}


class TestWrapperEnabled:
    """Tests for wrappers with disabled SDK."""

    def test_wrap_openai_returns_unwrapped_when_disabled(self, monkeypatch):
        """wrap_openai should return unwrapped client when disabled."""
        monkeypatch.setenv("BROKLE_ENABLED", "false")

        import brokle._client
        brokle._client._global_client = None

        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = MagicMock()

        original_create = mock_client.chat.completions.create

        wrapped = wrap_openai(mock_client)

        assert wrapped.chat.completions.create is original_create
