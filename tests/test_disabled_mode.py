"""
Test disabled mode functionality.

Tests that the SDK handles missing API keys gracefully,
logging warnings and operating in disabled mode.
"""

import logging

import pytest

from brokle import AsyncBrokle, Brokle


class TestDisabledMode:
    """Test disabled mode functionality."""

    def test_sync_client_disabled_mode(self, caplog):
        """Test sync client operates in disabled mode without API key."""
        with caplog.at_level(logging.WARNING, logger="brokle"):
            client = Brokle(api_key=None)

        # Check warning was logged
        assert "Authentication error" in caplog.text
        assert "api_key" in caplog.text
        assert "Client will be disabled" in caplog.text

        # Check client is disabled
        assert client.is_disabled

    def test_async_client_disabled_mode(self, caplog):
        """Test async client operates in disabled mode without API key."""
        with caplog.at_level(logging.WARNING, logger="brokle"):
            client = AsyncBrokle(api_key=None)

        # Check warning was logged
        assert "Authentication error" in caplog.text

        # Check client is disabled
        assert client.is_disabled

    def test_disabled_client_chat_returns_none(self):
        """Test disabled client chat methods return None."""
        client = Brokle(api_key=None)

        # Chat completions should return None
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}]
        )
        assert response is None

    def test_disabled_client_models_returns_none(self):
        """Test disabled client models methods return None."""
        client = Brokle(api_key=None)

        # Models list should return None
        models = client.models.list()
        assert models is None

    def test_disabled_client_embeddings_returns_none(self):
        """Test disabled client embeddings methods return None."""
        client = Brokle(api_key=None)

        # Embeddings create should return None
        embeddings = client.embeddings.create(
            input="test text",
            model="text-embedding-ada-002"
        )
        assert embeddings is None

    @pytest.mark.asyncio
    async def test_disabled_async_client_methods_return_none(self):
        """Test disabled async client methods return None."""
        client = AsyncBrokle(api_key=None)

        # All async methods should return None
        chat_response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}]
        )
        assert chat_response is None

        models = await client.models.list()
        assert models is None

        embeddings = await client.embeddings.create(
            input="test",
            model="text-embedding-ada-002"
        )
        assert embeddings is None

    def test_enabled_client_not_disabled(self):
        """Test client with API key is not disabled."""
        client = Brokle(api_key="bk_test_key")
        assert not client.is_disabled

    def test_warning_message_format(self, caplog):
        """Test warning message format."""
        with caplog.at_level(logging.WARNING, logger="brokle"):
            Brokle(api_key=None)

        # Should match expected pattern
        warning_text = caplog.text
        assert "Authentication error:" in warning_text
        assert "initialized without api_key" in warning_text
        assert "Client will be disabled" in warning_text
        assert "Provide an api_key parameter or set BROKLE_API_KEY environment variable" in warning_text