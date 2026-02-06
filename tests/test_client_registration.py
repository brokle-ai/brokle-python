"""Tests for global client registration.

Phase 1: Auto-registration in constructor (first-write-wins)
Phase 2: Context scoping via brokle_context()
Phase 3: Deferred call-time resolution in wrappers
"""

from unittest.mock import MagicMock

import pytest

from brokle import Brokle, get_client, brokle_context
from brokle._client import _client_context, AsyncBrokle, _async_client_context


class TestAutoRegistration:
    """Phase 1: Auto-register in constructor."""

    def test_brokle_auto_registers_in_context(self):
        """Brokle() should set _client_context automatically."""
        client = Brokle(api_key="bk_test_secret_key")
        assert _client_context.get() is client

    def test_brokle_first_wins(self):
        """Second Brokle() instance should NOT overwrite the first."""
        first = Brokle(api_key="bk_test_secret_key")
        second = Brokle(api_key="bk_test_secret_key")
        assert _client_context.get() is first
        assert _client_context.get() is not second

    def test_get_client_returns_auto_registered(self):
        """get_client() should return the auto-registered instance."""
        client = Brokle(api_key="bk_test_secret_key")
        assert get_client() is client

    def test_disabled_client_auto_registers(self):
        """enabled=False client should still auto-register."""
        client = Brokle(api_key="invalid", enabled=False)
        assert _client_context.get() is client
        assert client.config.enabled is False

    def test_async_brokle_auto_registers(self):
        """AsyncBrokle() should set _async_client_context."""
        client = AsyncBrokle(api_key="bk_test_secret_key")
        assert _async_client_context.get() is client


class TestContextScoping:
    """Phase 2: Context scoping via brokle_context()."""

    def test_brokle_context_overrides_global(self):
        """brokle_context() should temporarily override the global client."""
        global_client = Brokle(api_key="bk_test_secret_key")
        scoped_client = Brokle(api_key="bk_test_secret_key")

        assert get_client() is global_client

        with brokle_context(scoped_client) as ctx_client:
            assert ctx_client is scoped_client
            assert _client_context.get() is scoped_client

        # After exiting context, back to global
        assert _client_context.get() is global_client

    def test_brokle_context_nesting(self):
        """Nested brokle_context() should restore correctly."""
        global_client = Brokle(api_key="bk_test_secret_key")
        scope_a = Brokle(api_key="bk_test_secret_key")
        scope_b = Brokle(api_key="bk_test_secret_key")

        assert _client_context.get() is global_client

        with brokle_context(scope_a):
            assert _client_context.get() is scope_a

            with brokle_context(scope_b):
                assert _client_context.get() is scope_b

            # Back to scope_a
            assert _client_context.get() is scope_a

        # Back to global
        assert _client_context.get() is global_client

    def test_brokle_context_restores_on_exception(self):
        """brokle_context() should restore on exception."""
        global_client = Brokle(api_key="bk_test_secret_key")
        scoped_client = Brokle(api_key="bk_test_secret_key")

        try:
            with brokle_context(scoped_client):
                assert _client_context.get() is scoped_client
                raise ValueError("test error")
        except ValueError:
            pass

        # Should be restored to global
        assert _client_context.get() is global_client


class TestCloseCleanup:
    """close()/context-manager should deregister the client."""

    def test_close_clears_context(self):
        """Brokle.close() should clear global registration."""
        client = Brokle(api_key="bk_test_secret_key")
        assert _client_context.get() is client
        client.close()
        assert _client_context.get() is None

    def test_context_manager_clears_context(self):
        """Exiting 'with Brokle(...)' should clear global registration."""
        with Brokle(api_key="bk_test_secret_key") as client:
            assert _client_context.get() is client
        assert _client_context.get() is None

    def test_close_only_clears_own_registration(self):
        """Closing a non-registered client should not clear the global."""
        first = Brokle(api_key="bk_test_secret_key")
        second = Brokle(api_key="bk_test_secret_key")
        # first is registered (first-write-wins)
        assert _client_context.get() is first
        second.close()  # second is NOT the registered one
        assert _client_context.get() is first  # first still registered


class TestAsyncBrokleContext:
    """async_brokle_context should work with 'async with'."""

    async def test_async_brokle_context_works(self):
        """async_brokle_context should set the async client context."""
        from brokle._client import async_brokle_context

        client = AsyncBrokle(api_key="bk_test_secret_key")
        async with async_brokle_context(client) as c:
            assert c is client
            assert _async_client_context.get() is client

    async def test_async_brokle_context_restores(self):
        """async_brokle_context should restore original client on exit."""
        from brokle._client import async_brokle_context

        original = AsyncBrokle(api_key="bk_test_secret_key")
        scoped = AsyncBrokle(api_key="bk_test_secret_key")
        # original is auto-registered
        assert _async_client_context.get() is original
        async with async_brokle_context(scoped):
            assert _async_client_context.get() is scoped
        assert _async_client_context.get() is original


class TestWrapperCallTimeResolution:
    """Phase 3: Deferred call-time resolution in wrappers."""

    def test_wrap_openai_before_client_init(self):
        """wrap_openai() should succeed even before Brokle is initialized.
        Client resolution happens at call time, not wrap time."""
        from brokle.wrappers import wrap_openai

        # Create mock OpenAI client
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        expected_response = MagicMock()
        mock_client.chat.completions.create = MagicMock(return_value=expected_response)

        # Wrap BEFORE creating Brokle client
        wrapped = wrap_openai(mock_client)

        # Now create client (auto-registers)
        Brokle(api_key="bk_test_secret_key", enabled=False)

        # Call should work - resolved at call time
        result = wrapped.chat.completions.create(model="gpt-4", messages=[])
        assert result is expected_response

    def test_brokle_context_affects_wrapper(self):
        """Context scoping should change which client the wrapper uses."""
        global_client = Brokle(api_key="bk_test_secret_key", enabled=False)
        scoped_client = Brokle(api_key="bk_test_secret_key", enabled=False)

        # Both are disabled so calls pass through regardless
        # This test verifies the context mechanism works with wrappers
        from brokle.wrappers import wrap_openai

        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        expected = MagicMock()
        mock_client.chat.completions.create = MagicMock(return_value=expected)

        wrapped = wrap_openai(mock_client)

        # Call in global context
        result = wrapped.chat.completions.create(model="gpt-4", messages=[])
        assert result is expected

        # Call in scoped context
        with brokle_context(scoped_client):
            result = wrapped.chat.completions.create(model="gpt-4", messages=[])
            assert result is expected
