"""
Test suite for wrapper functions - Pattern 1 implementation.

Tests the explicit wrapper functions (wrap_openai, wrap_anthropic).
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from brokle import wrap_anthropic, wrap_openai
from brokle.exceptions import ProviderError


class TestWrapOpenAI:
    """Test wrap_openai() function."""

    def test_wrap_openai_import_available(self):
        """Test that wrap_openai is available in main imports."""
        from brokle import wrap_openai

        assert callable(wrap_openai)

    @patch("brokle.wrappers.openai.HAS_OPENAI", False)
    def test_wrap_openai_no_sdk_installed(self):
        """Test error when OpenAI SDK not installed."""
        mock_client = Mock()
        with pytest.raises(ProviderError) as exc_info:
            wrap_openai(mock_client)
        assert "OpenAI SDK not installed" in str(exc_info.value)

    @patch("brokle.wrappers.openai.HAS_OPENAI", True)
    def test_wrap_openai_invalid_client_type(self):
        """Test error when invalid client type passed."""

        # Create proper mock classes for isinstance() checks
        class MockOpenAI:
            pass

        class MockAsyncOpenAI:
            pass

        with patch("brokle.wrappers.openai._OpenAI", MockOpenAI):
            with patch("brokle.wrappers.openai._AsyncOpenAI", MockAsyncOpenAI):
                mock_client = "not_a_client"
                with pytest.raises(ProviderError) as exc_info:
                    wrap_openai(mock_client)
                assert "expects OpenAI" in str(exc_info.value)

    @patch("brokle.wrappers.openai.HAS_OPENAI", True)
    def test_wrap_openai_basic_success(self):
        """Test successful wrapping of OpenAI client."""

        # Create mock OpenAI classes
        class MockOpenAI:
            def __init__(self):
                self.chat = Mock()
                self.chat.completions = Mock()
                self.chat.completions.create = Mock()
                self.completions = Mock()
                self.completions.create = Mock()
                self.embeddings = Mock()
                self.embeddings.create = Mock()

        class MockAsyncOpenAI:
            pass

        # Create client instance
        mock_client = MockOpenAI()

        with patch("brokle.wrappers.openai._OpenAI", MockOpenAI):
            with patch("brokle.wrappers.openai._AsyncOpenAI", MockAsyncOpenAI):
                # Execute
                result = wrap_openai(mock_client)

                # Verify wrapping markers are set
                assert result is mock_client
                assert hasattr(result, "_brokle_instrumented")
                assert result._brokle_instrumented is True
                assert hasattr(result, "_brokle_wrapper_version")

    @patch("brokle.wrappers.openai.HAS_OPENAI", True)
    def test_wrap_openai_already_wrapped(self):
        """Test that wrapping an already-wrapped client is safe."""

        class MockOpenAI:
            def __init__(self):
                self.chat = Mock()
                self.chat.completions = Mock()
                self._brokle_instrumented = False

        class MockAsyncOpenAI:
            pass

        mock_client = MockOpenAI()

        with patch("brokle.wrappers.openai._OpenAI", MockOpenAI):
            with patch("brokle.wrappers.openai._AsyncOpenAI", MockAsyncOpenAI):
                # First wrap
                result1 = wrap_openai(mock_client)
                assert result1._brokle_instrumented is True

                # Second wrap should return safely
                result2 = wrap_openai(mock_client)
                assert result2 is mock_client
                assert result2._brokle_instrumented is True

    @patch("brokle.wrappers.openai.HAS_OPENAI", True)
    def test_wrap_openai_instruments_methods(self):
        """Test that wrap_openai instruments the expected methods."""

        class MockOpenAI:
            def __init__(self):
                self.chat = Mock()
                self.chat.completions = Mock()
                self.chat.completions.create = Mock()
                self.completions = Mock()
                self.completions.create = Mock()
                self.embeddings = Mock()
                self.embeddings.create = Mock()

        class MockAsyncOpenAI:
            pass

        mock_client = MockOpenAI()
        original_chat_create = mock_client.chat.completions.create
        original_completions_create = mock_client.completions.create
        original_embeddings_create = mock_client.embeddings.create

        with patch("brokle.wrappers.openai._OpenAI", MockOpenAI):
            with patch("brokle.wrappers.openai._AsyncOpenAI", MockAsyncOpenAI):
                result = wrap_openai(mock_client)

                # Methods should have been replaced with wrappers
                assert result.chat.completions.create is not original_chat_create
                assert result.completions.create is not original_completions_create
                assert result.embeddings.create is not original_embeddings_create


class TestWrapAnthropic:
    """Test wrap_anthropic() function."""

    def test_wrap_anthropic_import_available(self):
        """Test that wrap_anthropic is available in main imports."""
        from brokle import wrap_anthropic

        assert callable(wrap_anthropic)

    @patch("brokle.wrappers.anthropic.HAS_ANTHROPIC", False)
    def test_wrap_anthropic_no_sdk_installed(self):
        """Test error when Anthropic SDK not installed."""
        mock_client = Mock()
        with pytest.raises(ProviderError) as exc_info:
            wrap_anthropic(mock_client)
        assert "Anthropic SDK not installed" in str(exc_info.value)

    @patch("brokle.wrappers.anthropic.HAS_ANTHROPIC", True)
    def test_wrap_anthropic_invalid_client_type(self):
        """Test error when invalid client type passed."""

        class MockAnthropic:
            pass

        class MockAsyncAnthropic:
            pass

        with patch("brokle.wrappers.anthropic._Anthropic", MockAnthropic):
            with patch("brokle.wrappers.anthropic._AsyncAnthropic", MockAsyncAnthropic):
                mock_client = "not_a_client"
                with pytest.raises(ProviderError) as exc_info:
                    wrap_anthropic(mock_client)
                assert "expects Anthropic" in str(exc_info.value)

    @patch("brokle.wrappers.anthropic.HAS_ANTHROPIC", True)
    def test_wrap_anthropic_basic_success(self):
        """Test successful wrapping of Anthropic client."""

        class MockAnthropic:
            def __init__(self):
                self.messages = Mock()
                self.messages.create = Mock()

        class MockAsyncAnthropic:
            pass

        mock_client = MockAnthropic()

        with patch("brokle.wrappers.anthropic._Anthropic", MockAnthropic):
            with patch("brokle.wrappers.anthropic._AsyncAnthropic", MockAsyncAnthropic):
                result = wrap_anthropic(mock_client)

                # Verify wrapping markers are set
                assert result is mock_client
                assert hasattr(result, "_brokle_instrumented")
                assert result._brokle_instrumented is True
                assert hasattr(result, "_brokle_wrapper_version")

    @patch("brokle.wrappers.anthropic.HAS_ANTHROPIC", True)
    def test_wrap_anthropic_already_wrapped(self):
        """Test that wrapping an already-wrapped client is safe."""

        class MockAnthropic:
            def __init__(self):
                self.messages = Mock()
                self.messages.create = Mock()
                self._brokle_instrumented = False

        class MockAsyncAnthropic:
            pass

        mock_client = MockAnthropic()

        with patch("brokle.wrappers.anthropic._Anthropic", MockAnthropic):
            with patch("brokle.wrappers.anthropic._AsyncAnthropic", MockAsyncAnthropic):
                # First wrap
                result1 = wrap_anthropic(mock_client)
                assert result1._brokle_instrumented is True

                # Second wrap should return safely
                result2 = wrap_anthropic(mock_client)
                assert result2 is mock_client
                assert result2._brokle_instrumented is True

    @patch("brokle.wrappers.anthropic.HAS_ANTHROPIC", True)
    def test_wrap_anthropic_instruments_methods(self):
        """Test that wrap_anthropic instruments the expected methods."""

        class MockAnthropic:
            def __init__(self):
                self.messages = Mock()
                self.messages.create = Mock()

        class MockAsyncAnthropic:
            pass

        mock_client = MockAnthropic()
        original_messages_create = mock_client.messages.create

        with patch("brokle.wrappers.anthropic._Anthropic", MockAnthropic):
            with patch("brokle.wrappers.anthropic._AsyncAnthropic", MockAsyncAnthropic):
                result = wrap_anthropic(mock_client)

                # Method should have been replaced with wrapper
                assert result.messages.create is not original_messages_create


class TestWrapperPatternIntegration:
    """Integration tests for wrapper pattern."""

    def test_both_wrappers_available_together(self):
        """Test that both wrapper functions can be imported together."""
        from brokle import wrap_anthropic, wrap_openai

        assert callable(wrap_openai)
        assert callable(wrap_anthropic)

    def test_wrapper_functions_are_distinct(self):
        """Test that wrapper functions are different implementations."""
        from brokle import wrap_anthropic, wrap_openai

        assert wrap_openai is not wrap_anthropic
        assert wrap_openai.__name__ == "wrap_openai"
        assert wrap_anthropic.__name__ == "wrap_anthropic"
