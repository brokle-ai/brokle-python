"""
Test suite for wrapper functions - Pattern 1 implementation.

Tests the new explicit wrapper functions (wrap_openai, wrap_anthropic)
that replace the old drop-in replacement pattern.
"""

import warnings
from unittest.mock import MagicMock, Mock, patch

import pytest

from brokle import wrap_anthropic, wrap_openai
from brokle.exceptions import ProviderError, ValidationError


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
                assert "Expected OpenAI or AsyncOpenAI client" in str(exc_info.value)

    @patch("brokle.wrappers.openai.HAS_OPENAI", True)
    @patch("brokle.wrappers.openai._OpenAI")
    @patch("brokle.wrappers.openai._AsyncOpenAI")
    @patch("brokle.wrappers.openai.UniversalInstrumentation")
    @patch("brokle.wrappers.openai.get_provider")
    def test_wrap_openai_basic_success(
        self,
        mock_provider_class,
        mock_instrumentation_class,
        mock_async_openai_class,
        mock_openai_class,
    ):
        """Test successful wrapping of OpenAI client."""
        # Setup mocks
        mock_client = Mock()
        mock_client._brokle_instrumented = False
        mock_openai_class.return_value = mock_client

        mock_provider = Mock()
        mock_provider.name = "openai"
        mock_provider_class.return_value = mock_provider

        mock_instrumentation = Mock()
        mock_instrumentation.instrument_client.return_value = mock_client
        mock_instrumentation_class.return_value = mock_instrumentation

        # Execute
        result = wrap_openai(mock_client)

        # Verify
        assert result is mock_client
        assert hasattr(result, "_brokle_instrumented")
        assert hasattr(result, "_brokle_provider")
        assert hasattr(result, "_brokle_wrapper_version")
        mock_instrumentation.instrument_client.assert_called_once_with(mock_client)

    @patch("brokle.wrappers.openai.HAS_OPENAI", True)
    @patch("brokle.wrappers.openai._OpenAI")
    @patch("brokle.wrappers.openai._AsyncOpenAI")
    def test_wrap_openai_already_wrapped_warning(
        self, mock_async_openai_class, mock_openai_class
    ):
        """Test warning when client already wrapped."""
        mock_client = Mock()
        mock_client._brokle_instrumented = True
        mock_openai_class.return_value = mock_client

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = wrap_openai(mock_client)

            assert len(w) == 1
            assert "already wrapped" in str(w[0].message)
            assert result is mock_client

    @patch("brokle.wrappers.openai.HAS_OPENAI", True)
    @patch("brokle.wrappers.openai._OpenAI")
    @patch("brokle.wrappers.openai._AsyncOpenAI")
    def test_wrap_openai_invalid_tags_config(
        self, mock_async_openai_class, mock_openai_class
    ):
        """Test validation of invalid tags configuration."""
        mock_client = Mock()
        mock_client._brokle_instrumented = False
        with pytest.raises(ValidationError) as exc_info:
            wrap_openai(mock_client, tags="not_a_list")
        assert "tags must be a list" in str(exc_info.value)

    @patch("brokle.wrappers.openai.HAS_OPENAI", True)
    @patch("brokle.wrappers.openai._OpenAI")
    @patch("brokle.wrappers.openai._AsyncOpenAI")
    def test_wrap_openai_invalid_session_id_config(
        self, mock_async_openai_class, mock_openai_class
    ):
        """Test validation of invalid session_id configuration."""
        mock_client = Mock()
        mock_client._brokle_instrumented = False
        with pytest.raises(ValidationError) as exc_info:
            wrap_openai(mock_client, session_id=123)
        assert "session_id must be a string" in str(exc_info.value)


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

        # Create proper mock classes for isinstance() checks
        class MockAnthropic:
            pass

        class MockAsyncAnthropic:
            pass

        with patch("brokle.wrappers.anthropic._Anthropic", MockAnthropic):
            with patch("brokle.wrappers.anthropic._AsyncAnthropic", MockAsyncAnthropic):
                mock_client = "not_a_client"
                with pytest.raises(ProviderError) as exc_info:
                    wrap_anthropic(mock_client)
                assert "Expected Anthropic or AsyncAnthropic client" in str(
                    exc_info.value
                )

    @patch("brokle.wrappers.anthropic.HAS_ANTHROPIC", True)
    @patch("brokle.wrappers.anthropic._Anthropic")
    @patch("brokle.wrappers.anthropic._AsyncAnthropic")
    @patch("brokle.wrappers.anthropic.UniversalInstrumentation")
    @patch("brokle.wrappers.anthropic.get_provider")
    def test_wrap_anthropic_basic_success(
        self,
        mock_provider_class,
        mock_instrumentation_class,
        mock_async_anthropic_class,
        mock_anthropic_class,
    ):
        """Test successful wrapping of Anthropic client."""
        # Setup mocks
        mock_client = Mock()
        mock_client._brokle_instrumented = False
        mock_anthropic_class.return_value = mock_client

        mock_provider = Mock()
        mock_provider.name = "anthropic"
        mock_provider_class.return_value = mock_provider

        mock_instrumentation = Mock()
        mock_instrumentation.instrument_client.return_value = mock_client
        mock_instrumentation_class.return_value = mock_instrumentation

        # Execute
        result = wrap_anthropic(mock_client)

        # Verify
        assert result is mock_client
        assert hasattr(result, "_brokle_instrumented")
        assert hasattr(result, "_brokle_provider")
        assert hasattr(result, "_brokle_wrapper_version")
        mock_instrumentation.instrument_client.assert_called_once_with(mock_client)


class TestDeprecationWarnings:
    """Test that old import patterns raise proper deprecation errors."""

    def test_old_openai_import_raises_error(self):
        """Test that trying to import brokle.openai raises ImportError."""
        with pytest.raises(ImportError) as exc_info:
            import brokle.openai
        # Module was deleted, so Python raises standard ModuleNotFoundError
        assert "No module named 'brokle.openai'" in str(exc_info.value)

    def test_old_anthropic_import_raises_error(self):
        """Test that trying to import brokle.anthropic raises ImportError."""
        with pytest.raises(ImportError) as exc_info:
            import brokle.anthropic
        # Module was deleted, so Python raises standard ModuleNotFoundError
        assert "No module named 'brokle.anthropic'" in str(exc_info.value)

    def test_getattr_unknown_attribute(self):
        """Test __getattr__ for unknown attributes."""
        import brokle

        with pytest.raises(AttributeError) as exc_info:
            _ = brokle.unknown_attribute
        assert "has no attribute 'unknown_attribute'" in str(exc_info.value)


class TestConfiguration:
    """Test wrapper configuration validation."""

    def test_valid_configuration(self):
        """Test that valid configuration parameters work."""
        config = {
            "capture_content": True,
            "capture_metadata": False,
            "tags": ["production", "test"],
            "session_id": "session_123",
            "user_id": "user_456",
        }

        # This should not raise any validation errors
        from brokle._utils.wrapper_validation import validate_wrapper_config

        validate_wrapper_config(**config)

    def test_invalid_capture_content_type(self):
        """Test validation of capture_content parameter."""
        from brokle._utils.wrapper_validation import validate_wrapper_config

        with pytest.raises(ValidationError) as exc_info:
            validate_wrapper_config(capture_content="not_a_bool")
        assert "capture_content must be a boolean" in str(exc_info.value)

    def test_invalid_tags_not_list(self):
        """Test validation of tags parameter type."""
        from brokle._utils.wrapper_validation import validate_wrapper_config

        with pytest.raises(ValidationError) as exc_info:
            validate_wrapper_config(tags="not_a_list")
        assert "tags must be a list" in str(exc_info.value)

    def test_invalid_tag_not_string(self):
        """Test validation of individual tag types."""
        from brokle._utils.wrapper_validation import validate_wrapper_config

        with pytest.raises(ValidationError) as exc_info:
            validate_wrapper_config(tags=[123, "valid_tag"])
        assert "tags[0] must be a string" in str(exc_info.value)

    def test_tag_too_long(self):
        """Test validation of tag length."""
        from brokle._utils.wrapper_validation import validate_wrapper_config

        long_tag = "a" * 51  # 51 characters, over the limit
        with pytest.raises(ValidationError) as exc_info:
            validate_wrapper_config(tags=[long_tag])
        assert "must be <= 50 characters" in str(exc_info.value)

    def test_session_id_too_long(self):
        """Test validation of session_id length."""
        from brokle._utils.wrapper_validation import validate_wrapper_config

        long_session_id = "a" * 101  # 101 characters, over the limit
        with pytest.raises(ValidationError) as exc_info:
            validate_wrapper_config(session_id=long_session_id)
        assert "must be <= 100 characters" in str(exc_info.value)

    def test_user_id_too_long(self):
        """Test validation of user_id length."""
        from brokle._utils.wrapper_validation import validate_wrapper_config

        long_user_id = "a" * 101  # 101 characters, over the limit
        with pytest.raises(ValidationError) as exc_info:
            validate_wrapper_config(user_id=long_user_id)
        assert "must be <= 100 characters" in str(exc_info.value)
