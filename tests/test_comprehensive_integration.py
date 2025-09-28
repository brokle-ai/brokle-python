"""
Comprehensive integration test for Brokle SDK 2.0 architecture.

Tests all three patterns working with shared provider system:
- Pattern 1: Wrapper functions (wrap_openai, wrap_anthropic)
- Pattern 2: @observe decorator with AI detection
- Pattern 3: Native SDK with enhanced capabilities
"""

from unittest.mock import Mock, patch

import pytest

from brokle import Brokle, get_client, observe, wrap_anthropic, wrap_openai
from brokle.providers import get_provider, list_providers


class TestComprehensiveIntegration:
    """Test complete SDK integration with all three patterns."""

    def test_all_patterns_import_successfully(self):
        """Test all three patterns can be imported and basic functionality works."""
        # Test imports work
        assert wrap_openai is not None
        assert wrap_anthropic is not None
        assert observe is not None
        assert Brokle is not None
        assert get_client is not None

    def test_shared_providers_available(self):
        """Test shared providers are accessible to all patterns."""
        providers = list_providers()
        assert "openai" in providers
        assert "anthropic" in providers

        # Test providers can be instantiated
        openai_provider = get_provider("openai")
        anthropic_provider = get_provider("anthropic")

        assert openai_provider.get_provider_name() == "openai"
        assert anthropic_provider.get_provider_name() == "anthropic"

    @patch("brokle.wrappers.openai.HAS_OPENAI", True)
    @patch("brokle.wrappers.openai._OpenAI")
    def test_pattern_1_wrapper_with_shared_providers(self, mock_openai_class):
        """Test Pattern 1 uses shared providers correctly."""
        mock_client = Mock()
        mock_client._brokle_instrumented = False
        mock_openai_class.return_value = mock_client

        # This should work and use shared providers internally
        with patch("brokle.wrappers.openai.get_provider") as mock_get_provider:
            mock_provider = Mock()
            mock_get_provider.return_value = mock_provider

            with patch("brokle.wrappers.openai.UniversalInstrumentation") as mock_instr:
                mock_instr.return_value.instrument_client.return_value = mock_client

                result = wrap_openai(mock_client)

                # Verify shared provider was used
                mock_get_provider.assert_called_once_with(
                    "openai",
                    capture_content=True,
                    capture_metadata=True,
                    tags=[],
                    session_id=None,
                    user_id=None,
                )
                assert result == mock_client

    def test_pattern_2_decorator_has_ai_awareness(self):
        """Test Pattern 2 @observe decorator has AI detection capabilities."""

        # Create a mock function that might work with AI
        @observe(name="ai-test")
        def mock_ai_function(client, query):
            return f"processed: {query}"

        # Test function can be called (decorator applied successfully)
        result = mock_ai_function("mock_client", "test query")
        assert result == "processed: test query"

    def test_pattern_3_native_sdk_with_enhanced_capabilities(self):
        """Test Pattern 3 native SDK has enhanced generation capabilities."""
        # Test that enhanced generation methods exist
        from brokle.observability import BrokleGeneration

        # Test that enhanced methods exist
        assert hasattr(BrokleGeneration, "update_with_request_attributes")
        assert hasattr(BrokleGeneration, "update_with_response_attributes")
        assert hasattr(BrokleGeneration, "create_from_ai_request")

    def test_providers_work_across_patterns(self):
        """Test that provider functionality is consistent across all patterns."""
        # Get providers directly
        openai_provider = get_provider("openai")
        anthropic_provider = get_provider("anthropic")

        # Test provider methods exist and work
        assert callable(openai_provider.extract_request_attributes)
        assert callable(openai_provider.extract_response_attributes)
        assert callable(anthropic_provider.extract_request_attributes)
        assert callable(anthropic_provider.extract_response_attributes)

        # Test with sample request
        sample_request = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
        }

        openai_attrs = openai_provider.extract_request_attributes(sample_request)
        assert len(openai_attrs) > 0
        assert "llm.model" in openai_attrs or "llm.model.normalized" in openai_attrs

    def test_architecture_scalability(self):
        """Test that the architecture is ready for new providers."""
        # Test that adding a new provider would be straightforward
        providers = list_providers()

        # Current providers
        assert len(providers) == 2
        assert "openai" in providers
        assert "anthropic" in providers

        # Test provider registry is extensible
        from brokle.providers import register_provider

        assert callable(register_provider)

    def test_sdk_version_updated(self):
        """Test that SDK version is properly set and follows semantic versioning."""
        from brokle._version import __version__
        import re

        # Test that version exists and follows semantic versioning pattern
        assert __version__ is not None, "Version should not be None"
        assert isinstance(__version__, str), "Version should be a string"

        # Test semantic versioning format (major.minor.patch)
        semver_pattern = r'^\d+\.\d+\.\d+$'
        assert re.match(semver_pattern, __version__), f"Version {__version__} should follow semantic versioning format (X.Y.Z)"

        # Test that version is reasonable (not empty, not too long)
        assert len(__version__) >= 5, f"Version {__version__} seems too short"
        assert len(__version__) <= 20, f"Version {__version__} seems too long"

    def test_clean_public_api(self):
        """Test that public API exports are clean and complete."""
        import brokle

        # Essential exports should be available
        essential_exports = [
            "wrap_openai",
            "wrap_anthropic",  # Pattern 1
            "observe",  # Pattern 2
            "Brokle",
            "get_client",  # Pattern 3
        ]

        for export in essential_exports:
            assert hasattr(brokle, export), f"Missing essential export: {export}"

        # Test that __all__ contains expected items
        assert hasattr(brokle, "__all__")
        all_exports = brokle.__all__
        for export in essential_exports:
            assert export in all_exports, f"{export} not in __all__"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
