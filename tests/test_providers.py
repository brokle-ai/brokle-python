"""
Test suite for provider implementations.

Tests the provider-specific logic for OpenAI and Anthropic.
"""

from unittest.mock import Mock

import pytest

from brokle.observability.attributes import (
    BrokleOtelSpanAttributes as BrokleInstrumentationAttributes,
)
from brokle.providers.anthropic import AnthropicProvider
from brokle.providers.openai import OpenAIProvider


class TestOpenAIProvider:
    """Test OpenAI provider implementation."""

    def test_provider_name(self):
        """Test provider returns correct name."""
        provider = OpenAIProvider()
        assert provider.get_provider_name() == "openai"

    def test_methods_to_instrument(self):
        """Test provider defines methods to instrument."""
        provider = OpenAIProvider()
        methods = provider.get_methods_to_instrument()

        assert isinstance(methods, list)
        assert len(methods) > 0

        # Check for key methods
        method_paths = [method["path"] for method in methods]
        assert "chat.completions.create" in method_paths
        assert "embeddings.create" in method_paths

        # Check method structure
        for method in methods:
            assert "path" in method
            assert "operation" in method
            assert "async" in method
            assert "stream_support" in method
            assert "cost_tracked" in method

    def test_extract_request_attributes_chat(self):
        """Test request attribute extraction for chat completion."""
        provider = OpenAIProvider()
        kwargs = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
            "max_tokens": 100,
            "temperature": 0.7,
            "stream": True,
        }

        attributes = provider.extract_request_attributes(kwargs)

        assert attributes[BrokleInstrumentationAttributes.MODEL_NAME] == "gpt-4"
        assert attributes[BrokleInstrumentationAttributes.MESSAGE_COUNT] == 2
        assert attributes[BrokleInstrumentationAttributes.MAX_TOKENS] == 100
        assert attributes[BrokleInstrumentationAttributes.TEMPERATURE] == 0.7
        assert attributes[BrokleInstrumentationAttributes.STREAM_ENABLED] is True
        assert attributes[BrokleInstrumentationAttributes.SYSTEM_MESSAGE] is True
        assert BrokleInstrumentationAttributes.INPUT_TOKENS in attributes

    def test_extract_response_attributes_chat(self):
        """Test response attribute extraction for chat completion."""
        provider = OpenAIProvider()

        # Mock OpenAI response
        mock_response = Mock()
        mock_response.model = "gpt-4"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 15
        mock_response.usage.total_tokens = 35

        mock_response.choices = [Mock()]
        mock_response.choices[0].finish_reason = "stop"
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Hello! How can I help you?"

        attributes = provider.extract_response_attributes(mock_response)

        assert attributes[BrokleInstrumentationAttributes.INPUT_TOKENS] == 20
        assert attributes[BrokleInstrumentationAttributes.OUTPUT_TOKENS] == 15
        assert attributes[BrokleInstrumentationAttributes.TOTAL_TOKENS] == 35
        assert attributes[BrokleInstrumentationAttributes.FINISH_REASON] == "stop"
        assert attributes[BrokleInstrumentationAttributes.RESPONSE_CONTENT_LENGTH] == 26
        assert BrokleInstrumentationAttributes.COST_USD not in attributes

    def test_normalize_model_name(self):
        """Test model name normalization."""
        provider = OpenAIProvider()

        assert provider.normalize_model_name("gpt-4-0613") == "gpt-4"
        assert provider.normalize_model_name("gpt-4-turbo-preview") == "gpt-4-turbo"
        assert provider.normalize_model_name("gpt-3.5-turbo-0125") == "gpt-3.5-turbo"
        assert provider.normalize_model_name("gpt-4") == "gpt-4"

    def test_error_mapping(self):
        """Test error mapping for OpenAI errors."""
        provider = OpenAIProvider()
        error_mapping = provider.get_error_mapping()

        assert error_mapping["AuthenticationError"] == "AuthenticationError"
        assert error_mapping["RateLimitError"] == "RateLimitError"
        assert error_mapping["BadRequestError"] == "ValidationError"
        assert error_mapping["InternalServerError"] == "ProviderError"


class TestAnthropicProvider:
    """Test Anthropic provider implementation."""

    def test_provider_name(self):
        """Test provider returns correct name."""
        provider = AnthropicProvider()
        assert provider.get_provider_name() == "anthropic"

    def test_methods_to_instrument(self):
        """Test provider defines methods to instrument."""
        provider = AnthropicProvider()
        methods = provider.get_methods_to_instrument()

        assert isinstance(methods, list)
        assert len(methods) > 0

        # Check for key methods
        method_paths = [method["path"] for method in methods]
        assert "messages.create" in method_paths

        # Check method structure
        for method in methods:
            assert "path" in method
            assert "operation" in method
            assert "async" in method
            assert "stream_support" in method
            assert "cost_tracked" in method

    def test_extract_request_attributes_messages(self):
        """Test request attribute extraction for messages."""
        provider = AnthropicProvider()
        kwargs = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Hello Claude!"}],
            "max_tokens": 1000,
            "temperature": 0.7,
            "system": "You are Claude, an AI assistant.",
            "stream": False,
        }

        attributes = provider.extract_request_attributes(kwargs)

        assert (
            attributes[BrokleInstrumentationAttributes.MODEL_NAME]
            == "claude-3-opus-20240229"
        )
        assert attributes[BrokleInstrumentationAttributes.MESSAGE_COUNT] == 1
        assert attributes[BrokleInstrumentationAttributes.MAX_TOKENS] == 1000
        assert attributes[BrokleInstrumentationAttributes.TEMPERATURE] == 0.7
        assert attributes[BrokleInstrumentationAttributes.SYSTEM_MESSAGE] is True
        assert attributes[BrokleInstrumentationAttributes.STREAM_ENABLED] is False
        assert BrokleInstrumentationAttributes.INPUT_TOKENS in attributes

    def test_extract_response_attributes_messages(self):
        """Test response attribute extraction for messages."""
        provider = AnthropicProvider()

        # Mock Anthropic response
        mock_response = Mock()
        mock_response.model = "claude-3-opus-20240229"
        mock_response.usage = Mock()
        mock_response.usage.input_tokens = 25
        mock_response.usage.output_tokens = 35

        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Hello! I'm Claude, how can I help you today?"

        attributes = provider.extract_response_attributes(mock_response)

        assert attributes[BrokleInstrumentationAttributes.INPUT_TOKENS] == 25
        assert attributes[BrokleInstrumentationAttributes.OUTPUT_TOKENS] == 35
        assert attributes[BrokleInstrumentationAttributes.TOTAL_TOKENS] == 60
        assert attributes[BrokleInstrumentationAttributes.STOP_REASON] == "end_turn"
        assert attributes[BrokleInstrumentationAttributes.RESPONSE_CONTENT_LENGTH] == 44
        assert BrokleInstrumentationAttributes.COST_USD not in attributes

    def test_normalize_model_name(self):
        """Test model name normalization for Anthropic."""
        provider = AnthropicProvider()

        assert (
            provider.normalize_model_name("claude-3-opus-20240229") == "claude-3-opus"
        )
        assert (
            provider.normalize_model_name("claude-3-sonnet-20240229")
            == "claude-3-sonnet"
        )
        assert provider.normalize_model_name("claude-3-haiku") == "claude-3-haiku"

    def test_estimate_input_tokens_multimodal(self):
        """Test input token estimation with multimodal content."""
        provider = AnthropicProvider()

        kwargs = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What do you see in this image?"},
                        {
                            "type": "image",
                            "source": {"type": "base64", "data": "fake_image_data"},
                        },
                    ],
                }
            ],
            "system": "You are a helpful assistant.",
        }

        tokens = provider.estimate_input_tokens(kwargs)
        # Should include text tokens + image tokens + system message tokens
        assert tokens > 1000  # Image adds ~1000 tokens

    def test_error_mapping(self):
        """Test error mapping for Anthropic errors."""
        provider = AnthropicProvider()
        error_mapping = provider.get_error_mapping()

        assert error_mapping["AuthenticationError"] == "AuthenticationError"
        assert error_mapping["RateLimitError"] == "RateLimitError"
        assert error_mapping["BadRequestError"] == "ValidationError"
        assert error_mapping["InternalServerError"] == "ProviderError"


class TestBaseProviderInterface:
    """Test that providers implement the required interface."""

    def test_openai_provider_implements_interface(self):
        """Test OpenAI provider implements all required methods."""
        provider = OpenAIProvider()

        # Check all abstract methods are implemented
        assert hasattr(provider, "get_provider_name")
        assert hasattr(provider, "get_methods_to_instrument")
        assert hasattr(provider, "extract_request_attributes")
        assert hasattr(provider, "extract_response_attributes")

        # Check they're callable
        assert callable(provider.get_provider_name)
        assert callable(provider.get_methods_to_instrument)
        assert callable(provider.extract_request_attributes)
        assert callable(provider.extract_response_attributes)

    def test_anthropic_provider_implements_interface(self):
        """Test Anthropic provider implements all required methods."""
        provider = AnthropicProvider()

        # Check all abstract methods are implemented
        assert hasattr(provider, "get_provider_name")
        assert hasattr(provider, "get_methods_to_instrument")
        assert hasattr(provider, "extract_request_attributes")
        assert hasattr(provider, "extract_response_attributes")

        # Check they're callable
        assert callable(provider.get_provider_name)
        assert callable(provider.get_methods_to_instrument)
        assert callable(provider.extract_request_attributes)
        assert callable(provider.extract_response_attributes)
