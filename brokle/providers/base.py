"""
Abstract Base Provider Interface

Defines the contract that all AI provider integrations must implement.
Ensures consistent observability patterns across OpenAI, Anthropic, Google, etc.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import logging
import re

logger = logging.getLogger(__name__)


class BaseProvider(ABC):
    """
    Abstract base class for all AI provider integrations.

    Each provider must implement this interface to ensure consistent
    observability, error handling, and instrumentation patterns.
    """

    def __init__(self, **config):
        """
        Initialize provider with configuration.

        Args:
            **config: Provider-specific configuration options
        """
        self.config = config
        self.name = self.get_provider_name()
        self._cost_cache: Dict[str, float] = {}

    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Return the provider identifier.

        Returns:
            Provider name (e.g., 'openai', 'anthropic', 'google')
        """
        pass

    @abstractmethod
    def get_methods_to_instrument(self) -> List[Dict[str, Any]]:
        """
        Define which SDK methods to instrument.

        Returns:
            List of method definitions with instrumentation metadata

        Format:
        [
            {
                'path': 'chat.completions.create',      # SDK method path
                'operation': 'chat_completion',         # Operation type for telemetry
                'async': False,                         # Whether method is async
                'stream_support': True,                 # Whether method supports streaming
                'cost_tracked': True                    # Whether to track costs
            },
            {
                'path': 'embeddings.create',
                'operation': 'embedding',
                'async': False,
                'stream_support': False,
                'cost_tracked': True
            }
        ]
        """
        pass

    @abstractmethod
    def extract_request_attributes(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract OpenTelemetry attributes from request parameters.

        Args:
            kwargs: Method call arguments (model, messages, etc.)

        Returns:
            Dictionary mapping attribute names to values

        Required attributes to extract:
        - model_name: kwargs.get('model')
        - message_count: len(kwargs.get('messages', []))
        - max_tokens: kwargs.get('max_tokens')
        - temperature: kwargs.get('temperature')
        - stream_enabled: kwargs.get('stream', False)
        - input_tokens: estimated token count from messages

        Example implementation:
        {
            'llm.model': kwargs.get('model'),
            'llm.input.message_count': len(kwargs.get('messages', [])),
            'llm.request.max_tokens': kwargs.get('max_tokens'),
            'llm.request.temperature': kwargs.get('temperature', 1.0),
            'llm.request.stream': kwargs.get('stream', False),
            'llm.usage.prompt_tokens': self.estimate_input_tokens(kwargs)
        }
        """
        pass

    @abstractmethod
    def extract_response_attributes(self, response: Any) -> Dict[str, Any]:
        """
        Extract OpenTelemetry attributes from provider response.

        Args:
            response: Provider SDK response object

        Returns:
            Dictionary mapping attribute names to values

        Required attributes to extract:
        - output_tokens: response.usage.completion_tokens
        - total_tokens: response.usage.total_tokens
        - response_content_length: len(response.choices[0].message.content)
        - finish_reason: response.choices[0].finish_reason
        - cost_usd: calculated cost based on usage and model

        Handle different response formats:
        - Streaming vs non-streaming responses
        - Error responses vs successful responses
        - Different SDK versions and response structures
        """
        pass

    def estimate_input_tokens(self, kwargs: Dict[str, Any]) -> int:
        """
        Estimate input token count from request parameters.

        Args:
            kwargs: Method arguments containing messages, prompt, etc.

        Returns:
            Estimated input token count

        Default implementation provides rough approximation.
        Providers should override with more accurate tokenization.
        """
        total_chars = 0

        # Handle different input formats
        if 'messages' in kwargs:
            messages = kwargs['messages']
            if isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict):
                        content = msg.get('content', '')
                        if isinstance(content, str):
                            total_chars += len(content)
                        elif isinstance(content, list):
                            # Handle multimodal content
                            for item in content:
                                if isinstance(item, dict) and item.get('type') == 'text':
                                    total_chars += len(item.get('text', ''))

        elif 'prompt' in kwargs:
            prompt = kwargs['prompt']
            if isinstance(prompt, str):
                total_chars += len(prompt)
            elif isinstance(prompt, list):
                total_chars += sum(len(str(p)) for p in prompt)

        # Rough token estimation (4 characters ≈ 1 token)
        return max(1, total_chars // 4)

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost in USD for the request.

        Args:
            model: Model name (e.g., 'gpt-4', 'claude-3-opus')
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD

        Default implementation returns 0.0.
        Providers should override with actual pricing data.
        """
        # Cache key for pricing lookup
        cache_key = f"{model}_{input_tokens}_{output_tokens}"
        if cache_key in self._cost_cache:
            return self._cost_cache[cache_key]

        # Default implementation - providers should override
        cost = 0.0
        self._cost_cache[cache_key] = cost
        return cost

    def get_supported_models(self) -> List[str]:
        """
        Return list of supported models for this provider.

        Returns:
            List of model identifiers

        Used for validation and cost calculation.
        """
        return []

    def normalize_model_name(self, model: str) -> str:
        """
        Normalize model name for consistent telemetry.

        Args:
            model: Raw model name from request

        Returns:
            Normalized model name

        Example:
        - 'gpt-4-0613' → 'gpt-4'
        - 'claude-3-opus-20240229' → 'claude-3-opus'
        """
        # Remove version suffixes and normalize
        normalized = re.sub(r'-\d{8}$', '', model)  # Remove date suffixes
        normalized = re.sub(r'-\d+k$', '', normalized)  # Remove context length suffixes
        return normalized.lower()

    def handle_streaming_response(self, response_stream: Any) -> Dict[str, Any]:
        """
        Handle streaming response aggregation.

        Args:
            response_stream: Generator/iterator of response chunks

        Returns:
            Aggregated response attributes

        Default implementation returns empty dict.
        Providers should override for streaming support.
        """
        return {}

    def validate_request(self, kwargs: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate request parameters before instrumentation.

        Args:
            kwargs: Method call arguments

        Returns:
            Tuple of (is_valid, error_message)

        Default implementation always returns (True, None).
        Providers can override for request validation.
        """
        return True, None

    def get_error_mapping(self) -> Dict[str, str]:
        """
        Map provider-specific errors to Brokle error types.

        Returns:
            Dictionary mapping provider exception names to Brokle exception names

        Example:
        {
            'AuthenticationError': 'AuthenticationError',
            'RateLimitError': 'RateLimitError',
            'InvalidRequestError': 'ValidationError',
            'InternalServerError': 'ProviderError'
        }
        """
        return {}

    def get_instrumentation_config(self) -> Dict[str, Any]:
        """
        Get configuration specific to this provider's instrumentation.

        Returns:
            Dictionary with provider-specific instrumentation settings
        """
        return {
            'provider': self.name,
            'capture_content': self.config.get('capture_content', True),
            'capture_metadata': self.config.get('capture_metadata', True),
            'tags': self.config.get('tags', []),
            'session_id': self.config.get('session_id'),
            'user_id': self.config.get('user_id'),
        }