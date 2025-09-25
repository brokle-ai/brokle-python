"""
Anthropic Provider Implementation

Specific instrumentation logic for Anthropic SDK with comprehensive
support for messages API and multimodal content.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import re

from .base import BaseProvider
from ..observability.attributes import BrokleOtelSpanAttributes as BrokleInstrumentationAttributes
from ..exceptions import ProviderError, ValidationError

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseProvider):
    """Anthropic-specific provider implementation."""

    # Anthropic pricing per 1K tokens (as of 2024)
    MODEL_PRICING = {
        'claude-3-opus': {'input': 0.015, 'output': 0.075},
        'claude-3-sonnet': {'input': 0.003, 'output': 0.015},
        'claude-3-haiku': {'input': 0.00025, 'output': 0.00125},
        'claude-3-5-sonnet': {'input': 0.003, 'output': 0.015},
        'claude-2.1': {'input': 0.008, 'output': 0.024},
        'claude-2.0': {'input': 0.008, 'output': 0.024},
        'claude-instant-1.2': {'input': 0.0008, 'output': 0.0024},
    }

    def get_provider_name(self) -> str:
        """Return Anthropic provider identifier."""
        return "anthropic"

    def get_methods_to_instrument(self) -> List[Dict[str, Any]]:
        """Define Anthropic SDK methods to instrument."""
        return [
            # Messages API (primary API for Claude)
            {
                'path': 'messages.create',
                'operation': 'chat_completion',
                'async': False,
                'stream_support': True,
                'cost_tracked': True
            },
            {
                'path': 'messages.acreate',
                'operation': 'chat_completion',
                'async': True,
                'stream_support': True,
                'cost_tracked': True
            },

            # Completions API (legacy, but still supported)
            {
                'path': 'completions.create',
                'operation': 'completion',
                'async': False,
                'stream_support': True,
                'cost_tracked': True
            },

            # Beta features (tool use, etc.)
            {
                'path': 'beta.messages.create',
                'operation': 'chat_completion_beta',
                'async': False,
                'stream_support': True,
                'cost_tracked': True
            },

            # Batch API (if available)
            {
                'path': 'messages.batch',
                'operation': 'batch_completion',
                'async': False,
                'stream_support': False,
                'cost_tracked': True
            },
        ]

    def extract_request_attributes(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Anthropic request attributes for telemetry."""
        attributes = {}

        # Model information
        if 'model' in kwargs:
            model = kwargs['model']
            attributes[BrokleInstrumentationAttributes.MODEL_NAME] = model
            attributes[BrokleInstrumentationAttributes.MODEL_NAME_NORMALIZED] = self.normalize_model_name(model)

        # Messages API specific
        if 'messages' in kwargs:
            messages = kwargs['messages']
            if isinstance(messages, list):
                attributes[BrokleInstrumentationAttributes.MESSAGE_COUNT] = len(messages)

                # Extract message roles and types
                roles = [msg.get('role', 'unknown') for msg in messages if isinstance(msg, dict)]
                attributes[BrokleInstrumentationAttributes.MESSAGE_ROLES] = ','.join(roles)

        # System message (separate parameter in Anthropic)
        if 'system' in kwargs:
            attributes[BrokleInstrumentationAttributes.SYSTEM_MESSAGE] = bool(kwargs['system'])

        # Legacy completion specific
        if 'prompt' in kwargs:
            prompt = kwargs['prompt']
            if isinstance(prompt, str):
                attributes[BrokleInstrumentationAttributes.PROMPT_LENGTH] = len(prompt)

        # Common parameters
        common_params = {
            'max_tokens': BrokleInstrumentationAttributes.MAX_TOKENS,
            'temperature': BrokleInstrumentationAttributes.TEMPERATURE,
            'top_p': BrokleInstrumentationAttributes.TOP_P,
        }

        for param, attr in common_params.items():
            if param in kwargs and kwargs[param] is not None:
                attributes[attr] = kwargs[param]

        # Streaming
        if 'stream' in kwargs:
            attributes[BrokleInstrumentationAttributes.STREAM_ENABLED] = kwargs['stream']

        # Tool use (beta feature)
        if 'tools' in kwargs:
            tools = kwargs['tools']
            if isinstance(tools, list):
                attributes[BrokleInstrumentationAttributes.TOOL_COUNT] = len(tools)
                tool_types = [t.get('type', 'unknown') for t in tools if isinstance(t, dict)]
                attributes[BrokleInstrumentationAttributes.TOOL_TYPES] = ','.join(set(tool_types))

        # Estimate input tokens
        input_tokens = self.estimate_input_tokens(kwargs)
        attributes[BrokleInstrumentationAttributes.INPUT_TOKENS] = input_tokens

        return attributes

    def extract_response_attributes(self, response: Any) -> Dict[str, Any]:
        """Extract Anthropic response attributes for telemetry."""
        attributes = {}

        try:
            # Handle usage information
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage

                # Token usage
                if hasattr(usage, 'input_tokens'):
                    attributes[BrokleInstrumentationAttributes.INPUT_TOKENS] = usage.input_tokens
                if hasattr(usage, 'output_tokens'):
                    attributes[BrokleInstrumentationAttributes.OUTPUT_TOKENS] = usage.output_tokens

                # Calculate total tokens
                input_tokens = getattr(usage, 'input_tokens', 0)
                output_tokens = getattr(usage, 'output_tokens', 0)
                attributes[BrokleInstrumentationAttributes.TOTAL_TOKENS] = input_tokens + output_tokens

                # Calculate cost
                if hasattr(response, 'model'):
                    cost = self.calculate_cost(response.model, input_tokens, output_tokens)
                    if cost > 0:
                        attributes[BrokleInstrumentationAttributes.COST_USD] = cost

            # Model from response
            if hasattr(response, 'model'):
                attributes[BrokleInstrumentationAttributes.MODEL_NAME_RESPONSE] = response.model

            # Response content
            if hasattr(response, 'content') and response.content:
                content_length = 0
                for item in response.content:
                    if hasattr(item, 'text'):
                        content_length += len(item.text)

                attributes[BrokleInstrumentationAttributes.RESPONSE_CONTENT_LENGTH] = content_length

            # Stop reason
            if hasattr(response, 'stop_reason'):
                attributes[BrokleInstrumentationAttributes.STOP_REASON] = response.stop_reason

            # Tool use in response
            if hasattr(response, 'content') and response.content:
                tool_calls = []
                for item in response.content:
                    if hasattr(item, 'type') and item.type == 'tool_use':
                        if hasattr(item, 'name'):
                            tool_calls.append(item.name)

                if tool_calls:
                    attributes[BrokleInstrumentationAttributes.TOOL_CALL_NAMES] = ','.join(tool_calls)

        except Exception as e:
            logger.warning(f"Failed to extract Anthropic response attributes: {e}")

        return attributes

    def estimate_input_tokens(self, kwargs: Dict[str, Any]) -> int:
        """Estimate input token count from Anthropic request parameters."""
        total_chars = 0

        # Handle messages format
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
                                if isinstance(item, dict):
                                    if item.get('type') == 'text':
                                        total_chars += len(item.get('text', ''))
                                    elif item.get('type') == 'image':
                                        # Images count as ~1000 tokens approximately
                                        total_chars += 4000  # 4000 chars ≈ 1000 tokens

        # Handle system message
        if 'system' in kwargs:
            system = kwargs['system']
            if isinstance(system, str):
                total_chars += len(system)

        # Handle legacy prompt format
        elif 'prompt' in kwargs:
            prompt = kwargs['prompt']
            if isinstance(prompt, str):
                total_chars += len(prompt)

        # Rough token estimation (4 characters ≈ 1 token)
        return max(1, total_chars // 4)

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate Anthropic API cost based on current pricing."""
        normalized_model = self.normalize_model_name(model)

        if normalized_model not in self.MODEL_PRICING:
            logger.warning(f"Unknown Anthropic model for cost calculation: {model}")
            return 0.0

        pricing = self.MODEL_PRICING[normalized_model]

        # Calculate cost per 1K tokens
        input_cost = (input_tokens / 1000) * pricing['input']
        output_cost = (output_tokens / 1000) * pricing['output']

        total_cost = input_cost + output_cost
        return round(total_cost, 6)  # Round to 6 decimal places for accuracy

    def get_supported_models(self) -> List[str]:
        """Return list of supported Anthropic models."""
        return list(self.MODEL_PRICING.keys())

    def normalize_model_name(self, model: str) -> str:
        """Normalize Anthropic model names for consistent telemetry."""
        # Remove version suffixes
        normalized = re.sub(r'-\d{8}$', '', model)  # Remove date suffixes
        normalized = re.sub(r'-v\d+$', '', normalized)  # Remove version suffixes

        # Map aliases to canonical names
        model_aliases = {
            'claude-3-opus-20240229': 'claude-3-opus',
            'claude-3-sonnet-20240229': 'claude-3-sonnet',
            'claude-3-haiku-20240307': 'claude-3-haiku',
            'claude-3-5-sonnet-20240620': 'claude-3-5-sonnet',
        }

        return model_aliases.get(normalized, normalized)

    def validate_request(self, kwargs: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate Anthropic request parameters."""
        # Check required parameters
        if 'model' not in kwargs:
            return False, "Missing required parameter: model"

        # Validate messages format (for messages API)
        if 'messages' in kwargs:
            messages = kwargs['messages']
            if not isinstance(messages, list):
                return False, "Messages must be a list"

            if not messages:
                return False, "Messages list cannot be empty"

            for i, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    return False, f"Message {i} must be a dictionary"

                if 'role' not in msg:
                    return False, f"Message {i} missing required 'role' field"

                if msg['role'] not in ['user', 'assistant']:
                    return False, f"Message {i} has invalid role: {msg['role']} (Anthropic only supports 'user' and 'assistant')"

                if 'content' not in msg:
                    return False, f"Message {i} missing required 'content' field"

        # Validate legacy completion format
        elif 'prompt' in kwargs:
            prompt = kwargs['prompt']
            if not isinstance(prompt, str):
                return False, "Prompt must be a string for Anthropic completions"

        # Validate max_tokens (required for Anthropic)
        if 'max_tokens' not in kwargs:
            return False, "Missing required parameter: max_tokens"

        max_tokens = kwargs['max_tokens']
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            return False, "max_tokens must be a positive integer"

        return True, None

    def get_error_mapping(self) -> Dict[str, str]:
        """Map Anthropic errors to Brokle error types."""
        return {
            'AuthenticationError': 'AuthenticationError',
            'PermissionDeniedError': 'AuthenticationError',
            'RateLimitError': 'RateLimitError',
            'BadRequestError': 'ValidationError',
            'InvalidRequestError': 'ValidationError',
            'NotFoundError': 'ValidationError',
            'ConflictError': 'ValidationError',
            'UnprocessableEntityError': 'ValidationError',
            'InternalServerError': 'ProviderError',
            'APIConnectionError': 'ProviderError',
            'APITimeoutError': 'ProviderError',
            'APIError': 'ProviderError',
        }