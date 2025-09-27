"""
Anthropic Provider Implementation - Observability Only

Specific telemetry and instrumentation logic for Anthropic SDK.
Focused on request/response attribute extraction for observability.

Business logic (cost calculation, routing, caching) is handled by backend.
"""

import logging
import re
from typing import Any, Dict, List

from ..observability.attributes import (
    BrokleOtelSpanAttributes as BrokleInstrumentationAttributes,
)
from .base import BaseProvider

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseProvider):
    """Anthropic-specific provider implementation for observability."""

    def get_provider_name(self) -> str:
        """Return Anthropic provider identifier."""
        return "anthropic"

    def get_methods_to_instrument(self) -> List[Dict[str, Any]]:
        """Define Anthropic SDK methods to instrument."""
        return [
            # Messages API (primary API for Claude)
            {
                "path": "messages.create",
                "operation": "chat_completion",
                "async": False,
                "stream_support": True,
                "cost_tracked": True,
            },
            {
                "path": "messages.acreate",
                "operation": "chat_completion",
                "async": True,
                "stream_support": True,
                "cost_tracked": True,
            },
            # Completions API (legacy, but still supported)
            {
                "path": "completions.create",
                "operation": "completion",
                "async": False,
                "stream_support": True,
                "cost_tracked": True,
            },
            # Beta features (tool use, etc.)
            {
                "path": "beta.messages.create",
                "operation": "chat_completion_beta",
                "async": False,
                "stream_support": True,
                "cost_tracked": True,
            },
            # Batch API (if available)
            {
                "path": "messages.batch",
                "operation": "batch_completion",
                "async": False,
                "stream_support": False,
                "cost_tracked": True,
            },
        ]

    def extract_request_attributes(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Anthropic request attributes for telemetry."""
        attributes = {}

        # Model information
        if "model" in kwargs:
            model = kwargs["model"]
            attributes[BrokleInstrumentationAttributes.MODEL_NAME] = model
            attributes[BrokleInstrumentationAttributes.MODEL_NAME_NORMALIZED] = (
                self.normalize_model_name(model)
            )

        # Messages API specific
        if "messages" in kwargs:
            messages = kwargs["messages"]
            if isinstance(messages, list):
                attributes[BrokleInstrumentationAttributes.MESSAGE_COUNT] = len(
                    messages
                )

                # Extract message roles and types
                roles = [
                    msg.get("role", "unknown")
                    for msg in messages
                    if isinstance(msg, dict)
                ]
                attributes[BrokleInstrumentationAttributes.MESSAGE_ROLES] = ",".join(
                    roles
                )

        # System message (separate parameter in Anthropic)
        if "system" in kwargs:
            attributes[BrokleInstrumentationAttributes.SYSTEM_MESSAGE] = bool(
                kwargs["system"]
            )

        # Legacy completion specific
        if "prompt" in kwargs:
            prompt = kwargs["prompt"]
            if isinstance(prompt, str):
                attributes[BrokleInstrumentationAttributes.PROMPT_LENGTH] = len(prompt)

        # Common parameters
        common_params = {
            "max_tokens": BrokleInstrumentationAttributes.MAX_TOKENS,
            "temperature": BrokleInstrumentationAttributes.TEMPERATURE,
            "top_p": BrokleInstrumentationAttributes.TOP_P,
        }

        for param, attr in common_params.items():
            if param in kwargs and kwargs[param] is not None:
                attributes[attr] = kwargs[param]

        # Streaming
        if "stream" in kwargs:
            attributes[BrokleInstrumentationAttributes.STREAM_ENABLED] = kwargs[
                "stream"
            ]

        # Tool use (beta feature)
        if "tools" in kwargs:
            tools = kwargs["tools"]
            if isinstance(tools, list):
                attributes[BrokleInstrumentationAttributes.TOOL_COUNT] = len(tools)
                tool_types = [
                    t.get("type", "unknown") for t in tools if isinstance(t, dict)
                ]
                attributes[BrokleInstrumentationAttributes.TOOL_TYPES] = ",".join(
                    set(tool_types)
                )

        # Estimate input tokens
        input_tokens = self.estimate_input_tokens(kwargs)
        attributes[BrokleInstrumentationAttributes.INPUT_TOKENS] = input_tokens

        return attributes

    def extract_response_attributes(self, response: Any) -> Dict[str, Any]:
        """Extract Anthropic response attributes for telemetry."""
        attributes = {}

        try:
            # Handle usage information
            if hasattr(response, "usage") and response.usage:
                usage = response.usage

                # Token usage
                if hasattr(usage, "input_tokens"):
                    attributes[BrokleInstrumentationAttributes.INPUT_TOKENS] = (
                        usage.input_tokens
                    )
                if hasattr(usage, "output_tokens"):
                    attributes[BrokleInstrumentationAttributes.OUTPUT_TOKENS] = (
                        usage.output_tokens
                    )

                # Calculate total tokens
                input_tokens = getattr(usage, "input_tokens", 0)
                output_tokens = getattr(usage, "output_tokens", 0)
                attributes[BrokleInstrumentationAttributes.TOTAL_TOKENS] = (
                    input_tokens + output_tokens
                )

            # Model from response
            if hasattr(response, "model"):
                attributes[BrokleInstrumentationAttributes.MODEL_NAME_RESPONSE] = (
                    response.model
                )

            # Response content
            if hasattr(response, "content") and response.content:
                content_length = 0
                for item in response.content:
                    if hasattr(item, "text"):
                        content_length += len(item.text)

                attributes[BrokleInstrumentationAttributes.RESPONSE_CONTENT_LENGTH] = (
                    content_length
                )

            # Stop reason
            if hasattr(response, "stop_reason"):
                attributes[BrokleInstrumentationAttributes.STOP_REASON] = (
                    response.stop_reason
                )

            # Tool use in response
            if hasattr(response, "content") and response.content:
                tool_calls = []
                for item in response.content:
                    if hasattr(item, "type") and item.type == "tool_use":
                        if hasattr(item, "name"):
                            tool_calls.append(item.name)

                if tool_calls:
                    attributes[BrokleInstrumentationAttributes.TOOL_CALL_NAMES] = (
                        ",".join(tool_calls)
                    )

        except Exception as e:
            logger.warning(f"Failed to extract Anthropic response attributes: {e}")

        return attributes

    def estimate_input_tokens(self, kwargs: Dict[str, Any]) -> int:
        """Estimate input token count from Anthropic request parameters."""
        total_chars = 0

        # Handle messages format
        if "messages" in kwargs:
            messages = kwargs["messages"]
            if isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict):
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            total_chars += len(content)
                        elif isinstance(content, list):
                            # Handle multimodal content
                            for item in content:
                                if isinstance(item, dict):
                                    if item.get("type") == "text":
                                        total_chars += len(item.get("text", ""))
                                    elif item.get("type") == "image":
                                        # Images count as ~1000 tokens approximately
                                        total_chars += 4000  # 4000 chars ≈ 1000 tokens

        # Handle system message
        if "system" in kwargs:
            system = kwargs["system"]
            if isinstance(system, str):
                total_chars += len(system)

        # Handle legacy prompt format
        elif "prompt" in kwargs:
            prompt = kwargs["prompt"]
            if isinstance(prompt, str):
                total_chars += len(prompt)

        # Rough token estimation (4 characters ≈ 1 token)
        return max(1, total_chars // 4)

    def normalize_model_name(self, model: str) -> str:
        """Normalize Anthropic model names for consistent telemetry."""
        # Remove version suffixes
        normalized = re.sub(r"-\d{8}$", "", model)  # Remove date suffixes
        normalized = re.sub(r"-v\d+$", "", normalized)  # Remove version suffixes

        # Map aliases to canonical names
        model_aliases = {
            "claude-3-opus-20240229": "claude-3-opus",
            "claude-3-sonnet-20240229": "claude-3-sonnet",
            "claude-3-haiku-20240307": "claude-3-haiku",
            "claude-3-5-sonnet-20240620": "claude-3-5-sonnet",
        }

        return model_aliases.get(normalized, normalized)

    def get_error_mapping(self) -> Dict[str, str]:
        """Map Anthropic errors to Brokle error types."""
        return {
            "AuthenticationError": "AuthenticationError",
            "PermissionDeniedError": "AuthenticationError",
            "RateLimitError": "RateLimitError",
            "BadRequestError": "ValidationError",
            "InvalidRequestError": "ValidationError",
            "NotFoundError": "ValidationError",
            "ConflictError": "ValidationError",
            "UnprocessableEntityError": "ValidationError",
            "InternalServerError": "ProviderError",
            "APIConnectionError": "ProviderError",
            "APITimeoutError": "ProviderError",
            "APIError": "ProviderError",
        }
