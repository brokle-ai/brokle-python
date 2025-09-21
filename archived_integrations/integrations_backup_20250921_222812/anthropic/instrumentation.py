"""
Anthropic Instrumentation Implementation for Brokle Platform.

This module implements the BaseInstrumentation interface specifically for
Anthropic's SDK, providing comprehensive auto-instrumentation and manual
wrapper capabilities.

Features:
- Supports Anthropic SDK v0.25.0+
- Handles both sync and async methods
- Covers messages API and legacy completions
- Includes cost calculation and usage tracking
- Provides defensive error handling
"""

import logging
from typing import Any, Dict, List, Optional

from .._base import BaseInstrumentation, InstrumentationConfig

logger = logging.getLogger(__name__)


class AnthropicInstrumentation(BaseInstrumentation):
    """Anthropic-specific instrumentation implementation."""

    def __init__(self):
        """Initialize Anthropic instrumentation."""
        super().__init__("anthropic")

    def get_method_configs(self) -> List[InstrumentationConfig]:
        """Get Anthropic methods to instrument."""
        return [
            # Messages API (current)
            InstrumentationConfig(
                module="anthropic.resources.messages",
                object="Messages.create",
                name="anthropic_messages_create",
                operation_type="llm",
                is_async=False,
                supports_streaming=True
            ),
            InstrumentationConfig(
                module="anthropic.resources.messages",
                object="AsyncMessages.create",
                name="async_anthropic_messages_create",
                operation_type="llm",
                is_async=True,
                supports_streaming=True
            ),
            # Completions API (legacy, might still be available)
            InstrumentationConfig(
                module="anthropic.resources.completions",
                object="Completions.create",
                name="anthropic_completions_create",
                operation_type="llm",
                is_async=False,
                supports_streaming=True
            ),
            InstrumentationConfig(
                module="anthropic.resources.completions",
                object="AsyncCompletions.create",
                name="async_anthropic_completions_create",
                operation_type="llm",
                is_async=True,
                supports_streaming=True
            )
        ]

    def extract_request_metadata(self, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Extract safe request data from Anthropic method arguments."""
        try:
            # Skip 'self' parameter, extract key fields
            data = kwargs.copy()

            # Add positional args if present (wrapt already stripped 'self')
            if len(args) > 0:
                if 'model' not in data and len(args) > 0:
                    data['model'] = args[0]
                if 'messages' not in data and len(args) > 1:
                    data['messages'] = args[1]
                if 'prompt' not in data and len(args) > 1:
                    data['prompt'] = args[1]

            # Filter to safe, relevant fields only
            safe_fields = {
                'model', 'messages', 'prompt', 'temperature', 'max_tokens',
                'top_p', 'top_k', 'stop_sequences', 'stream', 'system'
            }

            filtered_data = {k: v for k, v in data.items() if k in safe_fields}

            # Sanitize messages if present (remove any sensitive content)
            if 'messages' in filtered_data and isinstance(filtered_data['messages'], list):
                sanitized_messages = []
                for msg in filtered_data['messages'][:10]:  # Limit to first 10 messages
                    if isinstance(msg, dict) and 'content' in msg:
                        sanitized_msg = {
                            'role': msg.get('role', 'unknown'),
                            'content': str(msg['content'])[:1000] if msg['content'] else None  # Limit content length
                        }
                        sanitized_messages.append(sanitized_msg)
                filtered_data['messages'] = sanitized_messages

            # Sanitize system prompt if present
            if 'system' in filtered_data and filtered_data['system']:
                filtered_data['system'] = str(filtered_data['system'])[:500]

            return filtered_data

        except Exception as e:
            logger.debug(f"Failed to extract Anthropic request data: {e}")
            return {"extraction_error": "Failed to extract request data"}

    def extract_response_metadata(self, result: Any) -> Dict[str, Any]:
        """Extract safe response data from Anthropic method result."""
        try:
            if hasattr(result, 'model_dump'):
                # Pydantic model
                data = result.model_dump()
            elif hasattr(result, '__dict__'):
                # Generic object with attributes
                data = result.__dict__.copy()
            else:
                return {"type": str(type(result).__name__)}

            # Filter to safe fields
            safe_fields = {
                'id', 'type', 'role', 'content', 'model', 'stop_reason',
                'stop_sequence', 'usage'
            }

            filtered_data = {k: v for k, v in data.items() if k in safe_fields}

            # Sanitize content if present (limit content length)
            if 'content' in filtered_data and isinstance(filtered_data['content'], list):
                sanitized_content = []
                for item in filtered_data['content'][:5]:  # Limit to first 5 content items
                    if hasattr(item, '__dict__'):
                        item_dict = item.__dict__.copy()
                    elif isinstance(item, dict):
                        item_dict = item.copy()
                    else:
                        continue

                    # Limit text content length
                    if 'text' in item_dict and item_dict['text']:
                        text = item_dict['text']
                        if len(str(text)) > 2000:
                            item_dict['text'] = str(text)[:2000] + "..."

                    sanitized_content.append(item_dict)
                filtered_data['content'] = sanitized_content

            return filtered_data

        except Exception as e:
            logger.debug(f"Failed to extract Anthropic response data: {e}")
            return {"extraction_error": "Failed to extract response data"}

    def calculate_cost(self, model: str, usage: Dict[str, Any]) -> Optional[float]:
        """Calculate approximate Anthropic cost based on model and usage."""
        if not model or not usage:
            return None

        try:
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

            # Anthropic pricing (per 1M tokens) - approximate rates as of 2024
            pricing_map = {
                "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
                "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00},
                "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
                "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
                "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
                "claude-2.1": {"input": 8.00, "output": 24.00},
                "claude-2.0": {"input": 8.00, "output": 24.00},
                "claude-instant-1.2": {"input": 0.80, "output": 2.40},
            }

            # Find matching pricing
            pricing = None
            model_lower = model.lower()
            for model_key, model_pricing in pricing_map.items():
                if model_key in model_lower:
                    pricing = model_pricing
                    break

            if not pricing:
                # Default fallback pricing based on model tier
                if "opus" in model_lower:
                    pricing = {"input": 15.00, "output": 75.00}
                elif "sonnet" in model_lower:
                    pricing = {"input": 3.00, "output": 15.00}
                elif "haiku" in model_lower:
                    pricing = {"input": 0.25, "output": 1.25}
                elif "instant" in model_lower:
                    pricing = {"input": 0.80, "output": 2.40}
                else:
                    # Default to Sonnet pricing
                    pricing = {"input": 3.00, "output": 15.00}

            # Calculate cost (pricing is per 1M tokens)
            input_cost = (input_tokens / 1_000_000) * pricing["input"]
            output_cost = (output_tokens / 1_000_000) * pricing["output"]

            return round(input_cost + output_cost, 6)

        except Exception as e:
            logger.debug(f"Failed to calculate Anthropic cost: {e}")
            return None

    def classify_error(self, exc: Exception) -> str:
        """Classify Anthropic-specific errors."""
        exc_name = type(exc).__name__.lower()
        exc_message = str(exc).lower()

        # Anthropic-specific error patterns
        if "ratelimiterror" in exc_name or "rate limit" in exc_message:
            return "rate_limit"
        elif "authenticationerror" in exc_name or "authentication" in exc_message:
            return "auth_failure"
        elif "permissionerror" in exc_name or "permission" in exc_message:
            return "permission_denied"
        elif "notfounderror" in exc_name or "not found" in exc_message:
            return "model_unavailable"
        elif "timeouterror" in exc_name or "timeout" in exc_message:
            return "timeout"
        elif "connectionerror" in exc_name or "connection" in exc_message:
            return "network_error"
        elif "badrequest" in exc_name or "invalid" in exc_message:
            return "invalid_request"
        elif "overloaded" in exc_message or "capacity" in exc_message:
            return "service_overloaded"
        else:
            return super().classify_error(exc)

    def should_suppress_error(self, exc: Exception) -> bool:
        """Determine if Anthropic error should be suppressed in instrumentation."""
        error_type = self.classify_error(exc)

        # Suppress temporary errors that shouldn't break instrumentation
        return error_type in ["network_error", "timeout", "rate_limit", "service_overloaded"]

    def get_supported_versions(self) -> List[str]:
        """Get supported Anthropic SDK versions."""
        return ["0.25.0", "0.30.0", "0.x"]

    def validate_environment(self) -> Dict[str, Any]:
        """Validate Anthropic environment."""
        issues = []
        version = "unknown"
        supported = True

        try:
            import anthropic
            version = getattr(anthropic, "__version__", "unknown")

            # Check version compatibility
            if version != "unknown":
                # Extract version components
                version_parts = version.split(".")
                if len(version_parts) >= 2:
                    major_version = int(version_parts[0])
                    minor_version = int(version_parts[1])

                    # Check if version is supported (0.25.0+)
                    if major_version == 0 and minor_version < 25:
                        issues.append(f"Anthropic SDK v{version} not supported. Please upgrade to v0.25.0+")
                        supported = False

        except ImportError:
            issues.append("Anthropic SDK not installed")
            supported = False
        except Exception as e:
            issues.append(f"Failed to validate Anthropic environment: {e}")
            supported = False

        return {
            "provider": self.provider_name,
            "supported": supported,
            "issues": issues,
            "version": version,
            "library_available": supported
        }