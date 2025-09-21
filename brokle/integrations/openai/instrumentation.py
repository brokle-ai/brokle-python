"""
OpenAI Instrumentation Implementation for Brokle Platform.

This module implements the BaseInstrumentation interface specifically for
OpenAI's SDK, providing comprehensive auto-instrumentation and manual
wrapper capabilities.

Features:
- Supports OpenAI SDK v1.0+
- Handles both sync and async methods
- Covers chat completions, text completions, and embeddings
- Includes cost calculation and usage tracking
- Provides defensive error handling
"""

import logging
from typing import Any, Dict, List, Optional
import re

from .._base import BaseInstrumentation, InstrumentationConfig

logger = logging.getLogger(__name__)


class OpenAIInstrumentation(BaseInstrumentation):
    """OpenAI-specific instrumentation implementation."""

    def __init__(self):
        """Initialize OpenAI instrumentation."""
        super().__init__("openai")

    def get_method_configs(self) -> List[InstrumentationConfig]:
        """Get OpenAI methods to instrument."""
        return [
            # Chat completions
            InstrumentationConfig(
                module="openai.resources.chat.completions",
                object="Completions.create",
                name="chat_completions_create",
                operation_type="llm",
                is_async=False,
                supports_streaming=True
            ),
            InstrumentationConfig(
                module="openai.resources.chat.completions",
                object="AsyncCompletions.create",
                name="async_chat_completions_create",
                operation_type="llm",
                is_async=True,
                supports_streaming=True
            ),
            # Text completions
            InstrumentationConfig(
                module="openai.resources.completions",
                object="Completions.create",
                name="completions_create",
                operation_type="llm",
                is_async=False,
                supports_streaming=True
            ),
            InstrumentationConfig(
                module="openai.resources.completions",
                object="AsyncCompletions.create",
                name="async_completions_create",
                operation_type="llm",
                is_async=True,
                supports_streaming=True
            ),
            # Embeddings
            InstrumentationConfig(
                module="openai.resources.embeddings",
                object="Embeddings.create",
                name="embeddings_create",
                operation_type="embedding",
                is_async=False,
                supports_streaming=False
            ),
            InstrumentationConfig(
                module="openai.resources.embeddings",
                object="AsyncEmbeddings.create",
                name="async_embeddings_create",
                operation_type="embedding",
                is_async=True,
                supports_streaming=False
            )
        ]

    def extract_request_metadata(self, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Extract safe request data from OpenAI method arguments."""
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
                'top_p', 'frequency_penalty', 'presence_penalty', 'stop',
                'stream', 'n', 'logprobs', 'echo', 'suffix', 'user',
                'input', 'encoding_format', 'dimensions'
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

            return filtered_data

        except Exception as e:
            logger.debug(f"Failed to extract OpenAI request data: {e}")
            return {"extraction_error": "Failed to extract request data"}

    def extract_response_metadata(self, result: Any) -> Dict[str, Any]:
        """Extract safe response data from OpenAI method result."""
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
                'id', 'object', 'created', 'model', 'choices', 'usage',
                'data', 'system_fingerprint'
            }

            filtered_data = {k: v for k, v in data.items() if k in safe_fields}

            # Sanitize choices if present (limit content length)
            if 'choices' in filtered_data and isinstance(filtered_data['choices'], list):
                sanitized_choices = []
                for choice in filtered_data['choices'][:5]:  # Limit to first 5 choices
                    if hasattr(choice, '__dict__'):
                        choice_dict = choice.__dict__.copy()
                    elif isinstance(choice, dict):
                        choice_dict = choice.copy()
                    else:
                        continue

                    # Limit message content length
                    if 'message' in choice_dict and hasattr(choice_dict['message'], 'content'):
                        content = choice_dict['message'].content
                        if content and len(str(content)) > 2000:
                            choice_dict['message'].content = str(content)[:2000] + "..."

                    sanitized_choices.append(choice_dict)
                filtered_data['choices'] = sanitized_choices

            return filtered_data

        except Exception as e:
            logger.debug(f"Failed to extract OpenAI response data: {e}")
            return {"extraction_error": "Failed to extract response data"}

    def calculate_cost(self, model: str, usage: Dict[str, Any]) -> Optional[float]:
        """Calculate approximate OpenAI cost based on model and usage."""
        if not model or not usage:
            return None

        try:
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            # Updated OpenAI pricing (per 1K tokens) - approximate rates as of 2024
            pricing_map = {
                "gpt-4": {"input": 0.03, "output": 0.06},
                "gpt-4-32k": {"input": 0.06, "output": 0.12},
                "gpt-4-turbo": {"input": 0.01, "output": 0.03},
                "gpt-4o": {"input": 0.005, "output": 0.015},
                "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
                "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
                "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
                "text-embedding-ada-002": {"input": 0.0001, "output": 0.0001},
                "text-embedding-3-small": {"input": 0.00002, "output": 0.00002},
                "text-embedding-3-large": {"input": 0.00013, "output": 0.00013},
            }

            # Find matching pricing
            pricing = None
            model_lower = model.lower()
            for model_key, model_pricing in pricing_map.items():
                if model_key in model_lower:
                    pricing = model_pricing
                    break

            if not pricing:
                # Default fallback pricing for unknown models
                if "gpt-4" in model_lower:
                    pricing = {"input": 0.01, "output": 0.03}
                elif "gpt-3.5" in model_lower:
                    pricing = {"input": 0.001, "output": 0.002}
                else:
                    pricing = {"input": 0.01, "output": 0.02}

            input_cost = (prompt_tokens / 1000) * pricing["input"]
            output_cost = (completion_tokens / 1000) * pricing["output"]

            return round(input_cost + output_cost, 6)

        except Exception as e:
            logger.debug(f"Failed to calculate OpenAI cost: {e}")
            return None

    def classify_error(self, exc: Exception) -> str:
        """Classify OpenAI-specific errors."""
        exc_name = type(exc).__name__.lower()
        exc_message = str(exc).lower()

        # OpenAI-specific error patterns
        if "ratelimiterror" in exc_name or "rate limit" in exc_message:
            return "rate_limit"
        elif "authenticationerror" in exc_name or "invalid api key" in exc_message:
            return "auth_failure"
        elif "permissionerror" in exc_name or "permission" in exc_message:
            return "permission_denied"
        elif "notfounderror" in exc_name or "model not found" in exc_message:
            return "model_unavailable"
        elif "timeouterror" in exc_name or "timeout" in exc_message:
            return "timeout"
        elif "connectionerror" in exc_name or "connection" in exc_message:
            return "network_error"
        elif "badrequest" in exc_name or "invalid request" in exc_message:
            return "invalid_request"
        else:
            return super().classify_error(exc)

    def should_suppress_error(self, exc: Exception) -> bool:
        """Determine if OpenAI error should be suppressed in instrumentation."""
        error_type = self.classify_error(exc)

        # Suppress temporary errors that shouldn't break instrumentation
        return error_type in ["network_error", "timeout", "rate_limit"]

    def get_supported_versions(self) -> List[str]:
        """Get supported OpenAI SDK versions."""
        return ["1.0.0", "1.12.0", "1.30.0", "1.x"]

    def validate_environment(self) -> Dict[str, Any]:
        """Validate OpenAI environment."""
        issues = []
        version = "unknown"
        supported = True

        try:
            import openai
            version = getattr(openai, "__version__", "unknown")

            # Check version compatibility
            if version != "unknown":
                # Extract major version
                major_version = int(version.split(".")[0])
                if major_version < 1:
                    issues.append(f"OpenAI SDK v{version} not supported. Please upgrade to v1.0+")
                    supported = False

        except ImportError:
            issues.append("OpenAI SDK not installed")
            supported = False
        except Exception as e:
            issues.append(f"Failed to validate OpenAI environment: {e}")
            supported = False

        return {
            "provider": self.provider_name,
            "supported": supported,
            "issues": issues,
            "version": version,
            "library_available": supported
        }