"""
Anthropic auto-instrumentation for Brokle observability.

This module automatically instruments Anthropic API calls to capture
comprehensive observability data including costs, tokens, and quality metrics.
"""

import asyncio
import functools
import json
import logging
import time
from typing import Any, Dict, Optional, Union, List
from datetime import datetime

try:
    import anthropic
    from anthropic import Anthropic, AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logger = logging.getLogger(__name__)


class AnthropicInstrumentation:
    """Auto-instrumentation for Anthropic library."""

    def __init__(self):
        self._config = None
        self._client = None
        self._original_methods = {}
        self._instrumented = False

    @property
    def config(self):
        """Get or create Brokle config with lazy loading."""
        if self._config is None:
            try:
                from ..config import get_config
                self._config = get_config()
            except Exception as e:
                logger.warning(f"Failed to load Brokle config: {e}")
                self._config = None
        return self._config

    @property
    def client(self):
        """Get or create Brokle client for observability."""
        if self._client is None:
            try:
                from ..client import get_client
                self._client = get_client()
            except Exception as e:
                logger.warning(f"Failed to initialize Brokle client: {e}")
                self._client = None
        return self._client

    def is_available(self) -> bool:
        """Check if Anthropic library is available."""
        return ANTHROPIC_AVAILABLE

    def instrument(self) -> bool:
        """Instrument Anthropic library for automatic observability."""
        if not self.is_available():
            logger.warning("Anthropic library not available for instrumentation")
            return False

        if self._instrumented:
            logger.info("Anthropic already instrumented")
            return True

        try:
            # Instrument synchronous client
            self._instrument_sync_client()

            # Instrument asynchronous client
            self._instrument_async_client()

            self._instrumented = True
            logger.info("Anthropic instrumentation enabled successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to instrument Anthropic: {e}")
            return False

    def uninstrument(self) -> bool:
        """Remove Anthropic instrumentation."""
        if not self._instrumented:
            return True

        try:
            # Restore original methods
            for method_path, original_method in self._original_methods.items():
                self._set_method_by_path(method_path, original_method)

            self._original_methods.clear()
            self._instrumented = False
            logger.info("Anthropic instrumentation removed successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to uninstrument Anthropic: {e}")
            return False

    def _instrument_sync_client(self):
        """Instrument synchronous Anthropic client methods."""
        # Messages (Chat) completions
        if hasattr(anthropic.resources.messages, 'Messages'):
            messages_class = anthropic.resources.messages.Messages
            if hasattr(messages_class, 'create'):
                original_create = messages_class.create
                self._original_methods['anthropic.resources.messages.Messages.create'] = original_create
                messages_class.create = self._wrap_sync_method(
                    original_create,
                    method_name="messages_create",
                    observation_type="llm"
                )

        # Text completions (legacy)
        if hasattr(anthropic.resources.completions, 'Completions'):
            completions_class = anthropic.resources.completions.Completions
            if hasattr(completions_class, 'create'):
                original_create = completions_class.create
                self._original_methods['anthropic.resources.completions.Completions.create'] = original_create
                completions_class.create = self._wrap_sync_method(
                    original_create,
                    method_name="completions_create",
                    observation_type="llm"
                )

    def _instrument_async_client(self):
        """Instrument asynchronous Anthropic client methods."""
        # Async Messages (Chat) completions
        if hasattr(anthropic.resources.messages, 'AsyncMessages'):
            async_messages_class = anthropic.resources.messages.AsyncMessages
            if hasattr(async_messages_class, 'create'):
                original_create = async_messages_class.create
                self._original_methods['anthropic.resources.messages.AsyncMessages.create'] = original_create
                async_messages_class.create = self._wrap_async_method(
                    original_create,
                    method_name="async_messages_create",
                    observation_type="llm"
                )

        # Async Text completions (legacy)
        if hasattr(anthropic.resources.completions, 'AsyncCompletions'):
            async_completions_class = anthropic.resources.completions.AsyncCompletions
            if hasattr(async_completions_class, 'create'):
                original_create = async_completions_class.create
                self._original_methods['anthropic.resources.completions.AsyncCompletions.create'] = original_create
                async_completions_class.create = self._wrap_async_method(
                    original_create,
                    method_name="async_completions_create",
                    observation_type="llm"
                )

    def _wrap_sync_method(self, original_method, method_name: str, observation_type: str):
        """Wrap synchronous method with observability."""

        @functools.wraps(original_method)
        def wrapper(*args, **kwargs):
            # Extract request data
            request_data = self._extract_request_data(args, kwargs, method_name)

            # Create trace if not exists
            trace_id = self._get_or_create_trace(
                name=f"anthropic_{method_name}",
                metadata={"library": "anthropic", "method": method_name}
            )

            # Create observation
            observation_data = {
                "trace_id": trace_id,
                "name": f"Anthropic {method_name}",
                "observation_type": observation_type,
                "model": request_data.get("model"),
                "provider": "anthropic",
                "input_data": request_data,
                "start_time": datetime.utcnow()
            }

            try:
                observation = self.client.observability.create_observation_sync(**observation_data)
                observation_id = observation.id
            except Exception as e:
                logger.error(f"Failed to create observation: {e}")
                observation_id = None

            # Execute original method
            start_time = time.perf_counter()
            try:
                result = original_method(*args, **kwargs)

                # Extract response data
                response_data = self._extract_response_data(result, method_name)

                # Complete observation
                if observation_id:
                    try:
                        end_time = datetime.utcnow()
                        latency_ms = int((time.perf_counter() - start_time) * 1000)

                        completion_data = {
                            "end_time": end_time,
                            "output_data": response_data,
                            "latency_ms": latency_ms,
                            "prompt_tokens": response_data.get("usage", {}).get("input_tokens"),
                            "completion_tokens": response_data.get("usage", {}).get("output_tokens"),
                            "total_tokens": self._calculate_total_tokens(response_data),
                            "total_cost": self._calculate_cost(request_data, response_data),
                            "status_message": "success"
                        }

                        self.client.observability.complete_observation_sync(
                            observation_id,
                            **completion_data
                        )
                    except Exception as e:
                        logger.error(f"Failed to complete observation: {e}")

                return result

            except Exception as e:
                # Handle error in observation
                if observation_id:
                    try:
                        end_time = datetime.utcnow()
                        latency_ms = int((time.perf_counter() - start_time) * 1000)

                        self.client.observability.complete_observation_sync(
                            observation_id,
                            end_time=end_time,
                            latency_ms=latency_ms,
                            status_message=f"error: {str(e)}"
                        )
                    except Exception as complete_error:
                        logger.error(f"Failed to complete error observation: {complete_error}")

                raise

        return wrapper

    def _wrap_async_method(self, original_method, method_name: str, observation_type: str):
        """Wrap asynchronous method with observability."""

        @functools.wraps(original_method)
        async def async_wrapper(*args, **kwargs):
            # Extract request data
            request_data = self._extract_request_data(args, kwargs, method_name)

            # Create trace if not exists
            trace_id = await self._get_or_create_trace_async(
                name=f"anthropic_{method_name}",
                metadata={"library": "anthropic", "method": method_name}
            )

            # Create observation
            observation_data = {
                "trace_id": trace_id,
                "name": f"Anthropic {method_name}",
                "observation_type": observation_type,
                "model": request_data.get("model"),
                "provider": "anthropic",
                "input_data": request_data,
                "start_time": datetime.utcnow()
            }

            try:
                observation = await self.client.observability.create_observation(**observation_data)
                observation_id = observation.id
            except Exception as e:
                logger.error(f"Failed to create observation: {e}")
                observation_id = None

            # Execute original method
            start_time = time.perf_counter()
            try:
                result = await original_method(*args, **kwargs)

                # Extract response data
                response_data = self._extract_response_data(result, method_name)

                # Complete observation
                if observation_id:
                    try:
                        end_time = datetime.utcnow()
                        latency_ms = int((time.perf_counter() - start_time) * 1000)

                        completion_data = {
                            "end_time": end_time,
                            "output_data": response_data,
                            "latency_ms": latency_ms,
                            "prompt_tokens": response_data.get("usage", {}).get("input_tokens"),
                            "completion_tokens": response_data.get("usage", {}).get("output_tokens"),
                            "total_tokens": self._calculate_total_tokens(response_data),
                            "total_cost": self._calculate_cost(request_data, response_data),
                            "status_message": "success"
                        }

                        await self.client.observability.complete_observation(
                            observation_id,
                            **completion_data
                        )
                    except Exception as e:
                        logger.error(f"Failed to complete observation: {e}")

                return result

            except Exception as e:
                # Handle error in observation
                if observation_id:
                    try:
                        end_time = datetime.utcnow()
                        latency_ms = int((time.perf_counter() - start_time) * 1000)

                        await self.client.observability.complete_observation(
                            observation_id,
                            end_time=end_time,
                            latency_ms=latency_ms,
                            status_message=f"error: {str(e)}"
                        )
                    except Exception as complete_error:
                        logger.error(f"Failed to complete error observation: {complete_error}")

                raise

        return async_wrapper

    def _extract_request_data(self, args, kwargs, method_name: str) -> Dict[str, Any]:
        """Extract request data from method arguments."""
        try:
            # For Anthropic, the first argument is usually 'self', so we start from kwargs
            request_data = kwargs.copy()

            # Handle positional arguments for some cases
            if args and len(args) > 1:
                # Skip 'self' argument
                if method_name in ["messages_create", "async_messages_create"]:
                    if len(args) > 1:
                        request_data.update({
                            "model": args[1] if len(args) > 1 else kwargs.get("model"),
                            "messages": args[2] if len(args) > 2 else kwargs.get("messages")
                        })

            # Clean sensitive data with safe serialization
            cleaned_data = self._serialize_safely(request_data)

            return cleaned_data

        except Exception as e:
            logger.warning(f"Failed to extract request data: {e}")
            return {"error": f"Failed to serialize request: {e}"}

    def _extract_response_data(self, result, method_name: str) -> Dict[str, Any]:
        """Extract response data from method result."""
        try:
            if hasattr(result, 'model_dump'):
                # Pydantic model
                return result.model_dump()
            elif hasattr(result, 'to_dict'):
                # Some Anthropic objects have to_dict method
                return result.to_dict()
            elif hasattr(result, '__dict__'):
                # Generic object with attributes
                response_dict = result.__dict__.copy()

                # Handle Anthropic-specific response structure
                if hasattr(result, 'content') and hasattr(result, 'usage'):
                    response_dict['content'] = getattr(result, 'content', [])
                    response_dict['usage'] = getattr(result, 'usage', {})

                return self._serialize_safely(response_dict)
            else:
                # Try to serialize directly
                return self._serialize_safely(result)

        except Exception as e:
            logger.warning(f"Failed to extract response data: {e}")
            return {"error": f"Failed to serialize response: {e}"}

    def _serialize_safely(self, data: Any) -> Dict[str, Any]:
        """Safely serialize data with proper error handling."""
        try:
            from ..core.serialization import serialize
            return serialize(data)
        except ImportError:
            # Fallback serialization if core.serialization not available
            return self._basic_serialize(data)
        except Exception as e:
            logger.warning(f"Serialization failed: {e}")
            return {"serialization_error": str(e)}

    def _basic_serialize(self, data: Any) -> Dict[str, Any]:
        """Basic serialization fallback."""
        try:
            if isinstance(data, dict):
                return {k: self._basic_serialize(v) for k, v in data.items()
                       if isinstance(v, (str, int, float, bool, list, dict)) or v is None}
            elif isinstance(data, (list, tuple)):
                return [self._basic_serialize(item) for item in data]
            elif isinstance(data, (str, int, float, bool)) or data is None:
                return data
            else:
                return str(data)
        except Exception:
            return {"type": str(type(data).__name__)}

    def _calculate_total_tokens(self, response_data: Dict[str, Any]) -> Optional[int]:
        """Calculate total tokens for Anthropic response."""
        try:
            usage = response_data.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            return input_tokens + output_tokens if input_tokens or output_tokens else None
        except:
            return None

    def _calculate_cost(self, request_data: Dict[str, Any], response_data: Dict[str, Any]) -> Optional[float]:
        """Calculate approximate cost for Anthropic API call."""
        try:
            model = request_data.get("model", "")
            usage = response_data.get("usage", {})

            if not usage:
                return None

            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

            # Anthropic pricing (approximate, as of 2024)
            pricing_map = {
                "claude-3-opus": {"input": 0.015, "output": 0.075},  # per 1K tokens
                "claude-3-sonnet": {"input": 0.003, "output": 0.015},
                "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
                "claude-2.1": {"input": 0.008, "output": 0.024},
                "claude-2.0": {"input": 0.008, "output": 0.024},
                "claude-instant-1.2": {"input": 0.0008, "output": 0.0024},
            }

            # Find matching pricing
            pricing = None
            for model_key, model_pricing in pricing_map.items():
                if model_key in model.lower():
                    pricing = model_pricing
                    break

            if not pricing:
                # Default to Claude-3-Sonnet pricing if model not found
                pricing = {"input": 0.003, "output": 0.015}

            input_cost = (input_tokens / 1000) * pricing["input"]
            output_cost = (output_tokens / 1000) * pricing["output"]

            return round(input_cost + output_cost, 6)

        except Exception as e:
            logger.warning(f"Failed to calculate cost: {e}")
            return None

    def _get_or_create_trace(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Get or create a trace synchronously."""
        try:
            trace = self.client.observability.create_trace_sync(
                name=name,
                metadata=metadata or {}
            )
            return trace.id
        except Exception as e:
            logger.error(f"Failed to create trace: {e}")
            # Return a fallback trace ID
            import uuid
            return str(uuid.uuid4())

    async def _get_or_create_trace_async(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Get or create a trace asynchronously."""
        try:
            trace = await self.client.observability.create_trace(
                name=name,
                metadata=metadata or {}
            )
            return trace.id
        except Exception as e:
            logger.error(f"Failed to create trace: {e}")
            # Return a fallback trace ID
            import uuid
            return str(uuid.uuid4())

    def _set_method_by_path(self, method_path: str, method):
        """Set a method by its dotted path."""
        parts = method_path.split('.')
        obj = anthropic

        for part in parts[1:-1]:  # Skip 'anthropic' and method name
            obj = getattr(obj, part)

        setattr(obj, parts[-1], method)


# Global instance
_anthropic_instrumentation = AnthropicInstrumentation()


def instrument_anthropic() -> bool:
    """Instrument Anthropic library for automatic observability."""
    return _anthropic_instrumentation.instrument()


def uninstrument_anthropic() -> bool:
    """Remove Anthropic instrumentation."""
    return _anthropic_instrumentation.uninstrument()


def is_anthropic_instrumented() -> bool:
    """Check if Anthropic is currently instrumented."""
    return _anthropic_instrumentation._instrumented