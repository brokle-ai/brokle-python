"""
OpenAI auto-instrumentation for Brokle observability.

This module automatically instruments OpenAI API calls to capture
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
    import openai
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


class OpenAIInstrumentation:
    """Auto-instrumentation for OpenAI library."""

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
        """Check if OpenAI library is available."""
        return OPENAI_AVAILABLE

    def instrument(self) -> bool:
        """Instrument OpenAI library for automatic observability."""
        if not self.is_available():
            logger.warning("OpenAI library not available for instrumentation")
            return False

        if self._instrumented:
            logger.info("OpenAI already instrumented")
            return True

        try:
            # Instrument synchronous client
            self._instrument_sync_client()

            # Instrument asynchronous client
            self._instrument_async_client()

            self._instrumented = True
            logger.info("OpenAI instrumentation enabled successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to instrument OpenAI: {e}")
            return False

    def uninstrument(self) -> bool:
        """Remove OpenAI instrumentation."""
        if not self._instrumented:
            return True

        try:
            # Restore original methods
            for method_path, original_method in self._original_methods.items():
                self._set_method_by_path(method_path, original_method)

            self._original_methods.clear()
            self._instrumented = False
            logger.info("OpenAI instrumentation removed successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to uninstrument OpenAI: {e}")
            return False

    def _instrument_sync_client(self):
        """Instrument synchronous OpenAI client methods."""
        # Chat completions
        if hasattr(openai.resources.chat.completions, 'Completions'):
            completions_class = openai.resources.chat.completions.Completions
            if hasattr(completions_class, 'create'):
                original_create = completions_class.create
                self._original_methods['openai.resources.chat.completions.Completions.create'] = original_create
                completions_class.create = self._wrap_sync_method(
                    original_create,
                    method_name="chat_completions_create",
                    observation_type="llm"
                )

        # Text completions (legacy)
        if hasattr(openai.resources.completions, 'Completions'):
            completions_class = openai.resources.completions.Completions
            if hasattr(completions_class, 'create'):
                original_create = completions_class.create
                self._original_methods['openai.resources.completions.Completions.create'] = original_create
                completions_class.create = self._wrap_sync_method(
                    original_create,
                    method_name="completions_create",
                    observation_type="llm"
                )

        # Embeddings
        if hasattr(openai.resources.embeddings, 'Embeddings'):
            embeddings_class = openai.resources.embeddings.Embeddings
            if hasattr(embeddings_class, 'create'):
                original_create = embeddings_class.create
                self._original_methods['openai.resources.embeddings.Embeddings.create'] = original_create
                embeddings_class.create = self._wrap_sync_method(
                    original_create,
                    method_name="embeddings_create",
                    observation_type="llm"
                )

    def _instrument_async_client(self):
        """Instrument asynchronous OpenAI client methods."""
        # Async Chat completions
        if hasattr(openai.resources.chat.completions, 'AsyncCompletions'):
            async_completions_class = openai.resources.chat.completions.AsyncCompletions
            if hasattr(async_completions_class, 'create'):
                original_create = async_completions_class.create
                self._original_methods['openai.resources.chat.completions.AsyncCompletions.create'] = original_create
                async_completions_class.create = self._wrap_async_method(
                    original_create,
                    method_name="async_chat_completions_create",
                    observation_type="llm"
                )

        # Async Text completions (legacy)
        if hasattr(openai.resources.completions, 'AsyncCompletions'):
            async_completions_class = openai.resources.completions.AsyncCompletions
            if hasattr(async_completions_class, 'create'):
                original_create = async_completions_class.create
                self._original_methods['openai.resources.completions.AsyncCompletions.create'] = original_create
                async_completions_class.create = self._wrap_async_method(
                    original_create,
                    method_name="async_completions_create",
                    observation_type="llm"
                )

        # Async Embeddings
        if hasattr(openai.resources.embeddings, 'AsyncEmbeddings'):
            async_embeddings_class = openai.resources.embeddings.AsyncEmbeddings
            if hasattr(async_embeddings_class, 'create'):
                original_create = async_embeddings_class.create
                self._original_methods['openai.resources.embeddings.AsyncEmbeddings.create'] = original_create
                async_embeddings_class.create = self._wrap_async_method(
                    original_create,
                    method_name="async_embeddings_create",
                    observation_type="llm"
                )

    def _wrap_sync_method(self, original_method, method_name: str, observation_type: str):
        """Wrap synchronous method with observability."""

        @functools.wraps(original_method)
        def wrapper(*args, **kwargs):
            # Always execute original method first - never break user's code
            observation_id = None
            start_time = time.perf_counter()

            # Try to set up observability, but don't fail if it doesn't work
            try:
                if self.client is None:
                    # If no client available, just run original method
                    return original_method(*args, **kwargs)

                # Extract request data
                request_data = self._extract_request_data(args, kwargs, method_name)

                # Create trace if not exists
                trace_id = self._get_or_create_trace(
                    name=f"openai_{method_name}",
                    metadata={"library": "openai", "method": method_name}
                )

                # Create observation
                observation_data = {
                    "trace_id": trace_id,
                    "name": f"OpenAI {method_name}",
                    "observation_type": observation_type,
                    "model": request_data.get("model"),
                    "provider": "openai",
                    "input_data": request_data,
                    "start_time": datetime.utcnow()
                }

                observation = self.client.observability.create_observation_sync(**observation_data)
                observation_id = observation.id

            except Exception as e:
                logger.debug(f"Failed to setup observability for {method_name}: {e}")
                observation_id = None

            # Execute original method - this should never fail due to observability
            try:
                result = original_method(*args, **kwargs)

                # Try to complete observation, but don't fail if it doesn't work
                if observation_id:
                    try:
                        end_time = datetime.utcnow()
                        latency_ms = int((time.perf_counter() - start_time) * 1000)

                        # Extract response data safely
                        response_data = self._extract_response_data(result, method_name)

                        completion_data = {
                            "end_time": end_time,
                            "output_data": response_data,
                            "latency_ms": latency_ms,
                            "prompt_tokens": response_data.get("usage", {}).get("prompt_tokens"),
                            "completion_tokens": response_data.get("usage", {}).get("completion_tokens"),
                            "total_tokens": response_data.get("usage", {}).get("total_tokens"),
                            "total_cost": self._calculate_cost(request_data, response_data),
                            "status_message": "success"
                        }

                        self.client.observability.complete_observation_sync(
                            observation_id,
                            **completion_data
                        )
                    except Exception as e:
                        logger.debug(f"Failed to complete observation: {e}")

                return result

            except Exception as e:
                # Try to record the error, but don't fail if observation fails
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
                    except Exception:
                        # Silent failure for observability - never break user's code
                        pass

                # Always re-raise the original exception
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
                name=f"openai_{method_name}",
                metadata={"library": "openai", "method": method_name}
            )

            # Create observation
            observation_data = {
                "trace_id": trace_id,
                "name": f"OpenAI {method_name}",
                "observation_type": observation_type,
                "model": request_data.get("model"),
                "provider": "openai",
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
                            "prompt_tokens": response_data.get("usage", {}).get("prompt_tokens"),
                            "completion_tokens": response_data.get("usage", {}).get("completion_tokens"),
                            "total_tokens": response_data.get("usage", {}).get("total_tokens"),
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
            # For OpenAI, the first argument is usually 'self', so we start from kwargs
            request_data = kwargs.copy()

            # Handle positional arguments for some cases
            if args and len(args) > 1:
                # Skip 'self' argument
                if method_name in ["chat_completions_create", "async_chat_completions_create"]:
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

    def _extract_response_data(self, result, method_name: str) -> Dict[str, Any]:
        """Extract response data from method result."""
        try:
            if hasattr(result, 'model_dump'):
                # Pydantic model
                return result.model_dump()
            elif hasattr(result, 'to_dict'):
                # Some OpenAI objects have to_dict method
                return result.to_dict()
            elif hasattr(result, '__dict__'):
                # Generic object with attributes
                return self._serialize_safely(result.__dict__)
            else:
                # Try to serialize directly
                return self._serialize_safely(result)

        except Exception as e:
            logger.warning(f"Failed to extract response data: {e}")
            return {"error": f"Failed to serialize response: {e}"}

    def _calculate_cost(self, request_data: Dict[str, Any], response_data: Dict[str, Any]) -> Optional[float]:
        """Calculate approximate cost for OpenAI API call."""
        try:
            model = request_data.get("model", "")
            usage = response_data.get("usage", {})

            if not usage:
                return None

            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            # OpenAI pricing (approximate, as of 2024)
            pricing_map = {
                "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
                "gpt-4-32k": {"input": 0.06, "output": 0.12},
                "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
                "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
                "text-davinci-003": {"input": 0.02, "output": 0.02},
                "text-embedding-ada-002": {"input": 0.0001, "output": 0.0001},
            }

            # Find matching pricing
            pricing = None
            for model_key, model_pricing in pricing_map.items():
                if model_key in model.lower():
                    pricing = model_pricing
                    break

            if not pricing:
                return None

            input_cost = (prompt_tokens / 1000) * pricing["input"]
            output_cost = (completion_tokens / 1000) * pricing["output"]

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
        obj = openai

        for part in parts[1:-1]:  # Skip 'openai' and method name
            obj = getattr(obj, part)

        setattr(obj, parts[-1], method)


# Global instance
_openai_instrumentation = OpenAIInstrumentation()


def instrument_openai() -> bool:
    """Instrument OpenAI library for automatic observability."""
    return _openai_instrumentation.instrument()


def uninstrument_openai() -> bool:
    """Remove OpenAI instrumentation."""
    return _openai_instrumentation.uninstrument()


def is_openai_instrumented() -> bool:
    """Check if OpenAI is currently instrumented."""
    return _openai_instrumentation._instrumented