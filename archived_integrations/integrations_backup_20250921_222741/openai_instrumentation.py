"""
OpenAI auto-instrumentation for Brokle observability.

This module automatically instruments OpenAI API calls to capture
comprehensive observability data including costs, tokens, and quality metrics.
Features comprehensive error handling, circuit breakers, and graceful degradation.
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

from .error_handlers import (
    InstrumentationError,
    LibraryNotAvailableError,
    ObservabilityError,
    ConfigurationError,
    ErrorSeverity,
    safe_operation,
    safe_async_operation,
    instrumentation_context,
    validate_config,
    get_error_handler
)

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
                client = self.client
                if client is not None:
                    self._config = client.config
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

    def _is_client_available(self) -> bool:
        """Check if Brokle client is available and ready."""
        return self.client is not None and hasattr(self.client, 'observability')

    @safe_operation("openai", "instrument", ErrorSeverity.HIGH)
    def instrument(self) -> bool:
        """Instrument OpenAI library for automatic observability."""
        if not self.is_available():
            raise LibraryNotAvailableError(
                "OpenAI library not available for instrumentation",
                severity=ErrorSeverity.HIGH,
                library="openai",
                operation="instrument"
            )

        if self._instrumented:
            logger.info("OpenAI already instrumented")
            return True

        with instrumentation_context("openai", "instrument", ErrorSeverity.HIGH):
            # Validate configuration before proceeding
            if not self._validate_instrumentation_requirements():
                return False

            # Instrument synchronous client
            sync_success = self._instrument_sync_client()

            # Instrument asynchronous client
            async_success = self._instrument_async_client()

            # Consider successful if at least one client was instrumented
            if sync_success or async_success:
                self._instrumented = True
                logger.info("OpenAI instrumentation enabled successfully")

                # Reset error handler for successful instrumentation
                error_handler = get_error_handler()
                error_handler.reset_errors("openai", "instrument")

                return True
            else:
                raise InstrumentationError(
                    "Failed to instrument both sync and async OpenAI clients",
                    severity=ErrorSeverity.HIGH,
                    library="openai",
                    operation="instrument"
                )

    def _validate_instrumentation_requirements(self) -> bool:
        """Validate requirements for successful instrumentation."""
        try:
            # Check if we can access OpenAI modules
            if not hasattr(openai, 'resources'):
                logger.warning("OpenAI resources not available - may be older version")
                return False

            # Validate configuration
            try:
                config = self.config
                if config and hasattr(config, 'validate'):
                    config.validate()
            except Exception as e:
                logger.debug(f"Configuration validation failed, continuing with defaults: {e}")

            return True

        except Exception as e:
            logger.error(f"Instrumentation requirements validation failed: {e}")
            return False

    @safe_operation("openai", "uninstrument", ErrorSeverity.MEDIUM)
    def uninstrument(self) -> bool:
        """Remove OpenAI instrumentation."""
        if not self._instrumented:
            logger.debug("OpenAI not currently instrumented")
            return True

        with instrumentation_context("openai", "uninstrument", ErrorSeverity.MEDIUM):
            success_count = 0
            total_methods = len(self._original_methods)

            # Restore original methods
            for method_path, original_method in self._original_methods.items():
                try:
                    self._set_method_by_path(method_path, original_method)
                    success_count += 1
                except Exception as e:
                    logger.warning(f"Failed to restore method {method_path}: {e}")

            self._original_methods.clear()
            self._instrumented = False

            if success_count == total_methods:
                logger.info("OpenAI instrumentation removed successfully")
                # Reset error handler for successful uninstrumentation
                error_handler = get_error_handler()
                error_handler.reset_errors("openai", "uninstrument")
                return True
            else:
                logger.warning(f"Partial uninstrumentation: {success_count}/{total_methods} methods restored")
                return success_count > 0

    @safe_operation("openai", "instrument_sync", ErrorSeverity.MEDIUM)
    def _instrument_sync_client(self) -> bool:
        """Instrument synchronous OpenAI client methods."""
        instrumented_methods = 0
        failed_methods = []

        # Define methods to instrument
        methods_to_instrument = [
            {
                "path": "openai.resources.chat.completions.Completions",
                "attr_path": ["resources", "chat", "completions"],
                "class_name": "Completions",
                "method": "create",
                "name": "chat_completions_create",
                "type": "llm"
            },
            {
                "path": "openai.resources.completions.Completions",
                "attr_path": ["resources", "completions"],
                "class_name": "Completions",
                "method": "create",
                "name": "completions_create",
                "type": "llm"
            },
            {
                "path": "openai.resources.embeddings.Embeddings",
                "attr_path": ["resources", "embeddings"],
                "class_name": "Embeddings",
                "method": "create",
                "name": "embeddings_create",
                "type": "llm"
            }
        ]

        for method_config in methods_to_instrument:
            try:
                # Navigate to the class
                obj = openai
                for attr in method_config["attr_path"]:
                    if not hasattr(obj, attr):
                        raise AttributeError(f"Missing attribute: {attr}")
                    obj = getattr(obj, attr)

                if not hasattr(obj, method_config["class_name"]):
                    logger.debug(f"Class {method_config['class_name']} not found in {method_config['path']}")
                    continue

                target_class = getattr(obj, method_config["class_name"])

                if not hasattr(target_class, method_config["method"]):
                    logger.debug(f"Method {method_config['method']} not found in {method_config['class_name']}")
                    continue

                # Instrument the method
                original_method = getattr(target_class, method_config["method"])
                method_path = f"{method_config['path']}.{method_config['method']}"

                self._original_methods[method_path] = original_method
                setattr(target_class, method_config["method"], self._wrap_sync_method(
                    original_method,
                    method_name=method_config["name"],
                    observation_type=method_config["type"]
                ))

                instrumented_methods += 1
                logger.debug(f"Successfully instrumented {method_path}")

            except Exception as e:
                failed_methods.append(f"{method_config['path']}.{method_config['method']}: {e}")
                logger.debug(f"Failed to instrument {method_config['path']}: {e}")

        if failed_methods:
            logger.warning(f"Failed to instrument some sync methods: {failed_methods}")

        success = instrumented_methods > 0
        logger.info(f"Sync client instrumentation: {instrumented_methods} methods instrumented")

        return success

    @safe_operation("openai", "instrument_async", ErrorSeverity.MEDIUM)
    def _instrument_async_client(self) -> bool:
        """Instrument asynchronous OpenAI client methods."""
        instrumented_methods = 0
        failed_methods = []

        # Define async methods to instrument
        async_methods_to_instrument = [
            {
                "path": "openai.resources.chat.completions.AsyncCompletions",
                "attr_path": ["resources", "chat", "completions"],
                "class_name": "AsyncCompletions",
                "method": "create",
                "name": "async_chat_completions_create",
                "type": "llm"
            },
            {
                "path": "openai.resources.completions.AsyncCompletions",
                "attr_path": ["resources", "completions"],
                "class_name": "AsyncCompletions",
                "method": "create",
                "name": "async_completions_create",
                "type": "llm"
            },
            {
                "path": "openai.resources.embeddings.AsyncEmbeddings",
                "attr_path": ["resources", "embeddings"],
                "class_name": "AsyncEmbeddings",
                "method": "create",
                "name": "async_embeddings_create",
                "type": "llm"
            }
        ]

        for method_config in async_methods_to_instrument:
            try:
                # Navigate to the class
                obj = openai
                for attr in method_config["attr_path"]:
                    if not hasattr(obj, attr):
                        raise AttributeError(f"Missing attribute: {attr}")
                    obj = getattr(obj, attr)

                if not hasattr(obj, method_config["class_name"]):
                    logger.debug(f"Async class {method_config['class_name']} not found in {method_config['path']}")
                    continue

                target_class = getattr(obj, method_config["class_name"])

                if not hasattr(target_class, method_config["method"]):
                    logger.debug(f"Method {method_config['method']} not found in {method_config['class_name']}")
                    continue

                # Instrument the async method
                original_method = getattr(target_class, method_config["method"])
                method_path = f"{method_config['path']}.{method_config['method']}"

                self._original_methods[method_path] = original_method
                setattr(target_class, method_config["method"], self._wrap_async_method(
                    original_method,
                    method_name=method_config["name"],
                    observation_type=method_config["type"]
                ))

                instrumented_methods += 1
                logger.debug(f"Successfully instrumented async {method_path}")

            except Exception as e:
                failed_methods.append(f"{method_config['path']}.{method_config['method']}: {e}")
                logger.debug(f"Failed to instrument async {method_config['path']}: {e}")

        if failed_methods:
            logger.warning(f"Failed to instrument some async methods: {failed_methods}")

        success = instrumented_methods > 0
        logger.info(f"Async client instrumentation: {instrumented_methods} methods instrumented")

        return success

    def _wrap_sync_method(self, original_method, method_name: str, observation_type: str):
        """Wrap synchronous method with observability."""

        @functools.wraps(original_method)
        def wrapper(*args, **kwargs):
            # Initialize tracking variables
            observation_id = None
            start_time = time.perf_counter()

            # Check if observability is healthy for this operation
            error_handler = get_error_handler()
            observability_healthy = error_handler.is_operation_healthy("openai", f"observe_{method_name}")

            # Try to set up observability, but never break user's code
            if observability_healthy and self.client is not None:
                with instrumentation_context("openai", f"observe_{method_name}", ErrorSeverity.LOW):
                    try:
                        # Extract request data safely
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
                            "start_time": datetime.now(datetime.timezone.utc)
                        }

                        observation = self.client.observability.create_observation_sync(**observation_data)
                        observation_id = observation.id
                        logger.debug(f"Created observation {observation_id} for {method_name}")

                    except Exception as e:
                        logger.debug(f"Failed to setup observability for {method_name}: {e}")
                        error_handler.handle_error(e, "openai", f"observe_{method_name}", ErrorSeverity.LOW)
                        observation_id = None
            else:
                if not observability_healthy:
                    logger.debug(f"Skipping observability for {method_name} - circuit breaker open")
                elif self.client is None:
                    logger.debug(f"Skipping observability for {method_name} - no client available")

            # Execute original method - this should never fail due to observability
            try:
                result = original_method(*args, **kwargs)

                # Try to complete observation with success data
                if observation_id and observability_healthy:
                    with instrumentation_context("openai", f"complete_{method_name}", ErrorSeverity.LOW):
                        try:
                            end_time = datetime.now(datetime.timezone.utc)
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
                            logger.debug(f"Completed observation {observation_id} for {method_name}")

                        except Exception as e:
                            logger.debug(f"Failed to complete observation: {e}")
                            error_handler.handle_error(e, "openai", f"complete_{method_name}", ErrorSeverity.LOW)

                return result

            except Exception as e:
                # Try to record the error in observation, but don't fail if it doesn't work
                if observation_id and observability_healthy:
                    with instrumentation_context("openai", f"error_{method_name}", ErrorSeverity.LOW):
                        try:
                            end_time = datetime.now(datetime.timezone.utc)
                            latency_ms = int((time.perf_counter() - start_time) * 1000)

                            self.client.observability.complete_observation_sync(
                                observation_id,
                                end_time=end_time,
                                latency_ms=latency_ms,
                                status_message=f"error: {str(e)}"
                            )
                            logger.debug(f"Recorded error in observation {observation_id} for {method_name}")

                        except Exception as complete_error:
                            logger.debug(f"Failed to record error in observation: {complete_error}")
                            error_handler.handle_error(complete_error, "openai", f"error_{method_name}", ErrorSeverity.LOW)

                # Always re-raise the original exception - observability must never break user code
                raise

        return wrapper

    def _wrap_async_method(self, original_method, method_name: str, observation_type: str):
        """Wrap asynchronous method with observability."""

        @functools.wraps(original_method)
        @safe_async_operation("openai", f"async_{method_name}", ErrorSeverity.LOW)
        async def async_wrapper(*args, **kwargs):
            # Initialize tracking variables
            observation_id = None
            start_time = time.perf_counter()
            request_data = {}

            # Check if observability is healthy for this operation
            error_handler = get_error_handler()
            observability_healthy = error_handler.is_operation_healthy("openai", f"async_observe_{method_name}")

            # Try to set up observability, but never break user's code
            if observability_healthy and self.client is not None:
                try:
                    # Extract request data safely
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
                        "start_time": datetime.now(datetime.timezone.utc)
                    }

                    observation = await self.client.observability.create_observation(**observation_data)
                    observation_id = observation.id
                    logger.debug(f"Created async observation {observation_id} for {method_name}")

                except Exception as e:
                    logger.debug(f"Failed to create async observation: {e}")
                    error_handler.handle_error(e, "openai", f"async_observe_{method_name}", ErrorSeverity.LOW)
                    observation_id = None
            else:
                if not observability_healthy:
                    logger.debug(f"Skipping async observability for {method_name} - circuit breaker open")
                elif self.client is None:
                    logger.debug(f"Skipping async observability for {method_name} - no client available")

            # Execute original async method - this should never fail due to observability
            try:
                result = await original_method(*args, **kwargs)

                # Try to complete observation with success data
                if observation_id and observability_healthy:
                    try:
                        end_time = datetime.now(datetime.timezone.utc)
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

                        await self.client.observability.complete_observation(
                            observation_id,
                            **completion_data
                        )
                        logger.debug(f"Completed async observation {observation_id} for {method_name}")

                    except Exception as e:
                        logger.debug(f"Failed to complete async observation: {e}")
                        error_handler.handle_error(e, "openai", f"async_complete_{method_name}", ErrorSeverity.LOW)

                return result

            except Exception as e:
                # Try to record the error in observation, but don't fail if it doesn't work
                if observation_id and observability_healthy:
                    try:
                        end_time = datetime.now(datetime.timezone.utc)
                        latency_ms = int((time.perf_counter() - start_time) * 1000)

                        await self.client.observability.complete_observation(
                            observation_id,
                            end_time=end_time,
                            latency_ms=latency_ms,
                            status_message=f"error: {str(e)}"
                        )
                        logger.debug(f"Recorded error in async observation {observation_id} for {method_name}")

                    except Exception as complete_error:
                        logger.debug(f"Failed to record error in async observation: {complete_error}")
                        error_handler.handle_error(complete_error, "openai", f"async_error_{method_name}", ErrorSeverity.LOW)

                # Always re-raise the original exception - observability must never break user code
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

    @safe_operation("openai", "create_trace", ErrorSeverity.LOW)
    def _get_or_create_trace(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Get or create a trace synchronously."""
        if self.client is None:
            # Return fallback trace ID if no client
            import uuid
            return str(uuid.uuid4())

        try:
            trace = self.client.observability.create_trace_sync(
                name=name,
                metadata=metadata or {}
            )
            return trace.id
        except Exception as e:
            logger.debug(f"Failed to create trace: {e}")
            # Always return a valid trace ID to prevent downstream failures
            import uuid
            fallback_id = str(uuid.uuid4())
            logger.debug(f"Using fallback trace ID: {fallback_id}")
            return fallback_id

    @safe_async_operation("openai", "create_trace_async", ErrorSeverity.LOW)
    async def _get_or_create_trace_async(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Get or create a trace asynchronously."""
        if self.client is None:
            # Return fallback trace ID if no client
            import uuid
            return str(uuid.uuid4())

        try:
            trace = await self.client.observability.create_trace(
                name=name,
                metadata=metadata or {}
            )
            return trace.id
        except Exception as e:
            logger.debug(f"Failed to create async trace: {e}")
            # Always return a valid trace ID to prevent downstream failures
            import uuid
            fallback_id = str(uuid.uuid4())
            logger.debug(f"Using fallback async trace ID: {fallback_id}")
            return fallback_id

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
