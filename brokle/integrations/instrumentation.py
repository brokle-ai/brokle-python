"""
Universal Instrumentation Engine

Core engine that provides consistent instrumentation patterns across
all AI providers using wrapt for method wrapping and OpenTelemetry for observability.

This is the heart of Brokle's provider-agnostic observability system.
"""

import asyncio
import functools
import logging
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Union

try:
    import wrapt

    HAS_WRAPT = True
except ImportError:
    wrapt = None
    HAS_WRAPT = False

from .._utils.error_handling import handle_provider_error
from ..exceptions import BrokleError, ProviderError
from ..observability import create_span, get_client
from ..observability.attributes import (
    BrokleOtelSpanAttributes as BrokleInstrumentationAttributes,
)
from ..providers.base import BaseProvider
from .validators import AttributeValidator

logger = logging.getLogger(__name__)


class InstrumentationContext:
    """
    Context manager for individual method instrumentation.

    Handles span creation, attribute collection, error handling,
    and cleanup for each instrumented method call.
    """

    def __init__(
        self,
        provider: BaseProvider,
        method_def: Dict[str, Any],
        args: tuple,
        kwargs: dict,
    ):
        self.provider = provider
        self.method_def = method_def
        self.args = args
        self.kwargs = kwargs
        self.span = None
        self.start_time = None
        self.response = None
        self.error = None

    def __enter__(self):
        """Initialize span and capture request attributes."""
        self.start_time = time.time()

        try:

            # Create span with initial attributes
            span_name = f"{self.provider.name}.{self.method_def['operation']}"

            initial_attributes = {
                BrokleInstrumentationAttributes.PROVIDER: self.provider.name,
                BrokleInstrumentationAttributes.OPERATION_TYPE: self.method_def[
                    "operation"
                ],
                BrokleInstrumentationAttributes.REQUEST_START_TIME: self.start_time,
                BrokleInstrumentationAttributes.METHOD_PATH: self.method_def["path"],
                BrokleInstrumentationAttributes.IS_ASYNC: self.method_def.get(
                    "async", False
                ),
                BrokleInstrumentationAttributes.STREAM_SUPPORT: self.method_def.get(
                    "stream_support", False
                ),
            }

            # Add provider-specific request attributes
            try:
                request_attributes = self.provider.extract_request_attributes(
                    self.kwargs
                )
                initial_attributes.update(request_attributes)
            except Exception as e:
                logger.warning(f"Failed to extract request attributes: {e}")

            # Add wrapper configuration if available
            if hasattr(self.provider, "config"):
                config = self.provider.config
                if config.get("tags"):
                    initial_attributes[BrokleInstrumentationAttributes.TAGS] = ",".join(
                        config["tags"]
                    )
                if config.get("session_id"):
                    initial_attributes[BrokleInstrumentationAttributes.SESSION_ID] = (
                        config["session_id"]
                    )
                if config.get("user_id"):
                    initial_attributes[BrokleInstrumentationAttributes.USER_ID] = (
                        config["user_id"]
                    )

            # Sanitize attributes before creating span
            sanitized_attributes = AttributeValidator.sanitize_attributes(
                initial_attributes
            )

            # Create the span
            self.span = create_span(name=span_name, attributes=sanitized_attributes)

            logger.debug(f"Created span for {span_name}")

        except Exception as e:
            logger.warning(f"Failed to create instrumentation span: {e}")
            # Continue without span rather than failing the request

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finalize span with response attributes and timing."""
        try:
            if self.span:
                # Calculate duration
                duration = time.time() - self.start_time if self.start_time else 0

                # Add timing attributes
                end_attributes = {
                    BrokleInstrumentationAttributes.REQUEST_DURATION: duration
                    * 1000,  # Convert to ms
                    BrokleInstrumentationAttributes.RESPONSE_END_TIME: time.time(),
                }

                # Add response attributes if we have a successful response
                if self.response is not None and exc_type is None:
                    try:
                        response_attributes = self.provider.extract_response_attributes(
                            self.response
                        )
                        end_attributes.update(response_attributes)
                    except Exception as e:
                        logger.warning(f"Failed to extract response attributes: {e}")

                # Add error attributes if there was an exception
                if exc_type is not None:
                    self.record_span_exception(self.span, exc_val)

                    # Add provider-specific error mapping
                    error_mapping = self.provider.get_error_mapping()
                    provider_error_name = exc_type.__name__
                    brokle_error_type = error_mapping.get(
                        provider_error_name, "ProviderError"
                    )

                    end_attributes.update(
                        {
                            BrokleInstrumentationAttributes.ERROR_TYPE: provider_error_name,
                            BrokleInstrumentationAttributes.ERROR_MESSAGE: str(exc_val),
                            BrokleInstrumentationAttributes.BROKLE_ERROR_TYPE: brokle_error_type,
                            BrokleInstrumentationAttributes.TRACEBACK: traceback.format_exc(),
                        }
                    )

                # Set success/failure status
                end_attributes[BrokleInstrumentationAttributes.SUCCESS] = (
                    exc_type is None
                )

                # Sanitize and apply all end attributes
                sanitized_end_attributes = AttributeValidator.sanitize_attributes(
                    end_attributes
                )
                self.add_span_attributes(self.span, sanitized_end_attributes)

                # End the span
                self.span.end()

                # Submit additional telemetry through background processor
                self._submit_instrumentation_telemetry(
                    end_attributes, exc_type, duration
                )

                logger.debug(
                    f"Finalized span for {self.provider.name}.{self.method_def['operation']}"
                )

        except Exception as e:
            logger.warning(f"Failed to finalize instrumentation span: {e}")

    def record_span_exception(self, span, exception):
        """Record exception in span (stub implementation)."""
        if hasattr(span, "set_status"):
            span.set_status("error", str(exception))

    def add_span_attributes(self, span, attributes):
        """Add attributes to span (stub implementation)."""
        if hasattr(span, "attributes"):
            span.attributes.update(attributes)

    def _submit_instrumentation_telemetry(
        self, end_attributes: dict, exc_type, duration: float
    ):
        """Submit instrumentation telemetry through background processor."""
        try:
            # Get Brokle client for telemetry submission
            from ..observability.context import get_client

            client = get_client()
            if not client:
                return

            # Build comprehensive telemetry data
            telemetry_data = {
                "type": "instrumentation",
                "provider": self.provider.name,
                "operation": self.method_def["operation"],
                "method_path": self.method_def["path"],
                "duration_ms": duration * 1000,
                "success": exc_type is None,
                "timestamp": time.time(),
            }

            # Add request attributes (from span initial attributes)
            if self.span and hasattr(self.span, "attributes"):
                # Extract meaningful request attributes
                request_attrs = {}
                for key, value in self.span.attributes.items():
                    if any(
                        req_key in key.lower()
                        for req_key in [
                            "model",
                            "input",
                            "prompt",
                            "messages",
                            "temperature",
                        ]
                    ):
                        request_attrs[key] = value

                if request_attrs:
                    telemetry_data["request_attributes"] = request_attrs

            # Add response attributes
            response_attrs = {}
            for key, value in end_attributes.items():
                if any(
                    resp_key in key.lower()
                    for resp_key in [
                        "output",
                        "tokens",
                        "cost",
                        "finish_reason",
                        "model",
                    ]
                ):
                    response_attrs[key] = value

            if response_attrs:
                telemetry_data["response_attributes"] = response_attrs

            # Add error information if present
            if exc_type is not None:
                telemetry_data.update(
                    {
                        "error_type": exc_type.__name__,
                        "error_message": (
                            str(exc_type)
                            if hasattr(exc_type, "__str__")
                            else "Unknown error"
                        ),
                        "brokle_error_type": end_attributes.get(
                            BrokleInstrumentationAttributes.BROKLE_ERROR_TYPE,
                            "ProviderError",
                        ),
                    }
                )

            # Add span information for correlation
            if self.span:
                telemetry_data.update(
                    {
                        "span_id": getattr(self.span, "span_id", None),
                        "trace_id": getattr(self.span, "trace_id", None),
                    }
                )

            # Submit telemetry
            client.submit_telemetry(telemetry_data)
            logger.debug(
                f"Submitted instrumentation telemetry for {self.provider.name}.{self.method_def['operation']}"
            )

        except Exception as e:
            # Don't let telemetry errors break instrumentation
            logger.debug(f"Failed to submit instrumentation telemetry: {e}")

    def record_response(self, response: Any):
        """Record the response for attribute extraction."""
        self.response = response


class UniversalInstrumentation:
    """
    Universal instrumentation engine for AI providers.

    Provides consistent observability patterns across different AI SDKs
    while maintaining provider-specific customizations.
    """

    def __init__(self, provider: BaseProvider):
        """
        Initialize instrumentation with provider.

        Args:
            provider: Provider-specific implementation (OpenAI, Anthropic, etc.)
        """
        self.provider = provider
        self.instrumented_methods: Dict[str, bool] = {}
        self._instrumentation_enabled = True

    def instrument_client(self, client: Any) -> Any:
        """
        Instrument client with provider-specific observability patterns.

        Args:
            client: Provider SDK client (OpenAI, Anthropic, etc.)

        Returns:
            Wrapped client with observability

        Raises:
            ProviderError: If instrumentation fails critically
        """
        if not HAS_WRAPT:
            logger.warning(
                "wrapt library not available. Instrumentation disabled. "
                "Install with: pip install wrapt"
            )
            return client

        # Check if Brokle client is available and telemetry is enabled
        try:
            brokle_client = get_client()
            if brokle_client and not brokle_client.config.telemetry_enabled:
                logger.info(
                    "Telemetry disabled in Brokle configuration. Skipping instrumentation."
                )
                return client
        except Exception as e:
            logger.debug(f"Brokle client check failed: {e}")

        # Get provider-specific methods to instrument
        try:
            methods_to_wrap = self.provider.get_methods_to_instrument()
            logger.debug(
                f"Instrumenting {len(methods_to_wrap)} methods for {self.provider.name}"
            )

            successful_wraps = 0
            failed_wraps = 0

            for method_def in methods_to_wrap:
                try:
                    self._wrap_method(client, method_def)
                    successful_wraps += 1
                except Exception as e:
                    failed_wraps += 1
                    logger.warning(f"Failed to wrap {method_def['path']}: {e}")

            # Log instrumentation summary
            logger.info(
                f"{self.provider.name} instrumentation complete: "
                f"{successful_wraps} methods wrapped, {failed_wraps} failed"
            )

            # Add metadata to client
            setattr(client, "_brokle_instrumented", True)
            setattr(client, "_brokle_provider", self.provider.name)
            setattr(client, "_brokle_successful_wraps", successful_wraps)
            setattr(client, "_brokle_failed_wraps", failed_wraps)

            return client

        except Exception as e:
            logger.error(
                f"Critical error during {self.provider.name} instrumentation: {e}"
            )
            raise ProviderError(
                f"Failed to instrument {self.provider.name} client: {e}"
            )

    def _wrap_method(self, client: Any, method_def: Dict[str, Any]):
        """
        Wrap individual method with observability.

        Args:
            client: Provider SDK client
            method_def: Method definition with path and metadata
        """
        try:
            # Navigate to the method location
            target_obj = client
            path_parts = method_def["path"].split(".")

            # Navigate to parent object
            for part in path_parts[:-1]:
                if not hasattr(target_obj, part):
                    raise AttributeError(
                        f"Path component '{part}' not found in {target_obj}"
                    )
                target_obj = getattr(target_obj, part)

            method_name = path_parts[-1]

            # Verify method exists
            if not hasattr(target_obj, method_name):
                raise AttributeError(
                    f"Method '{method_name}' not found in {target_obj}"
                )

            # Create appropriate wrapper based on async/sync
            if method_def.get("async", False):
                wrapper_func = self._create_async_wrapper(method_def)
            else:
                wrapper_func = self._create_sync_wrapper(method_def)

            # Apply wrapper using wrapt
            wrapt.wrap_function_wrapper(target_obj, method_name, wrapper_func)

            # Track successful instrumentation
            method_key = f"{self.provider.name}.{method_def['path']}"
            self.instrumented_methods[method_key] = True

            logger.debug(f"Successfully wrapped {method_key}")

        except Exception as e:
            logger.warning(f"Failed to wrap {method_def['path']}: {e}")
            raise

    def _create_sync_wrapper(self, method_def: Dict[str, Any]) -> Callable:
        """Create synchronous method wrapper."""

        def wrapper(wrapped, instance, args, kwargs):
            """Synchronous wrapper with comprehensive observability."""

            # Skip instrumentation if disabled
            if not self._instrumentation_enabled:
                return wrapped(*args, **kwargs)

            # Check if Brokle client is available
            try:
                brokle_client = get_client()
                if not brokle_client or not brokle_client.config.telemetry_enabled:
                    return wrapped(*args, **kwargs)
            except:
                return wrapped(*args, **kwargs)

            # Execute with instrumentation context
            with InstrumentationContext(self.provider, method_def, args, kwargs) as ctx:
                try:
                    # Execute the original method
                    response = wrapped(*args, **kwargs)

                    # Record response for attribute extraction
                    ctx.record_response(response)

                    return response

                except Exception as e:
                    # Handle provider-specific errors
                    handled_error = handle_provider_error(
                        e, self.provider.name, method_def["operation"]
                    )
                    raise handled_error

        return wrapper

    def _create_async_wrapper(self, method_def: Dict[str, Any]) -> Callable:
        """Create asynchronous method wrapper."""

        async def async_wrapper(wrapped, instance, args, kwargs):
            """Asynchronous wrapper with comprehensive observability."""

            # Skip instrumentation if disabled
            if not self._instrumentation_enabled:
                return await wrapped(*args, **kwargs)

            # Check if Brokle client is available
            try:
                brokle_client = get_client()
                if not brokle_client or not brokle_client.config.telemetry_enabled:
                    return await wrapped(*args, **kwargs)
            except:
                return await wrapped(*args, **kwargs)

            # Execute with instrumentation context
            with InstrumentationContext(self.provider, method_def, args, kwargs) as ctx:
                try:
                    # Execute the original async method
                    response = await wrapped(*args, **kwargs)

                    # Record response for attribute extraction
                    ctx.record_response(response)

                    return response

                except Exception as e:
                    # Handle provider-specific errors
                    handled_error = handle_provider_error(
                        e, self.provider.name, method_def["operation"]
                    )
                    raise handled_error

        return async_wrapper

    def disable_instrumentation(self):
        """Temporarily disable instrumentation for this client."""
        self._instrumentation_enabled = False
        logger.info(f"Instrumentation disabled for {self.provider.name}")

    def enable_instrumentation(self):
        """Re-enable instrumentation for this client."""
        self._instrumentation_enabled = True
        logger.info(f"Instrumentation enabled for {self.provider.name}")

    def get_instrumentation_status(self) -> Dict[str, Any]:
        """Get current instrumentation status and statistics."""
        return {
            "provider": self.provider.name,
            "enabled": self._instrumentation_enabled,
            "instrumented_methods": list(self.instrumented_methods.keys()),
            "method_count": len(self.instrumented_methods),
            "wrapt_available": HAS_WRAPT,
        }


# Utility functions for manual instrumentation
def create_manual_span(provider_name: str, operation: str, **attributes):
    """
    Create a manual span for custom operations.

    Useful for instrumenting operations that don't go through
    the standard provider SDKs.
    """
    span_name = f"{provider_name}.{operation}"

    span_attributes = {
        BrokleInstrumentationAttributes.PROVIDER: provider_name,
        BrokleInstrumentationAttributes.OPERATION_TYPE: operation,
        BrokleInstrumentationAttributes.REQUEST_START_TIME: time.time(),
        BrokleInstrumentationAttributes.MANUAL_INSTRUMENTATION: True,
        **attributes,
    }

    # Sanitize attributes
    sanitized_attributes = AttributeValidator.sanitize_attributes(span_attributes)

    return create_span(name=span_name, attributes=sanitized_attributes)


# Export public API
__all__ = [
    "UniversalInstrumentation",
    "InstrumentationContext",
    "create_manual_span",
]
