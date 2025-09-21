"""
Instrumentation Engine for Brokle Platform.

This module provides the universal instrumentation logic that powers both
auto-instrumentation and manual wrapper approaches. It eliminates code
duplication and ensures consistent behavior across all providers.

The engine handles:
- Span creation and management
- Metrics collection and updates
- Error handling and circuit breaking
- Async/sync method detection
- Defensive programming patterns

Usage:
    engine = InstrumentationEngine()
    wrapper = engine.create_wrapper(openai_instrumentation, config)
"""

import inspect
import logging
import time
from typing import Any, Callable, Dict, Optional
from functools import wraps

from ._base import BaseInstrumentation, InstrumentationConfig, StandardMetadata, ManualInstrumentationOptions, _normalize_token_usage

logger = logging.getLogger(__name__)


def _create_manual_options(options_dict: Optional[Dict[str, Any]]) -> Optional[ManualInstrumentationOptions]:
    """Convert dictionary options to typed ManualInstrumentationOptions object."""
    if not options_dict:
        return None

    # Extract known fields
    options = ManualInstrumentationOptions(
        enable_caching=options_dict.get("enable_caching"),
        enable_evaluation=options_dict.get("enable_evaluation"),
        tags=options_dict.get("tags"),
        metadata=options_dict.get("metadata")
    )

    # Everything else goes to extra (excluding runtime-only options)
    for key, value in options_dict.items():
        if key not in {"enable_caching", "enable_evaluation", "tags", "metadata", "sample_rate"}:
            options.extra[key] = value

    return options


class CircuitBreaker:
    """Circuit breaker to prevent instrumentation from breaking user code."""

    def __init__(self, max_errors: int = 5, reset_threshold: int = 3):
        self.max_errors = max_errors
        self.reset_threshold = reset_threshold
        self.error_count = 0
        self.success_count = 0
        self.is_open = False

    def record_success(self):
        """Record a successful operation."""
        self.success_count += 1

        # Reset error counter after consecutive successes
        if self.success_count >= self.reset_threshold:
            self.error_count = 0
            self.success_count = 0
            self.is_open = False

    def record_error(self, error: Exception) -> bool:
        """
        Record an error and return whether circuit should break.

        Returns:
            True if circuit should break (stop instrumentation)
        """
        self.error_count += 1
        self.success_count = 0

        if self.error_count >= self.max_errors:
            self.is_open = True
            logger.warning(
                f"Circuit breaker opened after {self.error_count} errors. "
                f"Instrumentation disabled until {self.reset_threshold} successes."
            )

        return self.is_open

    def should_skip(self) -> bool:
        """Check if circuit breaker is open."""
        return self.is_open


class InstrumentationEngine:
    """
    Universal instrumentation engine for all providers.

    This engine provides the core wrapper logic that's shared between
    auto-instrumentation and manual wrappers. It ensures consistent
    behavior and eliminates code duplication.
    """

    def __init__(self):
        """Initialize the instrumentation engine."""
        self._circuit_breakers = {}  # Per-method circuit breakers
        self._error_counts = {}      # Global error tracking

    def _get_brokle_client(self):
        """Get Brokle client with error handling."""
        try:
            from ..client import get_client
            return get_client()
        except Exception as e:
            logger.debug(f"Brokle client not available: {e}")
            return None

    def _get_circuit_breaker(self, method_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for a specific method."""
        if method_name not in self._circuit_breakers:
            self._circuit_breakers[method_name] = CircuitBreaker()
        return self._circuit_breakers[method_name]

    def _record_error(self, method_name: str, provider: str, error: Exception):
        """Record instrumentation error for debugging."""
        error_key = f"{provider}.{method_name}"
        if error_key not in self._error_counts:
            self._error_counts[error_key] = []

        self._error_counts[error_key].append({
            "error": str(error),
            "type": type(error).__name__,
            "timestamp": time.time()
        })

        # Keep only last 10 errors per method
        self._error_counts[error_key] = self._error_counts[error_key][-10:]

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of instrumentation errors for debugging."""
        return {
            "errors_by_method": self._error_counts.copy(),
            "circuit_breakers": {
                method: {
                    "is_open": cb.is_open,
                    "error_count": cb.error_count,
                    "success_count": cb.success_count
                }
                for method, cb in self._circuit_breakers.items()
            }
        }

    def create_span(
        self,
        provider: str,
        config: InstrumentationConfig,
        metadata: StandardMetadata
    ):
        """
        Create observability span for the operation.

        Args:
            provider: Provider name (e.g., "openai")
            config: Method configuration
            metadata: Standardized metadata

        Returns:
            Observation object (span or generation)
        """
        client = self._get_brokle_client()
        if not client:
            return None

        try:
            # Simple span naming
            span_name = f"{provider.title()} {config.name}"

            # Prepare base metadata
            base_metadata = {
                "method": config.name,
                "operation_type": config.operation_type,
                "library": provider,
                "auto_instrumented": metadata.auto_instrumented,
                **metadata.request
            }

            # Add manual configuration flags and extract additional options
            manual_tags = None
            if metadata.manual_options:
                if metadata.manual_options.enable_caching is not None:
                    base_metadata["caching_enabled"] = metadata.manual_options.enable_caching
                if metadata.manual_options.enable_evaluation is not None:
                    base_metadata["evaluation_enabled"] = metadata.manual_options.enable_evaluation

                # Extract tags for span creation
                manual_tags = metadata.manual_options.tags

                # Merge additional metadata
                if metadata.manual_options.metadata:
                    base_metadata.update(metadata.manual_options.metadata)

                # Merge any extra options
                if metadata.manual_options.extra:
                    base_metadata.update(metadata.manual_options.extra)

            # Create generation for LLM operations, span for others
            if config.operation_type == "llm":
                generation_kwargs = {
                    "name": span_name,
                    "model": metadata.model,
                    "provider": provider,
                    "metadata": base_metadata
                }
                if manual_tags is not None:
                    generation_kwargs["tags"] = manual_tags

                observation = client.generation(**generation_kwargs).start()
            else:
                span_metadata = {
                    "provider": provider,
                    "model": metadata.model,
                    **base_metadata
                }
                span_kwargs = {
                    "name": span_name,
                    "metadata": span_metadata
                }
                if manual_tags is not None:
                    span_kwargs["tags"] = manual_tags

                observation = client.span(**span_kwargs).start()

            return observation

        except Exception as e:
            logger.debug(f"Failed to create span for {provider}.{config.name}: {e}")
            return None

    def update_span_metrics(
        self,
        observation,
        metadata: StandardMetadata
    ):
        """
        Update span with metrics and metadata.

        Args:
            observation: Span or generation object
            metadata: Complete metadata with response data
        """
        if not observation:
            return

        try:
            # Update metrics if observation supports it
            if hasattr(observation, "update_metrics") and callable(getattr(observation, "update_metrics")):
                # Normalize token usage to preserve zero counts and handle field name variations
                normalized_usage = _normalize_token_usage(metadata.usage or {})

                observation.update_metrics(
                    input_tokens=normalized_usage.get("input_tokens"),
                    output_tokens=normalized_usage.get("output_tokens"),
                    total_tokens=normalized_usage.get("total_tokens"),
                    cost_usd=metadata.cost_usd,
                    latency_ms=metadata.latency_ms
                )

            # Update metadata
            existing_metadata = getattr(observation, "metadata", {})
            if isinstance(existing_metadata, dict):
                updated_metadata = {
                    **existing_metadata,
                    "response": metadata.response,
                    "usage": metadata.usage,
                    "total_cost": metadata.cost_usd
                }
                observation.update(metadata=updated_metadata)

        except Exception as e:
            logger.debug(f"Failed to update span metrics: {e}")

    def finalize_span_success(self, observation, metadata: StandardMetadata):
        """Finalize span for successful operations."""
        if not observation:
            return

        try:
            self.update_span_metrics(observation, metadata)
            observation.end(status_message="success")

            logger.debug(
                f"Completed observation for {metadata.provider}.{metadata.operation_type} "
                f"(cost: ${metadata.cost_usd})"
            )

        except Exception as e:
            logger.debug(f"Failed to finalize successful span: {e}")

    def finalize_span_error(self, observation, exception: Exception):
        """Finalize span for error cases."""
        if not observation:
            return

        try:
            from opentelemetry.trace import StatusCode

            observation.end(
                status=StatusCode.ERROR,
                status_message=f"error: {str(exception)[:200]}"
            )

            logger.debug(f"Recorded error in observation: {exception}")

        except Exception as complete_error:
            logger.debug(f"Failed to record error in observation: {complete_error}")

    def create_wrapper(
        self,
        instrumentation: BaseInstrumentation,
        config: InstrumentationConfig,
        manual_options: Optional[Dict[str, Any]] = None
    ) -> Callable:
        """
        Create a universal wrapper for any provider method.

        This is the core wrapper factory that handles both sync and async
        methods with comprehensive error handling and observability.

        Args:
            instrumentation: Provider-specific instrumentation implementation
            config: Configuration for the specific method
            manual_options: Manual wrapper options (project, tags, metadata, etc.)

        Returns:
            Wrapper function suitable for use with wrapt
        """

        def wrapper(wrapped, instance, args, kwargs):
            """Universal wrapper for provider methods."""
            start_time = time.perf_counter()
            circuit_breaker = self._get_circuit_breaker(config.name)

            # Circuit breaker: skip instrumentation if too many errors
            if circuit_breaker.should_skip():
                return wrapped(*args, **kwargs)

            # Initialize tracking variables
            observation = None

            def _finalize_success(result_data: Any, *, latency_ms: int) -> None:
                """Finalize observation on success."""
                nonlocal observation

                try:
                    # Create standardized metadata
                    metadata = instrumentation.create_standard_metadata(
                        config, args, kwargs, result_data, latency_ms
                    )

                    # Fix manual instrumentation flags
                    metadata.auto_instrumented = (manual_options is None)
                    metadata.manual_options = _create_manual_options(manual_options)

                    # Update and finalize span
                    self.finalize_span_success(observation, metadata)

                    # Record success for circuit breaker
                    circuit_breaker.record_success()

                except Exception as e:
                    logger.debug(f"Failed to finalize successful observation: {e}")

            def _finalize_error(exc: Exception) -> None:
                """Finalize observation for error cases."""
                nonlocal observation

                try:
                    self.finalize_span_error(observation, exc)

                    # Update circuit breaker
                    should_break = circuit_breaker.record_error(exc)
                    if should_break:
                        logger.warning(
                            f"Circuit breaker opened for {instrumentation.provider_name}.{config.name}"
                        )

                except Exception as complete_error:
                    logger.debug(f"Failed to record error: {complete_error}")

            # Setup observability (non-blocking)
            try:
                # Extract initial metadata for span creation
                request_metadata = instrumentation.extract_request_metadata(args, kwargs)

                # Create initial metadata (will be completed after response)
                initial_metadata = StandardMetadata(
                    provider=instrumentation.provider_name,
                    model=request_metadata.get("model", "unknown"),
                    operation_type=config.operation_type,
                    request=request_metadata,
                    response={},  # Will be filled after response
                    usage={},     # Will be filled after response
                    auto_instrumented=manual_options is None,  # False if manual options provided
                    manual_options=_create_manual_options(manual_options)
                )

                # Create span
                observation = self.create_span(
                    instrumentation.provider_name,
                    config,
                    initial_metadata
                )

                logger.debug(
                    f"Created observation for {instrumentation.provider_name}.{config.name}"
                )

            except Exception as e:
                logger.debug(f"Failed to setup observability for {config.name}: {e}")
                self._record_error(config.name, instrumentation.provider_name, e)
                # Continue without observability - never break user code

            # Execute original method (this should NEVER fail due to observability)
            try:
                result = wrapped(*args, **kwargs)

                # Handle async methods
                if inspect.isawaitable(result):

                    async def _async_wrapper():
                        try:
                            awaited_result = await result
                            latency_ms = int((time.perf_counter() - start_time) * 1000)
                            _finalize_success(awaited_result, latency_ms=latency_ms)
                            return awaited_result
                        except Exception as async_exc:
                            _finalize_error(async_exc)
                            raise

                    return _async_wrapper()

                # Handle sync methods
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                _finalize_success(result, latency_ms=latency_ms)

                return result

            except Exception as e:
                _finalize_error(e)
                # Always re-raise the original exception
                raise

        return wrapper

    def create_manual_wrapper(
        self,
        instrumentation: BaseInstrumentation,
        original_method: Callable,
        config: InstrumentationConfig,
        **wrapper_options
    ) -> Callable:
        """
        Create a manual wrapper for selective instrumentation.

        This creates a wrapper that can be applied manually to specific
        client methods, with additional configuration options.

        Args:
            instrumentation: Provider instrumentation
            original_method: Original method to wrap
            config: Method configuration
            **wrapper_options: Additional options (project, sample_rate, etc.)

        Returns:
            Wrapped method with instrumentation
        """
        universal_wrapper = self.create_wrapper(instrumentation, config, manual_options=wrapper_options)

        @wraps(original_method)
        def manual_wrapper(*args, **kwargs):
            # Apply sampling if specified
            sample_rate = wrapper_options.get("sample_rate", 1.0)
            if sample_rate < 1.0:
                import random
                if random.random() > sample_rate:
                    return original_method(*args, **kwargs)

            # Use universal wrapper with original method
            return universal_wrapper(original_method, None, args, kwargs)

        # Attach configuration for debugging
        manual_wrapper._brokle_config = config
        manual_wrapper._brokle_options = wrapper_options

        return manual_wrapper