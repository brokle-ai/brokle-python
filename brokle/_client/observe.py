"""
Observe decorator for Brokle SDK.

This module provides the @observe decorator with Brokle-specific enhancements for AI platform features.
"""

import asyncio
import functools
import inspect
import logging
from typing import Any, Callable, Dict, Optional, TypeVar, Union, overload
from datetime import datetime

from typing_extensions import ParamSpec

from .client import get_client
from .span import BrokleSpan, BrokleGeneration
from .._utils.serialization import serialize

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])
P = ParamSpec("P")
R = TypeVar("R")


class BrokleDecorator:
    """
    Decorator implementation for Brokle Platform.

    This decorator provides seamless integration of Brokle observability
    with Brokle-specific enhancements.
    """

    def __init__(self):
        # Get client which will handle configuration internally
        self.client = get_client()

    @overload
    def observe(self, func: F) -> F: ...

    @overload
    def observe(
        self,
        func: None = None,
        *,
        name: Optional[str] = None,
        as_type: Optional[str] = None,
        capture_input: Optional[bool] = None,
        capture_output: Optional[bool] = None,
        # LLM-specific parameters
        model: Optional[str] = None,
        provider: Optional[str] = None,
        # Brokle-specific parameters
        routing_strategy: Optional[str] = None,
        cache_enabled: Optional[bool] = None,
        quality_evaluation: Optional[bool] = None,
        # Standard parameters
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        parent_observation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
        level: str = "DEFAULT",
        **kwargs
    ) -> Callable[[F], F]: ...

    def observe(
        self,
        func: Optional[F] = None,
        *,
        name: Optional[str] = None,
        as_type: Optional[str] = None,
        capture_input: Optional[bool] = None,
        capture_output: Optional[bool] = None,
        # LLM-specific parameters
        model: Optional[str] = None,
        provider: Optional[str] = None,
        # Brokle-specific parameters
        routing_strategy: Optional[str] = None,
        cache_enabled: Optional[bool] = None,
        quality_evaluation: Optional[bool] = None,
        # Standard parameters
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        parent_observation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
        level: str = "DEFAULT",
        **kwargs
    ) -> Union[F, Callable[[F], F]]:
        """
        Observe decorator with Brokle enhancements.

        This decorator automatically creates spans around function execution,
        capturing timing, inputs/outputs, and LLM-specific metrics.

        Args:
            func: The function to decorate
            name: Custom name for the span
            as_type: Type of observation ("span", "generation", "tool", "chain")
            capture_input: Whether to capture function inputs
            capture_output: Whether to capture function outputs
            model: LLM model name (for generation spans)
            provider: LLM provider name (for generation spans)
            routing_strategy: Brokle routing strategy
            cache_enabled: Whether semantic caching is enabled
            quality_evaluation: Whether to evaluate response quality
            user_id: User ID for tracking
            session_id: Session ID for tracking
            trace_id: Specific trace ID
            parent_observation_id: Parent observation ID
            metadata: Additional metadata
            tags: Tags for the observation
            level: Observation level (DEBUG, INFO, WARN, ERROR, DEFAULT)
            **kwargs: Additional attributes

        Examples:
            Basic function tracing:
            ```python
            @observe()
            def process_data(data):
                return transform(data)
            ```

            LLM generation tracking:
            ```python
            @observe(as_type="generation", model="gpt-4")
            async def generate_text(prompt):
                response = await openai.chat.completions.create(...)
                return response.choices[0].message.content
            ```

            Brokle AI platform features:
            ```python
            @observe(
                as_type="generation",
                routing_strategy="cost_optimized",
                cache_enabled=True,
                quality_evaluation=True
            )
            async def smart_completion(prompt):
                # Brokle handles routing, caching, and quality scoring
                return await llm_call(prompt)
            ```
        """
        def decorator(func: F) -> F:
            if asyncio.iscoroutinefunction(func):
                return self._async_observe(
                    func,
                    name=name or func.__name__,
                    as_type=as_type or "span",
                    capture_input=capture_input if capture_input is not None else True,
                    capture_output=capture_output if capture_output is not None else True,
                    model=model,
                    provider=provider,
                    routing_strategy=routing_strategy,
                    cache_enabled=cache_enabled,
                    quality_evaluation=quality_evaluation,
                    user_id=user_id,
                    session_id=session_id,
                    trace_id=trace_id,
                    parent_observation_id=parent_observation_id,
                    metadata=metadata or {},
                    tags=tags or [],
                    level=level,
                    **kwargs
                )
            else:
                return self._sync_observe(
                    func,
                    name=name or func.__name__,
                    as_type=as_type or "span",
                    capture_input=capture_input if capture_input is not None else True,
                    capture_output=capture_output if capture_output is not None else True,
                    model=model,
                    provider=provider,
                    routing_strategy=routing_strategy,
                    cache_enabled=cache_enabled,
                    quality_evaluation=quality_evaluation,
                    user_id=user_id,
                    session_id=session_id,
                    trace_id=trace_id,
                    parent_observation_id=parent_observation_id,
                    metadata=metadata or {},
                    tags=tags or [],
                    level=level,
                    **kwargs
                )

        if func is None:
            return decorator
        else:
            return decorator(func)

    def _async_observe(
        self,
        func: Callable,
        name: str,
        as_type: str,
        capture_input: bool,
        capture_output: bool,
        model: Optional[str],
        provider: Optional[str],
        routing_strategy: Optional[str],
        cache_enabled: Optional[bool],
        quality_evaluation: Optional[bool],
        user_id: Optional[str],
        session_id: Optional[str],
        trace_id: Optional[str],
        parent_observation_id: Optional[str],
        metadata: Dict[str, Any],
        tags: list[str],
        level: str,
        **kwargs
    ) -> Callable:
        """Async function decorator implementation."""

        @functools.wraps(func)
        async def async_wrapper(*args, **func_kwargs):
            client = get_client()

            # Create appropriate span type
            if as_type == "generation":
                span = client.generation(
                    name=name,
                    model=model,
                    provider=provider,
                    trace_id=trace_id,
                    parent_observation_id=parent_observation_id,
                    level=level,
                    metadata=metadata,
                    tags=tags
                )
            else:
                span = client.span(
                    name=name,
                    trace_id=trace_id,
                    parent_observation_id=parent_observation_id,
                    level=level,
                    metadata=metadata,
                    tags=tags
                )

            # Capture input
            if capture_input:
                try:
                    bound_args = inspect.signature(func).bind(*args, **func_kwargs)
                    bound_args.apply_defaults()
                    input_data = serialize(dict(bound_args.arguments))
                    span.set_attribute("brokle.span.input", input_data)
                except Exception as e:
                    logger.warning(f"Failed to capture input for {name}: {e}")

            # Add Brokle-specific attributes
            if routing_strategy:
                span.set_attribute("brokle.routing.strategy", routing_strategy)
            if cache_enabled is not None:
                span.set_attribute("brokle.cache.enabled", cache_enabled)
            if quality_evaluation is not None:
                span.set_attribute("brokle.evaluation.enabled", quality_evaluation)

            # Start span and execute function
            with span:
                start_time = datetime.utcnow()

                try:
                    result = await func(*args, **func_kwargs)

                    # Capture output
                    if capture_output:
                        try:
                            output_data = serialize(result)
                            span.set_attribute("brokle.span.output", output_data)
                        except Exception as e:
                            logger.warning(f"Failed to capture output for {name}: {e}")

                    # Calculate latency
                    end_time = datetime.utcnow()
                    latency_ms = int((end_time - start_time).total_seconds() * 1000)

                    # Update generation metrics if applicable
                    if isinstance(span, BrokleGeneration):
                        span.update_metrics(latency_ms=latency_ms)

                    return result

                except Exception as e:
                    # Record error
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                    raise

        return async_wrapper

    def _sync_observe(
        self,
        func: Callable,
        name: str,
        as_type: str,
        capture_input: bool,
        capture_output: bool,
        model: Optional[str],
        provider: Optional[str],
        routing_strategy: Optional[str],
        cache_enabled: Optional[bool],
        quality_evaluation: Optional[bool],
        user_id: Optional[str],
        session_id: Optional[str],
        trace_id: Optional[str],
        parent_observation_id: Optional[str],
        metadata: Dict[str, Any],
        tags: list[str],
        level: str,
        **kwargs
    ) -> Callable:
        """Sync function decorator implementation."""

        @functools.wraps(func)
        def sync_wrapper(*args, **func_kwargs):
            client = get_client()

            # Create appropriate span type
            if as_type == "generation":
                span = client.generation(
                    name=name,
                    model=model,
                    provider=provider,
                    trace_id=trace_id,
                    parent_observation_id=parent_observation_id,
                    level=level,
                    metadata=metadata,
                    tags=tags
                )
            else:
                span = client.span(
                    name=name,
                    trace_id=trace_id,
                    parent_observation_id=parent_observation_id,
                    level=level,
                    metadata=metadata,
                    tags=tags
                )

            # Capture input
            if capture_input:
                try:
                    bound_args = inspect.signature(func).bind(*args, **func_kwargs)
                    bound_args.apply_defaults()
                    input_data = serialize(dict(bound_args.arguments))
                    span.set_attribute("brokle.span.input", input_data)
                except Exception as e:
                    logger.warning(f"Failed to capture input for {name}: {e}")

            # Add Brokle-specific attributes
            if routing_strategy:
                span.set_attribute("brokle.routing.strategy", routing_strategy)
            if cache_enabled is not None:
                span.set_attribute("brokle.cache.enabled", cache_enabled)
            if quality_evaluation is not None:
                span.set_attribute("brokle.evaluation.enabled", quality_evaluation)

            # Start span and execute function
            with span:
                start_time = datetime.utcnow()

                try:
                    result = func(*args, **func_kwargs)

                    # Capture output
                    if capture_output:
                        try:
                            output_data = serialize(result)
                            span.set_attribute("brokle.span.output", output_data)
                        except Exception as e:
                            logger.warning(f"Failed to capture output for {name}: {e}")

                    # Calculate latency
                    end_time = datetime.utcnow()
                    latency_ms = int((end_time - start_time).total_seconds() * 1000)

                    # Update generation metrics if applicable
                    if isinstance(span, BrokleGeneration):
                        span.update_metrics(latency_ms=latency_ms)

                    return result

                except Exception as e:
                    # Record error
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                    raise

        return sync_wrapper


# Global decorator instance
_decorator = BrokleDecorator()


def observe(
    func: Optional[F] = None,
    *,
    name: Optional[str] = None,
    as_type: Optional[str] = None,
    capture_input: Optional[bool] = None,
    capture_output: Optional[bool] = None,
    # LLM-specific parameters
    model: Optional[str] = None,
    provider: Optional[str] = None,
    # Brokle-specific parameters
    routing_strategy: Optional[str] = None,
    cache_enabled: Optional[bool] = None,
    quality_evaluation: Optional[bool] = None,
    # Standard parameters
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    parent_observation_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[list[str]] = None,
    level: str = "DEFAULT",
    **kwargs
) -> Union[F, Callable[[F], F]]:
    """
    Global observe decorator function.

    This is the main @observe decorator that users will import and use.
    It delegates to the global decorator instance.
    """
    return _decorator.observe(
        func,
        name=name,
        as_type=as_type,
        capture_input=capture_input,
        capture_output=capture_output,
        model=model,
        provider=provider,
        routing_strategy=routing_strategy,
        cache_enabled=cache_enabled,
        quality_evaluation=quality_evaluation,
        user_id=user_id,
        session_id=session_id,
        trace_id=trace_id,
        parent_observation_id=parent_observation_id,
        metadata=metadata,
        tags=tags,
        level=level,
        **kwargs
    )