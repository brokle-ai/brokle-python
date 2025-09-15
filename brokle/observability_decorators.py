"""
Enhanced observability decorators for Brokle SDK.

This module provides enhanced @observe decorators that integrate directly
with the Brokle observability backend for comprehensive LLM observability.
"""

import asyncio
import functools
import inspect
import logging
import time
import uuid
from typing import Any, Callable, Dict, Optional, TypeVar, Union, List
from datetime import datetime

from .config import get_config
from .client import get_client
from .core.serialization import serialize

logger = logging.getLogger(__name__)

# Type variables
F = TypeVar('F', bound=Callable[..., Any])


class ObservabilityDecorator:
    """Enhanced implementation of the @observe decorator for Brokle Platform."""

    def __init__(self):
        self.config = get_config()
        self._client = None

    @property
    def client(self):
        """Get or create Brokle client for observability."""
        if self._client is None:
            self._client = get_client()
        return self._client

    def observe(
        self,
        func: Optional[F] = None,
        *,
        name: Optional[str] = None,
        as_type: str = "span",
        capture_input: bool = True,
        capture_output: bool = True,
        capture_timing: bool = True,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        parent_observation_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        version: Optional[str] = None,
        level: str = "DEFAULT",
        auto_complete: bool = True,
        **kwargs
    ) -> Union[F, Callable[[F], F]]:
        """
        Enhanced decorator for observing function calls with Brokle observability backend.

        Args:
            func: The function to decorate
            name: Custom name for the observation
            as_type: Type of observation ('llm', 'span', 'event', 'generation', etc.)
            capture_input: Whether to capture function input
            capture_output: Whether to capture function output
            capture_timing: Whether to capture execution timing
            user_id: User ID for tracking
            session_id: Session ID for tracking
            trace_id: Specific trace ID to use
            parent_observation_id: Parent observation ID for nesting
            tags: Tags for the observation
            metadata: Additional metadata
            model: Model name for LLM observations
            provider: Provider name for LLM observations
            version: Version for the observation
            level: Observation level (DEBUG, INFO, WARN, ERROR, DEFAULT)
            auto_complete: Whether to automatically complete the observation
            **kwargs: Additional attributes

        Returns:
            Decorated function with enhanced observability
        """
        def decorator(func: F) -> F:
            if asyncio.iscoroutinefunction(func):
                return self._async_observe(
                    func,
                    name=name or func.__name__,
                    as_type=as_type,
                    capture_input=capture_input,
                    capture_output=capture_output,
                    capture_timing=capture_timing,
                    user_id=user_id,
                    session_id=session_id,
                    trace_id=trace_id,
                    parent_observation_id=parent_observation_id,
                    tags=tags or {},
                    metadata=metadata or {},
                    model=model,
                    provider=provider,
                    version=version,
                    level=level,
                    auto_complete=auto_complete,
                    **kwargs
                )
            else:
                return self._sync_observe(
                    func,
                    name=name or func.__name__,
                    as_type=as_type,
                    capture_input=capture_input,
                    capture_output=capture_output,
                    capture_timing=capture_timing,
                    user_id=user_id,
                    session_id=session_id,
                    trace_id=trace_id,
                    parent_observation_id=parent_observation_id,
                    tags=tags or {},
                    metadata=metadata or {},
                    model=model,
                    provider=provider,
                    version=version,
                    level=level,
                    auto_complete=auto_complete,
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
        capture_timing: bool,
        user_id: Optional[str],
        session_id: Optional[str],
        trace_id: Optional[str],
        parent_observation_id: Optional[str],
        tags: Dict[str, Any],
        metadata: Dict[str, Any],
        model: Optional[str],
        provider: Optional[str],
        version: Optional[str],
        level: str,
        auto_complete: bool,
        **kwargs
    ) -> Callable:
        """Async function decorator implementation."""

        @functools.wraps(func)
        async def wrapper(*args, **func_kwargs):
            # Ensure we have a trace
            current_trace_id = trace_id or self._get_or_create_trace(
                name=f"{func.__module__}.{func.__name__}",
                user_id=user_id,
                session_id=session_id,
                metadata={"function": func.__name__, "module": func.__module__}
            )

            # Create observation
            observation_data = {
                "trace_id": current_trace_id,
                "name": name,
                "observation_type": as_type,
                "parent_observation_id": parent_observation_id,
                "level": level,
                "model": model,
                "provider": provider,
                "version": version,
                "metadata": {
                    **metadata,
                    "function": func.__name__,
                    "module": func.__module__,
                    "signature": str(inspect.signature(func))
                }
            }

            # Capture input if enabled
            if capture_input:
                try:
                    # Serialize function arguments
                    bound_args = inspect.signature(func).bind(*args, **func_kwargs)
                    bound_args.apply_defaults()
                    observation_data["input_data"] = serialize(dict(bound_args.arguments))
                except Exception as e:
                    logger.warning(f"Failed to capture input for {name}: {e}")
                    observation_data["input_data"] = {"error": f"Failed to serialize input: {e}"}

            start_time = datetime.utcnow()
            observation_data["start_time"] = start_time

            # Create the observation
            try:
                observation = await self.client.observability.create_observation(
                    **observation_data
                )
                observation_id = observation.id
            except Exception as e:
                logger.error(f"Failed to create observation for {name}: {e}")
                observation_id = None

            # Execute the function
            start_perf = time.perf_counter()
            try:
                result = await func(*args, **func_kwargs)
                status = "success"
                error_message = None
            except Exception as e:
                result = None
                status = "error"
                error_message = str(e)
                raise
            finally:
                # Complete the observation if auto_complete is enabled
                if observation_id and auto_complete:
                    try:
                        end_time = datetime.utcnow()
                        completion_data = {
                            "end_time": end_time,
                            "status_message": error_message or status
                        }

                        # Capture output if enabled and successful
                        if capture_output and result is not None:
                            try:
                                completion_data["output_data"] = serialize(result)
                            except Exception as e:
                                logger.warning(f"Failed to capture output for {name}: {e}")
                                completion_data["output_data"] = {"error": f"Failed to serialize output: {e}"}

                        # Capture timing if enabled
                        if capture_timing:
                            latency_ms = int((time.perf_counter() - start_perf) * 1000)
                            completion_data["latency_ms"] = latency_ms

                        await self.client.observability.complete_observation(
                            observation_id,
                            **completion_data
                        )
                    except Exception as e:
                        logger.error(f"Failed to complete observation for {name}: {e}")

            return result

        return wrapper

    def _sync_observe(
        self,
        func: Callable,
        name: str,
        as_type: str,
        capture_input: bool,
        capture_output: bool,
        capture_timing: bool,
        user_id: Optional[str],
        session_id: Optional[str],
        trace_id: Optional[str],
        parent_observation_id: Optional[str],
        tags: Dict[str, Any],
        metadata: Dict[str, Any],
        model: Optional[str],
        provider: Optional[str],
        version: Optional[str],
        level: str,
        auto_complete: bool,
        **kwargs
    ) -> Callable:
        """Sync function decorator implementation."""

        @functools.wraps(func)
        def wrapper(*args, **func_kwargs):
            # Ensure we have a trace
            current_trace_id = trace_id or self._get_or_create_trace_sync(
                name=f"{func.__module__}.{func.__name__}",
                user_id=user_id,
                session_id=session_id,
                metadata={"function": func.__name__, "module": func.__module__}
            )

            # Create observation
            observation_data = {
                "trace_id": current_trace_id,
                "name": name,
                "observation_type": as_type,
                "parent_observation_id": parent_observation_id,
                "level": level,
                "model": model,
                "provider": provider,
                "version": version,
                "metadata": {
                    **metadata,
                    "function": func.__name__,
                    "module": func.__module__,
                    "signature": str(inspect.signature(func))
                }
            }

            # Capture input if enabled
            if capture_input:
                try:
                    # Serialize function arguments
                    bound_args = inspect.signature(func).bind(*args, **func_kwargs)
                    bound_args.apply_defaults()
                    observation_data["input_data"] = serialize(dict(bound_args.arguments))
                except Exception as e:
                    logger.warning(f"Failed to capture input for {name}: {e}")
                    observation_data["input_data"] = {"error": f"Failed to serialize input: {e}"}

            start_time = datetime.utcnow()
            observation_data["start_time"] = start_time

            # Create the observation
            try:
                observation = self.client.observability.create_observation_sync(
                    **observation_data
                )
                observation_id = observation.id
            except Exception as e:
                logger.error(f"Failed to create observation for {name}: {e}")
                observation_id = None

            # Execute the function
            start_perf = time.perf_counter()
            try:
                result = func(*args, **func_kwargs)
                status = "success"
                error_message = None
            except Exception as e:
                result = None
                status = "error"
                error_message = str(e)
                raise
            finally:
                # Complete the observation if auto_complete is enabled
                if observation_id and auto_complete:
                    try:
                        end_time = datetime.utcnow()
                        completion_data = {
                            "end_time": end_time,
                            "status_message": error_message or status
                        }

                        # Capture output if enabled and successful
                        if capture_output and result is not None:
                            try:
                                completion_data["output_data"] = serialize(result)
                            except Exception as e:
                                logger.warning(f"Failed to capture output for {name}: {e}")
                                completion_data["output_data"] = {"error": f"Failed to serialize output: {e}"}

                        # Capture timing if enabled
                        if capture_timing:
                            latency_ms = int((time.perf_counter() - start_perf) * 1000)
                            completion_data["latency_ms"] = latency_ms

                        self.client.observability.complete_observation_sync(
                            observation_id,
                            **completion_data
                        )
                    except Exception as e:
                        logger.error(f"Failed to complete observation for {name}: {e}")

            return result

        return wrapper

    def _get_or_create_trace(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Get or create a trace asynchronously."""
        try:
            # For now, create a new trace. In the future, we could implement
            # trace context management to reuse existing traces
            trace = asyncio.run(self.client.observability.create_trace(
                name=name,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {}
            ))
            return trace.id
        except Exception as e:
            logger.error(f"Failed to create trace: {e}")
            # Return a fallback trace ID
            return str(uuid.uuid4())

    def _get_or_create_trace_sync(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Get or create a trace synchronously."""
        try:
            # For now, create a new trace. In the future, we could implement
            # trace context management to reuse existing traces
            trace = self.client.observability.create_trace_sync(
                name=name,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {}
            )
            return trace.id
        except Exception as e:
            logger.error(f"Failed to create trace: {e}")
            # Return a fallback trace ID
            return str(uuid.uuid4())


# Global decorator instance
_observability_decorator = ObservabilityDecorator()


def observe_enhanced(
    func: Optional[F] = None,
    *,
    name: Optional[str] = None,
    as_type: str = "span",
    capture_input: bool = True,
    capture_output: bool = True,
    capture_timing: bool = True,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    parent_observation_id: Optional[str] = None,
    tags: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    version: Optional[str] = None,
    level: str = "DEFAULT",
    auto_complete: bool = True,
    **kwargs
) -> Union[F, Callable[[F], F]]:
    """
    Enhanced @observe decorator that integrates with Brokle observability backend.

    This decorator provides comprehensive observability for any function, automatically
    creating traces and observations in the Brokle platform for monitoring and analytics.

    Args:
        func: The function to decorate
        name: Custom name for the observation
        as_type: Type of observation ('llm', 'span', 'event', 'generation', etc.)
        capture_input: Whether to capture function input
        capture_output: Whether to capture function output
        capture_timing: Whether to capture execution timing
        user_id: User ID for tracking
        session_id: Session ID for tracking
        trace_id: Specific trace ID to use
        parent_observation_id: Parent observation ID for nesting
        tags: Tags for the observation
        metadata: Additional metadata
        model: Model name for LLM observations
        provider: Provider name for LLM observations
        version: Version for the observation
        level: Observation level (DEBUG, INFO, WARN, ERROR, DEFAULT)
        auto_complete: Whether to automatically complete the observation
        **kwargs: Additional attributes

    Returns:
        Decorated function with enhanced observability

    Examples:
        Basic usage:
        ```python
        @observe_enhanced
        def my_function(x, y):
            return x + y
        ```

        LLM function observability:
        ```python
        @observe_enhanced(
            as_type="llm",
            model="gpt-4",
            provider="openai",
            capture_timing=True
        )
        async def generate_text(prompt):
            # LLM call logic
            return result
        ```

        Nested observations:
        ```python
        @observe_enhanced(as_type="span", name="main_workflow")
        def main_workflow():
            step1()
            step2()

        @observe_enhanced(as_type="span", name="step_1")
        def step1():
            # Implementation
            pass
        ```
    """
    return _observability_decorator.observe(
        func,
        name=name,
        as_type=as_type,
        capture_input=capture_input,
        capture_output=capture_output,
        capture_timing=capture_timing,
        user_id=user_id,
        session_id=session_id,
        trace_id=trace_id,
        parent_observation_id=parent_observation_id,
        tags=tags,
        metadata=metadata,
        model=model,
        provider=provider,
        version=version,
        level=level,
        auto_complete=auto_complete,
        **kwargs
    )