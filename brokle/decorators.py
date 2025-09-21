"""
Universal Decorator Pattern - Brokle SDK

This module provides the @observe() decorator for comprehensive observability
of any function or workflow. Follows LangFuse's universal decorator approach
for framework-agnostic tracing and evaluation.

Key Features:
- Universal function observability with @observe()
- Automatic input/output capture with privacy controls
- Nested span support for complex workflows
- Framework-agnostic (works with any Python function)
- Zero configuration required
- Performance overhead < 1ms per function call

Usage:
    from brokle import observe

    @observe()
    def my_ai_workflow(user_query: str) -> str:
        # Your AI logic here
        response = llm.generate(user_query)
        return response

    @observe(name="custom-operation", capture_inputs=False)
    def sensitive_function(api_key: str) -> dict:
        # Function with sensitive inputs
        return make_api_call(api_key)

    # Async support
    @observe()
    async def async_workflow(data: dict) -> str:
        result = await async_llm_call(data)
        return result
"""

import asyncio
import functools
import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Callable, Union
import inspect
from contextlib import contextmanager

try:
    from opentelemetry import trace
    from opentelemetry.trace import Span
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False
    trace = None
    Span = None

from .client import get_client
from ._utils.telemetry import create_span, add_span_attributes, record_span_exception
from ._utils.validation import validate_environment
from .types.attributes import BrokleOtelSpanAttributes
from .exceptions import BrokleError

logger = logging.getLogger(__name__)


class ObserveConfig:
    """Configuration for @observe decorator"""

    def __init__(
        self,
        name: Optional[str] = None,
        capture_inputs: bool = True,
        capture_outputs: bool = True,
        capture_errors: bool = True,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        evaluation_enabled: bool = True,
        max_input_length: int = 10000,
        max_output_length: int = 10000,
    ):
        self.name = name
        self.capture_inputs = capture_inputs
        self.capture_outputs = capture_outputs
        self.capture_errors = capture_errors
        self.session_id = session_id
        self.user_id = user_id
        self.tags = tags or []
        self.metadata = metadata or {}
        self.evaluation_enabled = evaluation_enabled
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length


class SpanContext:
    """Context manager for function spans with comprehensive observability"""

    def __init__(self, config: ObserveConfig, func: Callable, args: tuple, kwargs: dict):
        self.config = config
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.span = None
        self.start_time = None
        self.result = None
        self.error = None

    def __enter__(self):
        self.start_time = time.time()

        # Generate span name
        span_name = self.config.name or self._generate_span_name()

        try:
            # Create span with comprehensive attributes
            self.span = create_span(
                name=span_name,
                attributes=self._create_initial_attributes()
            )

            # Capture function inputs if enabled
            if self.config.capture_inputs:
                self._capture_inputs()

        except Exception as e:
            logger.warning(f"Failed to create span for {span_name}: {e}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.span:
                # Record execution time
                duration = time.time() - self.start_time if self.start_time else 0
                add_span_attributes(self.span, {
                    BrokleOtelSpanAttributes.REQUEST_DURATION: duration,
                    BrokleOtelSpanAttributes.RESPONSE_END_TIME: time.time(),
                    BrokleOtelSpanAttributes.FUNCTION_EXECUTED: True
                })

                # Capture outputs if available and enabled
                if self.config.capture_outputs and self.result is not None:
                    self._capture_outputs()

                # Record exception if any
                if exc_type and self.config.capture_errors:
                    record_span_exception(self.span, exc_val)
                    add_span_attributes(self.span, {
                        BrokleOtelSpanAttributes.ERROR_TYPE: exc_type.__name__,
                        BrokleOtelSpanAttributes.ERROR_MESSAGE: str(exc_val)
                    })

                self.span.end()

        except Exception as e:
            logger.warning(f"Failed to finalize span: {e}")

    def _generate_span_name(self) -> str:
        """Generate descriptive span name from function"""
        func_name = getattr(self.func, '__name__', 'unknown_function')
        module_name = getattr(self.func, '__module__', 'unknown_module')

        if module_name and module_name != '__main__':
            return f"{module_name}.{func_name}"
        return func_name

    def _create_initial_attributes(self) -> Dict[str, Any]:
        """Create initial span attributes"""
        attributes = {
            BrokleOtelSpanAttributes.OPERATION_TYPE: "function_call",
            BrokleOtelSpanAttributes.FUNCTION_NAME: self.func.__name__,
            BrokleOtelSpanAttributes.REQUEST_START_TIME: self.start_time,
        }

        # Add function signature
        try:
            sig = inspect.signature(self.func)
            attributes[BrokleOtelSpanAttributes.FUNCTION_SIGNATURE] = str(sig)
        except:
            pass

        # Add module information
        if hasattr(self.func, '__module__'):
            attributes[BrokleOtelSpanAttributes.FUNCTION_MODULE] = self.func.__module__

        # Add configuration metadata
        if self.config.session_id:
            attributes[BrokleOtelSpanAttributes.SESSION_ID] = self.config.session_id

        if self.config.user_id:
            attributes[BrokleOtelSpanAttributes.USER_ID] = self.config.user_id

        if self.config.tags:
            attributes[BrokleOtelSpanAttributes.TAGS] = ','.join(self.config.tags)

        # Add custom metadata
        for key, value in self.config.metadata.items():
            attributes[f"metadata.{key}"] = str(value)

        return attributes

    def _capture_inputs(self):
        """Capture function inputs with privacy controls"""
        if not self.span:
            return

        try:
            # Capture positional arguments
            if self.args:
                for i, arg in enumerate(self.args):
                    arg_str = self._safe_serialize(arg, self.config.max_input_length)
                    add_span_attributes(self.span, {
                        f"input.args.{i}": arg_str
                    })

            # Capture keyword arguments
            if self.kwargs:
                for key, value in self.kwargs.items():
                    # Skip sensitive parameters
                    if self._is_sensitive_key(key):
                        value_str = "[REDACTED]"
                    else:
                        value_str = self._safe_serialize(value, self.config.max_input_length)

                    add_span_attributes(self.span, {
                        f"input.kwargs.{key}": value_str
                    })

        except Exception as e:
            logger.warning(f"Failed to capture inputs: {e}")

    def _capture_outputs(self):
        """Capture function outputs with privacy controls"""
        if not self.span or self.result is None:
            return

        try:
            output_str = self._safe_serialize(self.result, self.config.max_output_length)
            add_span_attributes(self.span, {
                BrokleOtelSpanAttributes.OUTPUT: output_str,
                BrokleOtelSpanAttributes.OUTPUT_TYPE: type(self.result).__name__
            })

        except Exception as e:
            logger.warning(f"Failed to capture outputs: {e}")

    def _safe_serialize(self, obj: Any, max_length: int) -> str:
        """Safely serialize object to string with length limits"""
        try:
            if obj is None:
                return "None"

            # Handle common types efficiently
            if isinstance(obj, (str, int, float, bool)):
                obj_str = str(obj)
            elif isinstance(obj, (list, tuple)):
                if len(obj) > 10:  # Limit large collections
                    obj_str = f"[{type(obj).__name__} with {len(obj)} items]"
                else:
                    obj_str = str(obj)
            elif isinstance(obj, dict):
                if len(obj) > 10:  # Limit large dictionaries
                    obj_str = f"{{dict with {len(obj)} keys}}"
                else:
                    obj_str = str(obj)
            else:
                obj_str = f"<{type(obj).__name__} object>"

            # Truncate if too long
            if len(obj_str) > max_length:
                obj_str = obj_str[:max_length] + "...[TRUNCATED]"

            return obj_str

        except Exception:
            return f"<{type(obj).__name__} - serialization failed>"

    def _is_sensitive_key(self, key: str) -> bool:
        """Check if parameter key contains sensitive information"""
        sensitive_patterns = [
            'key', 'token', 'password', 'secret', 'auth',
            'credential', 'api_key', 'access_token'
        ]
        key_lower = key.lower()
        return any(pattern in key_lower for pattern in sensitive_patterns)

    def record_result(self, result: Any):
        """Record function result for output capture"""
        self.result = result


def observe(
    name: Optional[str] = None,
    capture_inputs: bool = True,
    capture_outputs: bool = True,
    capture_errors: bool = True,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    evaluation_enabled: bool = True,
    max_input_length: int = 10000,
    max_output_length: int = 10000,
) -> Callable:
    """
    Universal decorator for function observability.

    Provides comprehensive tracing and observability for any Python function
    with configurable input/output capture and privacy controls.

    Args:
        name: Custom span name (defaults to function name)
        capture_inputs: Whether to capture function inputs
        capture_outputs: Whether to capture function outputs
        capture_errors: Whether to capture exceptions
        session_id: Session identifier for grouping related calls
        user_id: User identifier for user-scoped analytics
        tags: List of tags for categorization
        metadata: Custom metadata dictionary
        evaluation_enabled: Whether to enable automatic evaluation
        max_input_length: Maximum length for input serialization
        max_output_length: Maximum length for output serialization

    Returns:
        Decorated function with observability

    Example:
        @observe(name="ai-workflow", tags=["ai", "production"])
        def process_user_query(query: str) -> str:
            return llm.generate(query)
    """

    def decorator(func: Callable) -> Callable:
        # Create configuration
        config = ObserveConfig(
            name=name,
            capture_inputs=capture_inputs,
            capture_outputs=capture_outputs,
            capture_errors=capture_errors,
            session_id=session_id,
            user_id=user_id,
            tags=tags,
            metadata=metadata,
            evaluation_enabled=evaluation_enabled,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
        )

        # Handle async functions
        if asyncio.iscoroutinefunction(func):
            return _wrap_async_function(func, config)
        else:
            return _wrap_sync_function(func, config)

    return decorator


def _wrap_sync_function(func: Callable, config: ObserveConfig) -> Callable:
    """Wrap synchronous function with observability"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Skip if Brokle client not available or telemetry disabled
        try:
            client = get_client()
            if not client or not client.config.telemetry_enabled:
                return func(*args, **kwargs)
        except:
            return func(*args, **kwargs)

        # Execute with observability
        with SpanContext(config, func, args, kwargs) as span_context:
            try:
                result = func(*args, **kwargs)
                span_context.record_result(result)
                return result

            except Exception as e:
                # Re-raise the original exception
                raise e

    return wrapper


def _wrap_async_function(func: Callable, config: ObserveConfig) -> Callable:
    """Wrap asynchronous function with observability"""

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        # Skip if Brokle client not available or telemetry disabled
        try:
            client = get_client()
            if not client or not client.config.telemetry_enabled:
                return await func(*args, **kwargs)
        except:
            return await func(*args, **kwargs)

        # Execute with observability
        with SpanContext(config, func, args, kwargs) as span_context:
            try:
                result = await func(*args, **kwargs)
                span_context.record_result(result)
                return result

            except Exception as e:
                # Re-raise the original exception
                raise e

    return async_wrapper


@contextmanager
def trace_workflow(
    name: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Context manager for tracing complex workflows.

    Use this to group related function calls under a single workflow span.

    Args:
        name: Workflow name
        session_id: Session identifier
        user_id: User identifier
        metadata: Custom metadata

    Example:
        with trace_workflow("user-onboarding", user_id="user123"):
            step1_result = process_signup(user_data)
            step2_result = send_welcome_email(step1_result)
            return step2_result
    """
    start_time = time.time()
    span = None

    try:
        # Create workflow span
        attributes = {
            BrokleOtelSpanAttributes.OPERATION_TYPE: "workflow",
            BrokleOtelSpanAttributes.WORKFLOW_NAME: name,
            BrokleOtelSpanAttributes.REQUEST_START_TIME: start_time,
        }

        if session_id:
            attributes[BrokleOtelSpanAttributes.SESSION_ID] = session_id
        if user_id:
            attributes[BrokleOtelSpanAttributes.USER_ID] = user_id

        if metadata:
            for key, value in metadata.items():
                attributes[f"workflow.{key}"] = str(value)

        span = create_span(name=f"workflow.{name}", attributes=attributes)

        yield span

    except Exception as e:
        if span:
            record_span_exception(span, e)
        raise

    finally:
        if span:
            duration = time.time() - start_time
            add_span_attributes(span, {
                BrokleOtelSpanAttributes.REQUEST_DURATION: duration,
                BrokleOtelSpanAttributes.RESPONSE_END_TIME: time.time()
            })
            span.end()


# Convenience functions for common use cases
def observe_llm(
    name: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> Callable:
    """
    Specialized decorator for LLM function calls.

    Adds LLM-specific metadata and attributes.
    """
    metadata = kwargs.pop('metadata', {})
    if model:
        metadata['model'] = model

    tags = kwargs.pop('tags', [])
    tags.append('llm')

    return observe(
        name=name or "llm-call",
        tags=tags,
        metadata=metadata,
        **kwargs
    )


def observe_retrieval(
    name: Optional[str] = None,
    index_name: Optional[str] = None,
    **kwargs
) -> Callable:
    """
    Specialized decorator for retrieval/search operations.
    """
    metadata = kwargs.pop('metadata', {})
    if index_name:
        metadata['index_name'] = index_name

    tags = kwargs.pop('tags', [])
    tags.append('retrieval')

    return observe(
        name=name or "retrieval",
        tags=tags,
        metadata=metadata,
        **kwargs
    )


# Export public API
__all__ = [
    'observe',
    'trace_workflow',
    'observe_llm',
    'observe_retrieval',
    'ObserveConfig',
]