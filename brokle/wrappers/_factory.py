"""
Unified wrapper factory for LLM SDK instrumentation.

Provides a single factory function that generates both sync and async wrappers,
eliminating code duplication and ensuring attribute parity between code paths.

Based on the LangSmith SDK pattern for auto-detecting sync/async methods.
"""

import functools
import inspect
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Type, TypeVar, Union

from opentelemetry.trace import Status, StatusCode

from .._client import get_client
from ..streaming import StreamingAccumulator
from ..streaming.wrappers import BrokleAsyncStreamWrapper, BrokleStreamWrapper
from ..types import OperationType
from ._common import add_prompt_attributes, extract_brokle_options

if TYPE_CHECKING:
    from ..client import Brokle

C = TypeVar("C")  # Client type


def is_async_method(func: Callable) -> bool:
    """
    Detect if a function is async, including wrapped functions.

    Handles:
    - Native async functions (async def)
    - Wrapped async functions (via functools.wraps or similar)
    - Bound methods with async underlying functions

    Args:
        func: Function to check

    Returns:
        True if function is async, False otherwise
    """
    if inspect.iscoroutinefunction(func):
        return True

    # Check for wrapped async functions
    if hasattr(func, "__wrapped__"):
        return inspect.iscoroutinefunction(func.__wrapped__)

    # Check for bound methods
    if hasattr(func, "__func__"):
        return inspect.iscoroutinefunction(func.__func__)

    return False


def create_wrapper(
    original_method: Callable,
    build_attrs: Callable[[Dict[str, Any]], Dict[str, Any]],
    extract_response: Callable[[Any, Any, float], None],
    get_span_name: Callable[[Dict[str, Any]], str],
    get_model: Callable[[Dict[str, Any]], str],
    is_stream: Callable[[Dict[str, Any]], bool],
    stream_wrapper_class: Optional[Type] = None,
    async_stream_wrapper_class: Optional[Type] = None,
) -> Callable:
    """
    Create a sync or async wrapper based on the original method.

    This is the core factory function that generates instrumented wrappers
    with automatic sync/async detection.

    Args:
        original_method: The original LLM SDK method to wrap
        build_attrs: Function to build span attributes from kwargs
        extract_response: Function to extract response attributes onto span
        get_span_name: Function to generate span name from kwargs
        get_model: Function to extract model name from kwargs
        is_stream: Function to check if request is streaming
        stream_wrapper_class: Optional custom sync stream wrapper
        async_stream_wrapper_class: Optional custom async stream wrapper

    Returns:
        Wrapped function (sync or async based on original)
    """
    # Default to standard stream wrappers if not provided
    sync_stream_cls = stream_wrapper_class or BrokleStreamWrapper
    async_stream_cls = async_stream_wrapper_class or BrokleAsyncStreamWrapper

    @functools.wraps(original_method)
    def sync_wrapper(*args, **kwargs):
        """Synchronous wrapper with automatic tracing."""
        # Extract brokle_options before processing
        kwargs, brokle_opts = extract_brokle_options(kwargs)

        brokle_client = get_client()
        if not brokle_client.config.enabled:
            return original_method(*args, **kwargs)

        # Build span attributes
        attrs = build_attrs(kwargs)
        add_prompt_attributes(attrs, brokle_opts)

        span_name = get_span_name(kwargs)
        streaming = is_stream(kwargs)

        if streaming:
            return _handle_sync_streaming(
                brokle_client,
                original_method,
                args,
                kwargs,
                span_name,
                attrs,
                sync_stream_cls,
            )
        else:
            return _handle_sync_response(
                brokle_client,
                original_method,
                args,
                kwargs,
                span_name,
                attrs,
                extract_response,
            )

    @functools.wraps(original_method)
    async def async_wrapper(*args, **kwargs):
        """Asynchronous wrapper with automatic tracing."""
        # Extract brokle_options before processing
        kwargs, brokle_opts = extract_brokle_options(kwargs)

        brokle_client = get_client()
        if not brokle_client.config.enabled:
            return await original_method(*args, **kwargs)

        # Build span attributes
        attrs = build_attrs(kwargs)
        add_prompt_attributes(attrs, brokle_opts)

        span_name = get_span_name(kwargs)
        streaming = is_stream(kwargs)

        if streaming:
            return await _handle_async_streaming(
                brokle_client,
                original_method,
                args,
                kwargs,
                span_name,
                attrs,
                async_stream_cls,
            )
        else:
            return await _handle_async_response(
                brokle_client,
                original_method,
                args,
                kwargs,
                span_name,
                attrs,
                extract_response,
            )

    # Return appropriate wrapper based on original method type
    return async_wrapper if is_async_method(original_method) else sync_wrapper


def _handle_sync_response(
    brokle_client: "Brokle",
    original_method: Callable,
    args: tuple,
    kwargs: dict,
    span_name: str,
    attrs: Dict[str, Any],
    extract_response: Callable,
) -> Any:
    """Handle synchronous non-streaming response."""
    with brokle_client.start_as_current_span(span_name, attributes=attrs) as span:
        try:
            start_time = time.time()
            response = original_method(*args, **kwargs)
            latency_ms = (time.time() - start_time) * 1000
            extract_response(span, response, latency_ms)
            span.set_status(Status(StatusCode.OK))
            return response
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


async def _handle_async_response(
    brokle_client: "Brokle",
    original_method: Callable,
    args: tuple,
    kwargs: dict,
    span_name: str,
    attrs: Dict[str, Any],
    extract_response: Callable,
) -> Any:
    """Handle asynchronous non-streaming response."""
    with brokle_client.start_as_current_span(span_name, attributes=attrs) as span:
        try:
            start_time = time.time()
            response = await original_method(*args, **kwargs)
            latency_ms = (time.time() - start_time) * 1000
            extract_response(span, response, latency_ms)
            span.set_status(Status(StatusCode.OK))
            return response
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


def _handle_sync_streaming(
    brokle_client: "Brokle",
    original_method: Callable,
    args: tuple,
    kwargs: dict,
    span_name: str,
    attrs: Dict[str, Any],
    stream_wrapper_class: Type,
) -> Any:
    """Handle synchronous streaming response."""
    # Span will be ended by stream wrapper
    tracer = brokle_client._tracer
    span = tracer.start_span(span_name, attributes=attrs)

    try:
        start_time = time.perf_counter()
        response = original_method(*args, **kwargs)
        accumulator = StreamingAccumulator(start_time)
        return stream_wrapper_class(response, span, accumulator)
    except BaseException as e:
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.record_exception(e)
        span.end()
        raise


async def _handle_async_streaming(
    brokle_client: "Brokle",
    original_method: Callable,
    args: tuple,
    kwargs: dict,
    span_name: str,
    attrs: Dict[str, Any],
    stream_wrapper_class: Type,
) -> Any:
    """Handle asynchronous streaming response."""
    # Span will be ended by stream wrapper
    tracer = brokle_client._tracer
    span = tracer.start_span(span_name, attributes=attrs)

    try:
        start_time = time.perf_counter()
        response = await original_method(*args, **kwargs)
        accumulator = StreamingAccumulator(start_time)
        return stream_wrapper_class(response, span, accumulator)
    except BaseException as e:
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.record_exception(e)
        span.end()
        raise


def wrap_client_method(
    client: C,
    method_path: str,
    build_attrs: Callable[[Dict[str, Any]], Dict[str, Any]],
    extract_response: Callable[[Any, Any, float], None],
    get_span_name: Callable[[Dict[str, Any]], str],
    get_model: Callable[[Dict[str, Any]], str],
    is_stream: Callable[[Dict[str, Any]], bool],
    stream_wrapper_class: Optional[Type] = None,
    async_stream_wrapper_class: Optional[Type] = None,
) -> None:
    """
    Wrap a nested client method with instrumentation.

    This is a helper to wrap methods like `client.chat.completions.create`
    where the method is accessed through a chain of attributes.

    Args:
        client: The SDK client instance
        method_path: Dot-separated path to method (e.g., "chat.completions.create")
        build_attrs: Function to build span attributes from kwargs
        extract_response: Function to extract response attributes onto span
        get_span_name: Function to generate span name from kwargs
        get_model: Function to extract model name from kwargs
        is_stream: Function to check if request is streaming
        stream_wrapper_class: Optional custom sync stream wrapper
        async_stream_wrapper_class: Optional custom async stream wrapper

    Example:
        wrap_client_method(
            client,
            "chat.completions.create",
            build_attrs=lambda kw: {...},
            extract_response=extract_openai_response,
            get_span_name=lambda kw: f"chat {kw.get('model')}",
            get_model=lambda kw: kw.get("model", "unknown"),
            is_stream=lambda kw: kw.get("stream", False),
        )
    """
    parts = method_path.split(".")
    obj = client

    # Navigate to parent object
    for part in parts[:-1]:
        obj = getattr(obj, part)

    # Get the method to wrap
    method_name = parts[-1]
    original_method = getattr(obj, method_name)

    # Create and attach wrapper
    wrapper = create_wrapper(
        original_method=original_method,
        build_attrs=build_attrs,
        extract_response=extract_response,
        get_span_name=get_span_name,
        get_model=get_model,
        is_stream=is_stream,
        stream_wrapper_class=stream_wrapper_class,
        async_stream_wrapper_class=async_stream_wrapper_class,
    )

    setattr(obj, method_name, wrapper)
