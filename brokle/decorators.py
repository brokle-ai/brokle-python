"""
Decorators for automatic function tracing with OpenTelemetry.

Provides @observe decorator for zero-config instrumentation of Python functions,
including support for sync/async functions and generators.

Includes graceful degradation: tracer errors never break the application.
Pattern inspired by HoneyHive's approach.
"""

import functools
import inspect
import json
import logging
from typing import Any, Callable, Dict, List, Optional

from opentelemetry.trace import Status, StatusCode

from ._client import get_client
from .types import Attrs, SpanType
from .utils.serializer import EventSerializer, serialize_value

# Logger for tracer warnings (graceful degradation)
_logger = logging.getLogger("brokle.decorators")

# Sentinel to distinguish "user code not executed" from "user code returned None"
_NOT_EXECUTED = object()


def _build_observe_attrs(
    as_type: str,
    level: str,
    user_id: Optional[str],
    session_id: Optional[str],
    tags: Optional[List[str]],
    metadata: Optional[Dict[str, Any]],
    version: Optional[str],
    model: Optional[str],
    model_parameters: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build span attributes for @observe decorator.

    This is extracted to ensure identical attribute handling for sync, async,
    and generator wrappers.
    """
    attrs = {
        Attrs.BROKLE_SPAN_TYPE: as_type,
        Attrs.BROKLE_SPAN_LEVEL: level,
    }

    if user_id:
        attrs[Attrs.GEN_AI_REQUEST_USER] = user_id
        attrs[Attrs.USER_ID] = user_id
    if session_id:
        attrs[Attrs.SESSION_ID] = session_id
    if tags:
        attrs[Attrs.BROKLE_TRACE_TAGS] = json.dumps(tags)
    if metadata:
        attrs[Attrs.BROKLE_TRACE_METADATA] = json.dumps(metadata)
    if version:
        attrs[Attrs.BROKLE_SPAN_VERSION] = version

    if as_type == SpanType.GENERATION:
        if model:
            attrs[Attrs.GEN_AI_REQUEST_MODEL] = model
        if model_parameters:
            if "temperature" in model_parameters:
                attrs[Attrs.GEN_AI_REQUEST_TEMPERATURE] = model_parameters[
                    "temperature"
                ]
            if "max_tokens" in model_parameters:
                attrs[Attrs.GEN_AI_REQUEST_MAX_TOKENS] = model_parameters["max_tokens"]

    return attrs


def _capture_input_attrs(
    attrs: Dict[str, Any],
    func: Callable,
    args: tuple,
    kwargs: dict,
    as_type: str,
) -> None:
    """
    Capture input attributes onto attrs dict (mutates in place).

    Handles serialization errors gracefully.
    """
    try:
        input_data = _serialize_function_input(func, args, kwargs)
        input_str = json.dumps(input_data, cls=EventSerializer)
        attrs[Attrs.INPUT_VALUE] = input_str
        attrs[Attrs.INPUT_MIME_TYPE] = "application/json"
        if as_type in (SpanType.TOOL, SpanType.AGENT, SpanType.CHAIN):
            attrs[Attrs.GEN_AI_TOOL_NAME] = func.__name__
    except Exception as e:
        error_msg = f"<serialization failed: {str(e)}>"
        attrs[Attrs.INPUT_VALUE] = error_msg
        attrs[Attrs.INPUT_MIME_TYPE] = "text/plain"


def _set_output_attr(span, result: Any) -> None:
    """
    Set output attribute on span.

    Handles serialization errors gracefully.
    """
    try:
        output_data = serialize_value(result)
        output_str = json.dumps(output_data, cls=EventSerializer)
        span.set_attribute(Attrs.OUTPUT_VALUE, output_str)
        span.set_attribute(Attrs.OUTPUT_MIME_TYPE, "application/json")
    except Exception as e:
        error_msg = f"<serialization failed: {str(e)}>"
        span.set_attribute(Attrs.OUTPUT_VALUE, error_msg)
        span.set_attribute(Attrs.OUTPUT_MIME_TYPE, "text/plain")


def observe(
    *,
    name: Optional[str] = None,
    as_type: str = SpanType.SPAN,
    # Trace-level attributes
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    # Span-level attributes
    level: str = "DEFAULT",
    version: Optional[str] = None,
    model: Optional[str] = None,
    model_parameters: Optional[Dict[str, Any]] = None,
    # Input/output configuration
    capture_input: bool = True,
    capture_output: bool = True,
):
    """
    Decorator for automatic function tracing.

    Automatically creates a span for the decorated function and captures
    function arguments and return value. Supports sync functions, async
    functions, generators, and async generators.

    Args:
        name: Custom span name (default: function name)
        as_type: Span type (span, generation, event)
        session_id: Session grouping identifier
        user_id: User identifier
        tags: Categorization tags
        metadata: Custom metadata
        level: Span level (DEBUG, DEFAULT, WARNING, ERROR)
        version: Operation version
        model: LLM model (for generation type)
        model_parameters: Model parameters (for generation type)
        capture_input: Capture function arguments (default: True)
        capture_output: Capture return value (default: True)

    Returns:
        Decorated function

    Example:
        >>> @observe(name="process-request", user_id="user-123")
        ... def process(input_text: str):
        ...     return f"Processed: {input_text}"
        ...
        >>> result = process("hello")  # Automatically traced

        >>> @observe()
        ... def stream_tokens():
        ...     for token in ["Hello", " ", "World"]:
        ...         yield token
        ...
        >>> list(stream_tokens())  # Generator also traced

    Note:
        For prompt linking, use link_prompt() or update_current_span(prompt=)
        inside the function body for dynamic prompt linking at runtime.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Setup phase - graceful degradation for setup errors only
            try:
                client = get_client()
                # Check master switch and tracing config - zero overhead if disabled
                if not client.config.enabled or not client.config.tracing_enabled:
                    return func(*args, **kwargs)

                span_name = name or func.__name__
                attrs = _build_observe_attrs(
                    as_type,
                    level,
                    user_id,
                    session_id,
                    tags,
                    metadata,
                    version,
                    model,
                    model_parameters,
                )

                if capture_input:
                    _capture_input_attrs(attrs, func, args, kwargs, as_type)
            except Exception as setup_error:
                _logger.warning(
                    "Tracer setup error (continuing without tracing): %s",
                    setup_error,
                )
                return func(*args, **kwargs)

            # Track execution state to handle tracer errors correctly
            # Use sentinel to distinguish "not executed" from "returned None"
            user_result = _NOT_EXECUTED
            user_exception = None
            user_traceback = None  # Store original traceback for clean re-raise

            try:
                with client.start_as_current_span(span_name, attributes=attrs) as span:
                    # Step 1: Execute user code in isolation
                    try:
                        user_result = func(*args, **kwargs)
                    except Exception as exc:
                        user_exception = exc
                        user_traceback = exc.__traceback__  # Capture original traceback

                    # Step 2: Tracer operations (isolated - never propagate)
                    try:
                        if user_exception is not None:
                            span.set_status(Status(StatusCode.ERROR, str(user_exception)))
                            span.record_exception(user_exception)
                        else:
                            if capture_output:
                                _set_output_attr(span, user_result)
                            span.set_status(Status(StatusCode.OK))
                    except Exception as tracer_op_error:
                        _logger.warning(
                            "Tracer operation error (graceful degradation): %s",
                            tracer_op_error,
                        )

                    # Step 3: Return user outcome unchanged
                    if user_exception is not None:
                        raise user_exception.with_traceback(user_traceback)
                    return user_result

            except Exception as tracer_exc:
                # Span context manager failures (__enter__/__exit__)
                if user_exception is not None:
                    raise user_exception.with_traceback(user_traceback)
                # If user code executed, return captured result (even if None)
                if user_result is not _NOT_EXECUTED:
                    _logger.warning(
                        "Tracer span error (returning captured result): %s",
                        tracer_exc,
                    )
                    return user_result
                # Tracer error before user execution - fall back to untraced
                _logger.warning(
                    "Tracer error (continuing without tracing): %s",
                    tracer_exc,
                )
                return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Setup phase - graceful degradation for setup errors only
            try:
                client = get_client()
                # Check master switch and tracing config - zero overhead if disabled
                if not client.config.enabled or not client.config.tracing_enabled:
                    return await func(*args, **kwargs)

                span_name = name or func.__name__
                attrs = _build_observe_attrs(
                    as_type,
                    level,
                    user_id,
                    session_id,
                    tags,
                    metadata,
                    version,
                    model,
                    model_parameters,
                )

                if capture_input:
                    _capture_input_attrs(attrs, func, args, kwargs, as_type)
            except Exception as setup_error:
                _logger.warning(
                    "Tracer setup error (continuing without tracing): %s",
                    setup_error,
                )
                return await func(*args, **kwargs)

            # Track execution state to handle tracer errors correctly
            # Use sentinel to distinguish "not executed" from "returned None"
            user_result = _NOT_EXECUTED
            user_exception = None
            user_traceback = None  # Store original traceback for clean re-raise

            try:
                with client.start_as_current_span(span_name, attributes=attrs) as span:
                    # Step 1: Execute user code in isolation
                    try:
                        user_result = await func(*args, **kwargs)
                    except Exception as exc:
                        user_exception = exc
                        user_traceback = exc.__traceback__  # Capture original traceback

                    # Step 2: Tracer operations (isolated - never propagate)
                    try:
                        if user_exception is not None:
                            span.set_status(Status(StatusCode.ERROR, str(user_exception)))
                            span.record_exception(user_exception)
                        else:
                            if capture_output:
                                _set_output_attr(span, user_result)
                            span.set_status(Status(StatusCode.OK))
                    except Exception as tracer_op_error:
                        _logger.warning(
                            "Tracer operation error (graceful degradation): %s",
                            tracer_op_error,
                        )

                    # Step 3: Return user outcome unchanged
                    if user_exception is not None:
                        raise user_exception.with_traceback(user_traceback)
                    return user_result

            except Exception as tracer_exc:
                # Span context manager failures (__enter__/__exit__)
                if user_exception is not None:
                    raise user_exception.with_traceback(user_traceback)
                # If user code executed, return captured result (even if None)
                if user_result is not _NOT_EXECUTED:
                    _logger.warning(
                        "Tracer span error (returning captured result): %s",
                        tracer_exc,
                    )
                    return user_result
                # Tracer error before user execution - fall back to untraced
                _logger.warning(
                    "Tracer error (continuing without tracing): %s",
                    tracer_exc,
                )
                return await func(*args, **kwargs)

        @functools.wraps(func)
        def generator_wrapper(*args, **kwargs):
            # Setup phase - graceful degradation for setup errors only
            try:
                client = get_client()
                # Check master switch and tracing config - zero overhead if disabled
                if not client.config.enabled or not client.config.tracing_enabled:
                    yield from func(*args, **kwargs)
                    return

                span_name = name or func.__name__
                attrs = _build_observe_attrs(
                    as_type,
                    level,
                    user_id,
                    session_id,
                    tags,
                    metadata,
                    version,
                    model,
                    model_parameters,
                )

                if capture_input:
                    _capture_input_attrs(attrs, func, args, kwargs, as_type)
            except Exception as setup_error:
                _logger.warning(
                    "Tracer setup error (continuing without tracing): %s",
                    setup_error,
                )
                yield from func(*args, **kwargs)
                return

            # Track execution state to handle tracer errors correctly
            iteration_started = False
            generator_exhausted = False  # Track normal completion (even with 0 items)
            user_exception = None
            user_traceback = None  # Store original traceback for clean re-raise
            output_parts = []

            try:
                with client.start_as_current_span(span_name, attributes=attrs) as span:
                    # Step 1: Execute user code (iteration) in isolation
                    try:
                        for item in func(*args, **kwargs):
                            iteration_started = True
                            if capture_output:
                                output_parts.append(item)
                            yield item
                        generator_exhausted = True  # Loop completed normally
                    except Exception as exc:
                        user_exception = exc
                        user_traceback = exc.__traceback__  # Capture original traceback

                    # Step 2: Tracer operations (isolated - never propagate)
                    try:
                        if user_exception is not None:
                            span.set_status(Status(StatusCode.ERROR, str(user_exception)))
                            span.record_exception(user_exception)
                        else:
                            if capture_output and output_parts:
                                _set_output_attr(span, output_parts)
                            span.set_status(Status(StatusCode.OK))
                    except Exception as tracer_op_error:
                        _logger.warning(
                            "Tracer operation error (graceful degradation): %s",
                            tracer_op_error,
                        )

                    # Step 3: Re-raise user exception if any
                    if user_exception is not None:
                        raise user_exception.with_traceback(user_traceback)

            except Exception as tracer_exc:
                # Span context manager failures (__enter__/__exit__)
                if user_exception is not None:
                    raise user_exception.with_traceback(user_traceback)
                # If iteration started or completed, don't re-iterate (would duplicate side effects)
                if iteration_started or generator_exhausted:
                    _logger.warning(
                        "Tracer span error (generator already ran): %s",
                        tracer_exc,
                    )
                    return
                # Tracer error before iteration - fall back to untraced
                _logger.warning(
                    "Tracer error (continuing without tracing): %s",
                    tracer_exc,
                )
                yield from func(*args, **kwargs)

        @functools.wraps(func)
        async def async_generator_wrapper(*args, **kwargs):
            # Setup phase - graceful degradation for setup errors only
            try:
                client = get_client()
                # Check master switch and tracing config - zero overhead if disabled
                if not client.config.enabled or not client.config.tracing_enabled:
                    async for item in func(*args, **kwargs):
                        yield item
                    return

                span_name = name or func.__name__
                attrs = _build_observe_attrs(
                    as_type,
                    level,
                    user_id,
                    session_id,
                    tags,
                    metadata,
                    version,
                    model,
                    model_parameters,
                )

                if capture_input:
                    _capture_input_attrs(attrs, func, args, kwargs, as_type)
            except Exception as setup_error:
                _logger.warning(
                    "Tracer setup error (continuing without tracing): %s",
                    setup_error,
                )
                async for item in func(*args, **kwargs):
                    yield item
                return

            # Track execution state to handle tracer errors correctly
            iteration_started = False
            generator_exhausted = False  # Track normal completion (even with 0 items)
            user_exception = None
            user_traceback = None  # Store original traceback for clean re-raise
            output_parts = []

            try:
                with client.start_as_current_span(span_name, attributes=attrs) as span:
                    # Step 1: Execute user code (iteration) in isolation
                    try:
                        async for item in func(*args, **kwargs):
                            iteration_started = True
                            if capture_output:
                                output_parts.append(item)
                            yield item
                        generator_exhausted = True  # Loop completed normally
                    except Exception as exc:
                        user_exception = exc
                        user_traceback = exc.__traceback__  # Capture original traceback

                    # Step 2: Tracer operations (isolated - never propagate)
                    try:
                        if user_exception is not None:
                            span.set_status(Status(StatusCode.ERROR, str(user_exception)))
                            span.record_exception(user_exception)
                        else:
                            if capture_output and output_parts:
                                _set_output_attr(span, output_parts)
                            span.set_status(Status(StatusCode.OK))
                    except Exception as tracer_op_error:
                        _logger.warning(
                            "Tracer operation error (graceful degradation): %s",
                            tracer_op_error,
                        )

                    # Step 3: Re-raise user exception if any
                    if user_exception is not None:
                        raise user_exception.with_traceback(user_traceback)

            except Exception as tracer_exc:
                # Span context manager failures (__enter__/__exit__)
                if user_exception is not None:
                    raise user_exception.with_traceback(user_traceback)
                # If iteration started or completed, don't re-iterate (would duplicate side effects)
                if iteration_started or generator_exhausted:
                    _logger.warning(
                        "Tracer span error (generator already ran): %s",
                        tracer_exc,
                    )
                    return
                # Tracer error before iteration - fall back to untraced
                _logger.warning(
                    "Tracer error (continuing without tracing): %s",
                    tracer_exc,
                )
                async for item in func(*args, **kwargs):
                    yield item

        # Select appropriate wrapper based on function type
        if inspect.isasyncgenfunction(func):
            return async_generator_wrapper
        elif inspect.isgeneratorfunction(func):
            return generator_wrapper
        elif inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def _serialize_function_input(
    func: Callable, args: tuple, kwargs: dict
) -> Dict[str, Any]:
    """
    Serialize function input arguments.

    Uses the robust EventSerializer for comprehensive type handling including:
    - Pydantic models, dataclasses, numpy arrays
    - Datetime, UUID, Path objects
    - Circular reference detection
    - Large integer handling (JS safe range)

    Args:
        func: Function being decorated
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Serializable dictionary of arguments
    """
    sig = inspect.signature(func)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()

    serialized = {}
    for param_name, value in bound_args.arguments.items():
        serialized[param_name] = serialize_value(value)

    return serialized
