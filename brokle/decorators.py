"""
Decorators for automatic function tracing with OpenTelemetry.

Provides @observe decorator for zero-config instrumentation of Python functions.
"""

import functools
import inspect
import json
from typing import Any, Callable, Dict, List, Optional

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from .client import get_client
from .types import Attrs, SpanType
from .utils.serializer import EventSerializer, serialize_value, serialize_function_args


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
    function arguments and return value.

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
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            client = get_client()
            span_name = name or func.__name__

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
                attrs[Attrs.TAGS] = json.dumps(tags)
            if metadata:
                attrs[Attrs.BROKLE_TRACE_METADATA] = json.dumps(metadata)
                attrs[Attrs.METADATA] = json.dumps(metadata)
            if version:
                attrs[Attrs.BROKLE_VERSION] = version

            if as_type == SpanType.GENERATION:
                if model:
                    attrs[Attrs.GEN_AI_REQUEST_MODEL] = model
                if model_parameters:
                    if "temperature" in model_parameters:
                        attrs[Attrs.GEN_AI_REQUEST_TEMPERATURE] = model_parameters[
                            "temperature"
                        ]
                    if "max_tokens" in model_parameters:
                        attrs[Attrs.GEN_AI_REQUEST_MAX_TOKENS] = model_parameters[
                            "max_tokens"
                        ]

            if capture_input:
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

            with client.start_as_current_span(span_name, attributes=attrs) as span:
                try:
                    result = func(*args, **kwargs)

                    if capture_output:
                        try:
                            output_data = serialize_value(result)
                            output_str = json.dumps(output_data, cls=EventSerializer)
                            span.set_attribute(Attrs.OUTPUT_VALUE, output_str)
                            span.set_attribute(
                                Attrs.OUTPUT_MIME_TYPE, "application/json"
                            )
                        except Exception as e:
                            error_msg = f"<serialization failed: {str(e)}>"
                            span.set_attribute(Attrs.OUTPUT_VALUE, error_msg)
                            span.set_attribute(Attrs.OUTPUT_MIME_TYPE, "text/plain")

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            client = get_client()
            span_name = name or func.__name__

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
                attrs[Attrs.BROKLE_VERSION] = version

            if as_type == SpanType.GENERATION:
                if model:
                    attrs[Attrs.GEN_AI_REQUEST_MODEL] = model

            if capture_input:
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

            with client.start_as_current_span(span_name, attributes=attrs) as span:
                try:
                    result = await func(*args, **kwargs)

                    if capture_output:
                        try:
                            output_data = serialize_value(result)
                            output_str = json.dumps(output_data, cls=EventSerializer)
                            span.set_attribute(Attrs.OUTPUT_VALUE, output_str)
                            span.set_attribute(
                                Attrs.OUTPUT_MIME_TYPE, "application/json"
                            )
                        except Exception as e:
                            error_msg = f"<serialization failed: {str(e)}>"
                            span.set_attribute(Attrs.OUTPUT_VALUE, error_msg)
                            span.set_attribute(Attrs.OUTPUT_MIME_TYPE, "text/plain")

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        if inspect.iscoroutinefunction(func):
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
