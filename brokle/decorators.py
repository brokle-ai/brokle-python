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
            # Get or create client
            client = get_client()

            # Determine span name
            span_name = name or func.__name__

            # Build initial attributes
            attrs = {
                Attrs.BROKLE_SPAN_TYPE: as_type,
                Attrs.BROKLE_SPAN_LEVEL: level,
            }

            # Add trace-level attributes
            if user_id:
                attrs[Attrs.GEN_AI_REQUEST_USER] = user_id
                attrs[Attrs.USER_ID] = user_id  # Filterable
            if session_id:
                attrs[Attrs.SESSION_ID] = session_id
            if tags:
                attrs[Attrs.BROKLE_TRACE_TAGS] = json.dumps(tags)
                attrs[Attrs.TAGS] = json.dumps(tags)  # Filterable
            if metadata:
                attrs[Attrs.BROKLE_TRACE_METADATA] = json.dumps(metadata)
                attrs[Attrs.METADATA] = json.dumps(metadata)  # Filterable
            if version:
                attrs[Attrs.BROKLE_VERSION] = version

            # Add generation-specific attributes
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

            # Capture input if enabled
            if capture_input:
                # Serialize function arguments
                try:
                    input_data = _serialize_function_input(func, args, kwargs)
                    input_str = json.dumps(
                        input_data, default=str
                    )  # default=str handles non-serializable
                    attrs[Attrs.INPUT_VALUE] = input_str
                    attrs[Attrs.INPUT_MIME_TYPE] = (
                        "application/json"  # Function args always JSON
                    )
                except Exception as e:
                    # If serialization fails, store error message
                    error_msg = f"<serialization failed: {str(e)}>"
                    attrs[Attrs.INPUT_VALUE] = error_msg
                    attrs[Attrs.INPUT_MIME_TYPE] = "text/plain"

            # Create span using client
            with client.start_as_current_span(span_name, attributes=attrs) as span:
                try:
                    # Execute function
                    result = func(*args, **kwargs)

                    # Capture output if enabled
                    if capture_output:
                        try:
                            output_data = _serialize_value(result)
                            output_str = json.dumps(output_data, default=str)
                            span.set_attribute(Attrs.OUTPUT_VALUE, output_str)
                            span.set_attribute(
                                Attrs.OUTPUT_MIME_TYPE, "application/json"
                            )
                        except Exception as e:
                            error_msg = f"<serialization failed: {str(e)}>"
                            span.set_attribute(Attrs.OUTPUT_VALUE, error_msg)
                            span.set_attribute(Attrs.OUTPUT_MIME_TYPE, "text/plain")

                    # Mark span as successful
                    span.set_status(Status(StatusCode.OK))

                    return result

                except Exception as e:
                    # Record exception
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get or create client
            client = get_client()

            # Determine span name
            span_name = name or func.__name__

            # Build initial attributes (same as sync)
            attrs = {
                Attrs.BROKLE_SPAN_TYPE: as_type,
                Attrs.BROKLE_SPAN_LEVEL: level,
            }

            # Add trace-level attributes
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

            # Add generation-specific attributes
            if as_type == SpanType.GENERATION:
                if model:
                    attrs[Attrs.GEN_AI_REQUEST_MODEL] = model

            # Capture input if enabled
            if capture_input:
                try:
                    input_data = _serialize_function_input(func, args, kwargs)
                    attrs[Attrs.BROKLE_TRACE_INPUT] = json.dumps(input_data)
                except Exception as e:
                    attrs["brokle.input.error"] = str(e)

            # Create span using client
            with client.start_as_current_span(span_name, attributes=attrs) as span:
                try:
                    # Execute async function
                    result = await func(*args, **kwargs)

                    # Capture output if enabled
                    if capture_output:
                        try:
                            output_data = _serialize_value(result)
                            output_str = json.dumps(output_data, default=str)
                            span.set_attribute(Attrs.OUTPUT_VALUE, output_str)
                            span.set_attribute(
                                Attrs.OUTPUT_MIME_TYPE, "application/json"
                            )
                        except Exception as e:
                            error_msg = f"<serialization failed: {str(e)}>"
                            span.set_attribute(Attrs.OUTPUT_VALUE, error_msg)
                            span.set_attribute(Attrs.OUTPUT_MIME_TYPE, "text/plain")

                    # Mark span as successful
                    span.set_status(Status(StatusCode.OK))

                    return result

                except Exception as e:
                    # Record exception
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        # Return appropriate wrapper based on function type
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

    Args:
        func: Function being decorated
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Serializable dictionary of arguments
    """
    # Get function signature
    sig = inspect.signature(func)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()

    # Serialize each argument
    serialized = {}
    for param_name, value in bound_args.arguments.items():
        serialized[param_name] = _serialize_value(value)

    return serialized


def _serialize_value(value: Any) -> Any:
    """
    Serialize a value for JSON encoding.

    Handles common types and provides fallback for complex objects.

    Args:
        value: Value to serialize

    Returns:
        JSON-serializable value
    """
    # Handle None
    if value is None:
        return None

    # Handle primitives
    if isinstance(value, (str, int, float, bool)):
        return value

    # Handle lists
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]

    # Handle dicts
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}

    # Handle Pydantic models
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)

    # Handle dataclasses
    if hasattr(value, "__dataclass_fields__"):
        import dataclasses

        return dataclasses.asdict(value)

    # Fallback to string representation
    return str(value)
