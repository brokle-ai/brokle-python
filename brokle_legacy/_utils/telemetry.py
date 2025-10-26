"""
Telemetry utilities for Brokle SDK.

Clean, reusable telemetry functions extracted and refactored from
the old integration framework.
"""

import logging
from typing import Any, Dict, Optional

try:
    from opentelemetry import trace
    from opentelemetry.trace import Span

    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False
    trace = None
    Span = None

logger = logging.getLogger(__name__)


def create_span(
    name: str, attributes: Optional[Dict[str, Any]] = None
) -> Optional[Any]:
    """
    Create a new telemetry span.

    Args:
        name: Span name
        attributes: Optional span attributes

    Returns:
        Span object if telemetry available, None otherwise
    """
    if not HAS_OTEL:
        return None

    try:
        tracer = trace.get_tracer(__name__)
        span = tracer.start_span(name)

        if attributes:
            add_span_attributes(span, attributes)

        return span
    except Exception as e:
        logger.warning(f"Failed to create span {name}: {e}")
        return None


def add_span_attributes(span: Any, attributes: Dict[str, Any]) -> None:
    """
    Add attributes to a span.

    Args:
        span: Span object
        attributes: Dictionary of attributes to add
    """
    if not span or not attributes:
        return

    try:
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, value)
    except Exception as e:
        logger.warning(f"Failed to add span attributes: {e}")


def record_span_exception(span: Any, exception: Exception) -> None:
    """
    Record an exception on a span.

    Args:
        span: Span object
        exception: Exception to record
    """
    if not span or not exception:
        return

    try:
        span.record_exception(exception)
    except Exception as e:
        logger.warning(f"Failed to record span exception: {e}")


def end_span(span: Any) -> None:
    """
    End a span.

    Args:
        span: Span object to end
    """
    if not span:
        return

    try:
        span.end()
    except Exception as e:
        logger.warning(f"Failed to end span: {e}")


def get_current_span() -> Optional[Any]:
    """
    Get the current active span.

    Returns:
        Current span if available, None otherwise
    """
    if not HAS_OTEL:
        return None

    try:
        return trace.get_current_span()
    except Exception as e:
        logger.warning(f"Failed to get current span: {e}")
        return None


def set_span_status(
    span: Any, status_code: str, description: Optional[str] = None
) -> None:
    """
    Set span status.

    Args:
        span: Span object
        status_code: Status code
        description: Optional status description
    """
    if not span:
        return

    try:
        from opentelemetry.trace import Status, StatusCode

        if status_code == "OK":
            status = Status(StatusCode.OK, description)
        elif status_code == "ERROR":
            status = Status(StatusCode.ERROR, description)
        else:
            status = Status(StatusCode.UNSET, description)

        span.set_status(status)
    except Exception as e:
        logger.warning(f"Failed to set span status: {e}")
