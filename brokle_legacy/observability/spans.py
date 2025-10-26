"""
Span management for observability.

Provides span creation and management for Pattern 1/2 compatibility.
"""

import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from .context import get_client


class BrokleSpan(BaseModel):
    """
    Brokle span for observability.

    Maintains compatibility with existing Pattern 1/2 code.
    """

    span_id: str
    trace_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "started"  # started, completed, error
    attributes: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    tags: List[str] = []

    def __init__(self, **data):
        if "span_id" not in data:
            data["span_id"] = f"span_{uuid.uuid4().hex[:16]}"
        if "start_time" not in data:
            data["start_time"] = datetime.now(timezone.utc)
        super().__init__(**data)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set span attribute."""
        self.attributes[key] = value

    def set_status(self, status: str, description: Optional[str] = None) -> None:
        """Set span status."""
        self.status = status
        if description:
            self.attributes["status_description"] = description

    def add_tag(self, tag: str) -> None:
        """Add tag to span."""
        if tag not in self.tags:
            self.tags.append(tag)

    def finish(self) -> None:
        """Finish the span."""
        self.end_time = datetime.now(timezone.utc)
        if self.status == "started":
            self.status = "completed"

    def end(self) -> None:
        """End the span (alias for finish)."""
        self.finish()

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "attributes": self.attributes,
            "metadata": self.metadata,
            "tags": self.tags,
        }


class BrokleGeneration(BrokleSpan):
    """
    Brokle generation span for LLM calls.

    Extends BrokleSpan with generation-specific fields.
    """

    model: Optional[str] = None
    provider: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None

    def set_model_info(
        self,
        model: str,
        provider: str,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        cost_usd: Optional[float] = None,
    ) -> None:
        """Set model information."""
        self.model = model
        self.provider = provider
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost_usd = cost_usd

    def update_with_request_attributes(self, attributes: Dict[str, Any]) -> None:
        """Update generation with request attributes."""
        for key, value in attributes.items():
            self.set_attribute(key, value)

    def update_with_response_attributes(self, attributes: Dict[str, Any]) -> None:
        """Update generation with response attributes."""
        for key, value in attributes.items():
            self.set_attribute(key, value)

    @classmethod
    def create_from_ai_request(cls, request_data: Dict[str, Any], **kwargs):
        """Create a generation span from AI request data."""
        name = kwargs.pop("name", "ai_generation")
        generation = cls(name=name, **kwargs)

        # Extract common fields from request
        if "model" in request_data:
            generation.model = request_data["model"]

        return generation


# Thread-local current span storage
import threading

_current_span = threading.local()


def get_current_span() -> Optional[BrokleSpan]:
    """Get current active span."""
    return getattr(_current_span, "span", None)


def _set_current_span(span: Optional[BrokleSpan]) -> None:
    """Set current active span."""
    _current_span.span = span


def create_span(
    name: str,
    trace_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> BrokleSpan:
    """
    Create a new span.

    Args:
        name: Span name
        trace_id: Trace ID
        parent_span_id: Parent span ID
        attributes: Initial attributes
        tags: Initial tags

    Returns:
        New BrokleSpan instance
    """
    # Auto-set parent from current span
    if parent_span_id is None:
        current = get_current_span()
        if current:
            parent_span_id = current.span_id
            trace_id = trace_id or current.trace_id

    # Generate trace ID if not provided
    if trace_id is None:
        trace_id = f"trace_{uuid.uuid4().hex[:16]}"

    span = BrokleSpan(
        name=name,
        trace_id=trace_id,
        parent_span_id=parent_span_id,
        attributes=attributes or {},
        tags=tags or [],
    )

    return span


def record_span(span: BrokleSpan) -> None:
    """
    Record a span (send to backend via background processor).

    Args:
        span: Span to record
    """
    try:
        client = get_client()
        if client and span.status in ("completed", "error"):
            # Convert span to telemetry data
            telemetry_data = {
                "type": "span",
                "span_id": span.span_id,
                "trace_id": span.trace_id,
                "parent_span_id": span.parent_span_id,
                "name": span.name,
                "start_time": span.start_time.isoformat() if span.start_time else None,
                "end_time": span.end_time.isoformat() if span.end_time else None,
                "duration_ms": None,
                "status": span.status,
                "attributes": span.attributes,
                "metadata": span.metadata,
                "tags": span.tags,
                "timestamp": time.time(),
            }

            # Calculate duration if we have both timestamps
            if span.start_time and span.end_time:
                duration = span.end_time - span.start_time
                telemetry_data["duration_ms"] = duration.total_seconds() * 1000

            # Submit through background processor
            client.submit_telemetry(telemetry_data)

            import logging

            logger = logging.getLogger(__name__)
            logger.debug(f"Submitted span telemetry: {span.name} ({span.span_id})")
    except Exception as e:
        # Don't let observability failures break the application
        import logging

        logger = logging.getLogger(__name__)
        logger.debug(f"Failed to record span: {e}")
        pass


@contextmanager
def span_context(
    name: str,
    trace_id: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
):
    """
    Context manager for span management.

    Args:
        name: Span name
        trace_id: Trace ID
        attributes: Initial attributes
        tags: Initial tags
    """
    # Create and start span
    span = create_span(name, trace_id=trace_id, attributes=attributes, tags=tags)

    # Set as current span
    previous_span = get_current_span()
    _set_current_span(span)

    try:
        yield span
    except Exception as e:
        span.set_status("error", str(e))
        raise
    finally:
        # Finish span
        span.finish()

        # Record span
        record_span(span)

        # Restore previous span
        _set_current_span(previous_span)
