"""
Type definitions for the query module (THE WEDGE).

Provides types for querying production spans:
- QueriedSpan: Span from query results with OTEL GenAI conventions
- QueryResult: Result from query() with pagination
- ValidationResult: Result from validate()
- TokenUsage: Token usage extracted from span
- SpanEvent: Event attached to a span
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class TokenUsage:
    """
    Token usage from LLM span.

    Extracted from span attributes for convenience access.

    Attributes:
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens
        total_tokens: Total tokens (prompt + completion)
    """

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["TokenUsage"]:
        """Create TokenUsage from usage_details dict."""
        if not data:
            return None
        return cls(
            prompt_tokens=data.get("prompt_tokens"),
            completion_tokens=data.get("completion_tokens"),
            total_tokens=data.get("total_tokens"),
        )


@dataclass
class SpanEvent:
    """
    Event attached to a span.

    OTEL spans can have multiple events with timestamps.

    Attributes:
        name: Event name
        timestamp: ISO timestamp string
        attributes: Event attributes
    """

    name: str
    timestamp: str
    attributes: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpanEvent":
        """Create SpanEvent from dict."""
        return cls(
            name=data["name"],
            timestamp=data["timestamp"],
            attributes=data.get("attributes"),
        )


@dataclass
class QueriedSpan:
    """
    Span from query results (OTEL GenAI conventions).

    This is THE WEDGE - queried production spans that can be used
    for retrospective evaluation without re-instrumenting applications.

    Convenience fields (input, output, model, token_usage) are extracted
    from attributes for easy access to common GenAI data.

    Attributes:
        trace_id: Trace identifier
        span_id: Span identifier
        name: Span name (operation name)
        start_time: ISO timestamp when span started
        attributes: Merged resource + span attributes

        parent_span_id: Parent span ID (None for root spans)
        service_name: Service name from resource attributes
        end_time: ISO timestamp when span ended
        duration: Duration in microseconds
        status: Span status ('unset', 'ok', 'error')
        status_message: Error message if status is 'error'
        events: List of span events

        input: Extracted input from attributes (convenience)
        output: Extracted output from attributes (convenience)
        model: Model name from attributes (convenience)
        token_usage: Token usage from attributes (convenience)
    """

    trace_id: str
    span_id: str
    name: str
    start_time: str
    attributes: Dict[str, Any]

    # Optional fields
    parent_span_id: Optional[str] = None
    service_name: Optional[str] = None
    end_time: Optional[str] = None
    duration: Optional[int] = None  # microseconds
    status: str = "unset"
    status_message: Optional[str] = None
    events: Optional[List[SpanEvent]] = None

    # Convenience accessors (extracted from attributes)
    input: Optional[str] = None
    output: Optional[str] = None
    model: Optional[str] = None
    token_usage: Optional[TokenUsage] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueriedSpan":
        """
        Create QueriedSpan from API response dict.

        Maps backend field names to Python SDK properties.
        Merges resource_attributes and span_attributes into single attributes dict.
        Extracts convenience fields from attributes.
        """
        # Merge resource_attributes and span_attributes
        resource_attrs = data.get("resource_attributes") or {}
        span_attrs = data.get("span_attributes") or {}
        attributes = {**resource_attrs, **span_attrs}

        # Convert duration from nanoseconds to microseconds if present
        duration_ns = data.get("duration")
        duration_us = duration_ns // 1000 if duration_ns else None

        # Parse events if present
        events = None
        if data.get("events"):
            events = [SpanEvent.from_dict(e) for e in data["events"]]

        # Extract convenience fields
        input_val = data.get("input")
        output_val = data.get("output")
        # Model: check direct field first, then attributes
        model = (
            data.get("model_name")
            or attributes.get("gen_ai.response.model")
            or attributes.get("gen_ai.request.model")
        )
        token_usage = TokenUsage.from_dict(data.get("usage_details"))

        # Extract service_name - check direct field first, then resource attributes
        service_name = data.get("service_name") or resource_attrs.get("service.name")

        return cls(
            trace_id=data["trace_id"],
            span_id=data["span_id"],
            name=data.get("span_name", data.get("name", "")),
            start_time=data["start_time"],
            attributes=attributes,
            parent_span_id=data.get("parent_span_id"),
            service_name=service_name,
            end_time=data.get("end_time"),
            duration=duration_us,
            status=data.get("status", "unset"),
            status_message=data.get("status_message"),
            events=events,
            input=input_val,
            output=output_val,
            model=model,
            token_usage=token_usage,
        )


@dataclass
class QueryResult:
    """
    Result from query() call.

    Contains paginated spans and pagination metadata.

    Attributes:
        spans: List of queried spans
        total: Total number of matching spans
        has_more: Whether more results are available
        next_offset: Offset for next page (if has_more is True)
    """

    spans: List[QueriedSpan]
    total: int
    has_more: bool
    next_offset: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryResult":
        """Create QueryResult from API response dict."""
        spans_data = data.get("spans", [])
        spans = [QueriedSpan.from_dict(s) for s in spans_data]

        total = data.get("total_count", len(spans))
        has_more = data.get("has_more", False)
        next_offset = data.get("next_offset")

        return cls(
            spans=spans,
            total=total,
            has_more=has_more,
            next_offset=next_offset,
        )

    def __len__(self) -> int:
        """Return number of spans in this result."""
        return len(self.spans)

    def __iter__(self):
        """Iterate over spans."""
        return iter(self.spans)


@dataclass
class ValidationResult:
    """
    Result from validate() call.

    Indicates whether a filter expression is syntactically valid.

    Attributes:
        valid: Whether the filter is valid
        message: Success message if valid
        error: Error message if invalid
    """

    valid: bool
    message: Optional[str] = None
    error: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationResult":
        """Create ValidationResult from API response dict."""
        return cls(
            valid=data.get("valid", False),
            message=data.get("message"),
            error=data.get("error"),
        )
