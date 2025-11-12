"""
Trace client for observability.

Provides client-side trace management with auto-submit on completion.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..types.observability import Span, SpanType, Score, Trace, ScoreDataType, ScoreSource
from ..types.telemetry import TelemetryEventType
from .._utils.ulid import generate_ulid
from .context import push_trace, pop_trace

if TYPE_CHECKING:
    from ..client import Brokle, AsyncBrokle
    from .span import ObservationClient


class TraceClient:
    """
    Client-side trace management with auto-submit on completion.

    Provides a fluent API for building traces with spans and scores.

    Example:
        >>> trace = client.trace(
        ...     name="user-query",
        ...     user_id="user_123",
        ...     metadata={"intent": "search"}
        ... )
        >>> obs = trace.span(SpanType.LLM, "openai-call")
        >>> obs.generation(model="gpt-4", input={...}, output={...})
        >>> obs.end()
        >>> trace.score("quality", 0.95)
        >>> trace.end(output={"result": "success"})
    """

    def __init__(self, client: 'Brokle', name: str, **kwargs):
        """
        Initialize trace client.

        Args:
            client: Brokle client instance
            name: Human-readable trace name
            **kwargs: Additional trace fields (session_id, user_id, metadata, etc.)
        """
        # Generate ULID if not provided
        trace_id = kwargs.pop('id', None) or generate_ulid()

        # Create trace entity
        self.trace = Trace(
            id=trace_id,
            name=name,
            timestamp=datetime.now(timezone.utc),
            **kwargs
        )

        self._client = client
        self._spans: List['ObservationClient'] = []
        self._submitted = False
        
        # Push to context for Pattern 1/2 compatibility
        push_trace(self.trace.id)

    def span(
        self,
        type: SpanType,
        name: str,
        parent_span_id: Optional[str] = None,
        **kwargs
    ) -> 'ObservationClient':
        """
        Create child span.

        Args:
            type: Span type (LLM, GENERATION, SPAN, etc.)
            name: Human-readable span name
            parent_span_id: Optional parent span ID for nesting
            **kwargs: Additional span fields

        Returns:
            ObservationClient for fluent API

        Example:
            >>> obs = trace.span(SpanType.LLM, "openai-call")
            >>> obs.generation(model="gpt-4", input={...}, output={...})
            >>> obs.end()
        """
        from .span import ObservationClient

        obs = ObservationClient(
            client=self._client,
            trace_id=self.trace.id,
            type=type,
            name=name,
            parent_span_id=parent_span_id,
            **kwargs
        )
        self._spans.append(obs)
        return obs

    def score(
        self,
        name: str,
        value: Optional[float] = None,
        string_value: Optional[str] = None,
        data_type: Optional[ScoreDataType] = None,
        **kwargs
    ) -> str:
        """
        Add quality score to trace.

        Args:
            name: Score name/metric (e.g., "quality", "accuracy")
            value: Numeric value (for NUMERIC/BOOLEAN types)
            string_value: String value (for CATEGORICAL type)
            data_type: Score data type (auto-detected if not provided)
            **kwargs: Additional score fields

        Returns:
            Score ID (ULID)

        Example:
            >>> trace.score("quality", value=0.95)
            >>> trace.score("category", string_value="excellent")
        """
        # Auto-detect data type if not provided
        if data_type is None:
            if value is not None:
                data_type = ScoreDataType.NUMERIC
            elif string_value is not None:
                data_type = ScoreDataType.CATEGORICAL
            else:
                data_type = ScoreDataType.NUMERIC
        
        score = Score(
            id=generate_ulid(),
            trace_id=self.trace.id,
            name=name,
            value=value,
            string_value=string_value,
            data_type=data_type,
            source=ScoreSource.API,
            timestamp=datetime.now(timezone.utc),
            **kwargs
        )
        
        from ..types.telemetry import TelemetryEventType

        self._client.submit_batch_event(
            TelemetryEventType.QUALITY_SCORE,
            score.model_dump(mode="json", exclude_none=True)
        )

        return score.id

    def update(self, **kwargs) -> None:
        """
        Update trace fields.

        This creates a new version of the trace in the backend (immutable event pattern).

        Args:
            **kwargs: Fields to update (input, output, metadata, tags, etc.)

        Example:
            >>> trace.update(metadata={"updated": "true"})
            >>> trace.update(output={"result": "success"})
        """
        # Update trace fields
        for key, value in kwargs.items():
            if hasattr(self.trace, key):
                setattr(self.trace, key, value)

    def end(self, output: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """
        Complete trace and submit to backend.

        Args:
            output: Optional output data
            **kwargs: Additional fields to update before submission

        Example:
            >>> trace.end(output={"result": "success"})
        """
        if self._submitted:
            return  # Already submitted, ignore duplicate calls

        # Update final fields
        if output is not None:
            self.trace.output = output

        for key, value in kwargs.items():
            if hasattr(self.trace, key):
                setattr(self.trace, key, value)

        # Submit to backend via batch API
        from ..types.telemetry import TelemetryEventType

        self._client.submit_batch_event(
            TelemetryEventType.TRACE,
            self.trace.model_dump(mode="json", exclude_none=True)
        )

        self._submitted = True
        # Pop from context
        pop_trace()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with auto-end."""
        if exc_type is not None:
            # Error occurred, add to metadata
            self.trace.metadata["error"] = str(exc_val)
            self.trace.metadata["error_type"] = exc_type.__name__

        self.end()


class AsyncTraceClient:
    """
    Async trace client for AsyncBrokle.

    Provides the same fluent API as TraceClient but for async contexts.

    Example:
        >>> trace = await async_client.trace(name="user-query")
        >>> obs = await trace.span(SpanType.LLM, "openai-call")
        >>> await obs.end()
        >>> await trace.end()
    """

    def __init__(self, client: 'AsyncBrokle', name: str, **kwargs):
        """
        Initialize async trace client.

        Args:
            client: AsyncBrokle client instance
            name: Human-readable trace name
            **kwargs: Additional trace fields
        """
        # Generate ULID if not provided
        trace_id = kwargs.pop('id', None) or generate_ulid()

        # Create trace entity
        self.trace = Trace(
            id=trace_id,
            name=name,
            timestamp=datetime.now(timezone.utc),
            **kwargs
        )

        self._client = client
        self._spans: List['ObservationClient'] = []
        self._submitted = False

    def span(
        self,
        type: SpanType,
        name: str,
        parent_span_id: Optional[str] = None,
        **kwargs
    ) -> 'ObservationClient':
        """Create child span (sync method, span submission is async)."""
        from .span import ObservationClient

        obs = ObservationClient(
            client=self._client,
            trace_id=self.trace.id,
            type=type,
            name=name,
            parent_span_id=parent_span_id,
            **kwargs
        )
        self._spans.append(obs)
        return obs

    def score(
        self,
        name: str,
        value: Optional[float] = None,
        string_value: Optional[str] = None,
        **kwargs
    ) -> str:
        """Add quality score to trace (sync submission)."""
        from .score import score_trace

        return score_trace(
            client=self._client,
            trace_id=self.trace.id,
            name=name,
            value=value,
            string_value=string_value,
            **kwargs
        )

    def update(self, **kwargs) -> None:
        """Update trace fields."""
        for key, value in kwargs.items():
            if hasattr(self.trace, key):
                setattr(self.trace, key, value)

    def end(self, output: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Complete trace and submit to backend."""
        if self._submitted:
            return

        if output is not None:
            self.trace.output = output

        for key, value in kwargs.items():
            if hasattr(self.trace, key):
                setattr(self.trace, key, value)

        # Submit to backend (background processor handles async)
        self._client.submit_batch_event(
            TelemetryEventType.TRACE,
            self.trace.model_dump(mode="json", exclude_none=True)
        )

        self._submitted = True

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with auto-end."""
        if exc_type is not None:
            self.trace.metadata["error"] = str(exc_val)
            self.trace.metadata["error_type"] = exc_type.__name__

        self.end()


__all__ = ["TraceClient", "AsyncTraceClient"]
