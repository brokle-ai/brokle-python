"""
Span client for observability.

Provides client-side span management with fluent API for different span types.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional

from ..types.observability import Span, ObservationLevel, SpanType, Score, ScoreDataType, ScoreSource
from .._utils.ulid import generate_ulid
from .context import push_span, pop_span, get_current_span_id

if TYPE_CHECKING:
    from ..client import Brokle, AsyncBrokle


class SpanClient:
    """
    Client-side span management.

    Provides fluent API for building spans with type-specific helpers
    (generation, span, event, etc.).

    Example:
        >>> obs = trace.span(SpanType.GENERATION, "openai-call")
        >>> obs.generation(
        ...     model="gpt-4",
        ...     input={"prompt": "Hello"},
        ...     output={"response": "Hi there", "usage": {...}}
        ... )
        >>> obs.end()
    """

    def __init__(
        self,
        client: 'Brokle',
        trace_id: str,
        type: SpanType,
        name: str,
        parent_span_id: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize span client.

        Args:
            client: Brokle client instance
            trace_id: Parent trace ULID
            type: Span type
            name: Human-readable span name
            parent_span_id: Optional parent span ULID
            **kwargs: Additional span fields
        """
        # Generate ULID if not provided
        obs_id = kwargs.pop('id', None) or generate_ulid()

        # Auto-detect parent if not provided
        if parent_span_id is None:
            parent_span_id = get_current_span_id()
        
        # Create span entity
        self.span = Span(
            id=obs_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            type=type,
            name=name,
            start_time=datetime.now(timezone.utc),
            **kwargs
        )

        self._client = client
        self._submitted = False
        
        # Push to context for child spans
        push_span(self.span.id)

    def generation(
        self,
        model: str,
        input: Optional[Dict[str, Any]] = None,
        output: Optional[Dict[str, Any]] = None,
        model_parameters: Optional[Dict[str, str]] = None,
        usage: Optional[Dict[str, int]] = None,
        **kwargs
    ) -> 'ObservationClient':
        """
        Configure as LLM generation span.

        Auto-extracts token usage and calculates costs from output.

        Args:
            model: Model identifier (e.g., "gpt-4", "claude-3-opus")
            input: Input data (e.g., {"messages": [...]})
            output: Output data (e.g., {"response": "...", "usage": {...}})
            model_parameters: Model configuration (temperature, max_tokens, etc.)
            usage: Token usage dict (prompt_tokens, completion_tokens, total_tokens)
            **kwargs: Additional span fields

        Returns:
            Self for fluent chaining

        Example:
            >>> obs.generation(
            ...     model="gpt-4",
            ...     input={"messages": [{"role": "user", "content": "Hello"}]},
            ...     output={"response": "Hi", "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
            ...     model_parameters={"temperature": "0.7"}
            ... )
        """
        # Set span type to GENERATION
        self.span.type = SpanType.GENERATION
        self.span.model = model

        if input is not None:
            self.span.input = input

        if output is not None:
            self.span.output = output

            # Auto-extract token usage from output
            if "usage" in output and usage is None:
                usage = output["usage"]

        if model_parameters is not None:
            # Convert all values to strings for ClickHouse Map(String, String)
            self.span.model_parameters = {
                k: str(v) for k, v in model_parameters.items()
            }

        # Set token usage
        if usage is not None:
            self.span.usage_details = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

        # Set additional fields
        for key, value in kwargs.items():
            if hasattr(self.span, key):
                setattr(self.span, key, value)

        return self

    def span(
        self,
        input: Optional[Dict[str, Any]] = None,
        output: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> 'ObservationClient':
        """
        Configure as generic span span.

        Args:
            input: Input data
            output: Output data
            metadata: String key-value metadata
            **kwargs: Additional span fields

        Returns:
            Self for fluent chaining

        Example:
            >>> obs.span(
            ...     input={"query": "search term"},
            ...     output={"results": [...]},
            ...     metadata={"db": "postgres"}
            ... )
        """
        self.span.type = SpanType.SPAN

        if input is not None:
            self.span.input = input

        if output is not None:
            self.span.output = output

        if metadata is not None:
            self.span.metadata = metadata

        for key, value in kwargs.items():
            if hasattr(self.span, key):
                setattr(self.span, key, value)

        return self

    def event(
        self,
        input: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> 'ObservationClient':
        """
        Configure as point-in-time event span.

        Args:
            input: Event data
            metadata: String key-value metadata
            **kwargs: Additional span fields

        Returns:
            Self for fluent chaining

        Example:
            >>> obs.event(
            ...     input={"event_name": "user_action", "params": {...}},
            ...     metadata={"source": "frontend"}
            ... )
        """
        self.span.type = SpanType.EVENT

        if input is not None:
            self.span.input = input

        if metadata is not None:
            self.span.metadata = metadata

        # Events typically don't have end_time (point-in-time)
        self.span.end_time = self.span.start_time

        for key, value in kwargs.items():
            if hasattr(self.span, key):
                setattr(self.span, key, value)

        return self

    def child_span(
        self,
        type: SpanType,
        name: str,
        **kwargs
    ) -> 'ObservationClient':
        """
        Create child span.
        
        Args:
            type: Span type
            name: Human-readable span name
            **kwargs: Additional span fields
            
        Returns:
            New ObservationClient for child span
            
        Example:
            >>> child_obs = obs.child_span(SpanType.TOOL, "vector-search")
            >>> child_obs.end()
        """
        return ObservationClient(
            client=self._client,
            trace_id=self.span.trace_id,
            type=type,
            name=name,
            parent_span_id=self.span.id,
            **kwargs
        )

    def update(self, **kwargs) -> None:
        """
        Update span fields.

        This creates a new version in the backend (immutable event pattern).

        Args:
            **kwargs: Fields to update

        Example:
            >>> obs.update(metadata={"updated": "true"})
            >>> obs.update(output={"result": "success"})
        """
        for key, value in kwargs.items():
            if hasattr(self.span, key):
                setattr(self.span, key, value)

    def set_level(self, level: SpanLevel, status_message: Optional[str] = None) -> None:
        """
        Set span level and optional status message.

        Args:
            level: Span level (DEBUG, DEFAULT, WARNING, ERROR)
            status_message: Optional status/error message

        Example:
            >>> obs.set_level(ObservationLevel.ERROR, "Connection failed")
        """
        self.span.level = level
        if status_message is not None:
            self.span.status_message = status_message

    def score(
        self,
        name: str,
        value: Optional[float] = None,
        string_value: Optional[str] = None,
        data_type: Optional[ScoreDataType] = None,
        **kwargs
    ) -> str:
        """
        Add quality score to span.

        Args:
            name: Score name/metric
            value: Numeric value (for NUMERIC/BOOLEAN types)
            string_value: String value (for CATEGORICAL type)
            data_type: Score data type (auto-detected if not provided)
            **kwargs: Additional score fields

        Returns:
            Score ID (ULID)

        Example:
            >>> obs.score("quality", value=0.95)
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
            span_id=self.span.id,
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

    def end(self, output: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """
        Complete span and submit to backend.

        Args:
            output: Optional output data
            **kwargs: Additional fields to update before submission

        Example:
            >>> obs.end(output={"result": "success"})
        """
        if self._submitted:
            return  # Already submitted, ignore duplicate calls

        # Set end time
        self.span.end_time = datetime.now(timezone.utc)

        # Update final fields
        if output is not None:
            self.span.output = output

        for key, value in kwargs.items():
            if hasattr(self.span, key):
                setattr(self.span, key, value)

        # Submit to backend via batch API
        from ..types.telemetry import TelemetryEventType

        self._client.submit_batch_event(
            TelemetryEventType.SPAN,
            self.span.model_dump(mode="json", exclude_none=True)
        )

        self._submitted = True
        # Pop from context
        pop_span()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with auto-end."""
        if exc_type is not None:
            # Error occurred, set error level and message
            self.span.level = ObservationLevel.ERROR
            self.span.status_message = str(exc_val)

        self.end()


__all__ = ["ObservationClient"]
