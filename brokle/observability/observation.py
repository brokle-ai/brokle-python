"""
Observation client for observability.

Provides client-side observation management with fluent API for different observation types.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional

from ..types.observability import Observation, ObservationLevel, ObservationType, Score, ScoreDataType, ScoreSource
from .._utils.ulid import generate_ulid
from .context import push_observation, pop_observation, get_current_observation_id

if TYPE_CHECKING:
    from ..client import Brokle, AsyncBrokle


class ObservationClient:
    """
    Client-side observation management.

    Provides fluent API for building observations with type-specific helpers
    (generation, span, event, etc.).

    Example:
        >>> obs = trace.observation(ObservationType.GENERATION, "openai-call")
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
        type: ObservationType,
        name: str,
        parent_observation_id: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize observation client.

        Args:
            client: Brokle client instance
            trace_id: Parent trace ULID
            type: Observation type
            name: Human-readable observation name
            parent_observation_id: Optional parent observation ULID
            **kwargs: Additional observation fields
        """
        # Generate ULID if not provided
        obs_id = kwargs.pop('id', None) or generate_ulid()

        # Auto-detect parent if not provided
        if parent_observation_id is None:
            parent_observation_id = get_current_observation_id()
        
        # Create observation entity
        self.observation = Observation(
            id=obs_id,
            trace_id=trace_id,
            parent_observation_id=parent_observation_id,
            type=type,
            name=name,
            start_time=datetime.now(timezone.utc),
            **kwargs
        )

        self._client = client
        self._submitted = False
        
        # Push to context for child observations
        push_observation(self.observation.id)

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
        Configure as LLM generation observation.

        Auto-extracts token usage and calculates costs from output.

        Args:
            model: Model identifier (e.g., "gpt-4", "claude-3-opus")
            input: Input data (e.g., {"messages": [...]})
            output: Output data (e.g., {"response": "...", "usage": {...}})
            model_parameters: Model configuration (temperature, max_tokens, etc.)
            usage: Token usage dict (prompt_tokens, completion_tokens, total_tokens)
            **kwargs: Additional observation fields

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
        # Set observation type to GENERATION
        self.observation.type = ObservationType.GENERATION
        self.observation.model = model

        if input is not None:
            self.observation.input = input

        if output is not None:
            self.observation.output = output

            # Auto-extract token usage from output
            if "usage" in output and usage is None:
                usage = output["usage"]

        if model_parameters is not None:
            # Convert all values to strings for ClickHouse Map(String, String)
            self.observation.model_parameters = {
                k: str(v) for k, v in model_parameters.items()
            }

        # Set token usage
        if usage is not None:
            self.observation.usage_details = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

        # Set additional fields
        for key, value in kwargs.items():
            if hasattr(self.observation, key):
                setattr(self.observation, key, value)

        return self

    def span(
        self,
        input: Optional[Dict[str, Any]] = None,
        output: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> 'ObservationClient':
        """
        Configure as generic span observation.

        Args:
            input: Input data
            output: Output data
            metadata: String key-value metadata
            **kwargs: Additional observation fields

        Returns:
            Self for fluent chaining

        Example:
            >>> obs.span(
            ...     input={"query": "search term"},
            ...     output={"results": [...]},
            ...     metadata={"db": "postgres"}
            ... )
        """
        self.observation.type = ObservationType.SPAN

        if input is not None:
            self.observation.input = input

        if output is not None:
            self.observation.output = output

        if metadata is not None:
            self.observation.metadata = metadata

        for key, value in kwargs.items():
            if hasattr(self.observation, key):
                setattr(self.observation, key, value)

        return self

    def event(
        self,
        input: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> 'ObservationClient':
        """
        Configure as point-in-time event observation.

        Args:
            input: Event data
            metadata: String key-value metadata
            **kwargs: Additional observation fields

        Returns:
            Self for fluent chaining

        Example:
            >>> obs.event(
            ...     input={"event_name": "user_action", "params": {...}},
            ...     metadata={"source": "frontend"}
            ... )
        """
        self.observation.type = ObservationType.EVENT

        if input is not None:
            self.observation.input = input

        if metadata is not None:
            self.observation.metadata = metadata

        # Events typically don't have end_time (point-in-time)
        self.observation.end_time = self.observation.start_time

        for key, value in kwargs.items():
            if hasattr(self.observation, key):
                setattr(self.observation, key, value)

        return self

    def child_observation(
        self,
        type: ObservationType,
        name: str,
        **kwargs
    ) -> 'ObservationClient':
        """
        Create child observation.
        
        Args:
            type: Observation type
            name: Human-readable observation name
            **kwargs: Additional observation fields
            
        Returns:
            New ObservationClient for child observation
            
        Example:
            >>> child_obs = obs.child_observation(ObservationType.TOOL, "vector-search")
            >>> child_obs.end()
        """
        return ObservationClient(
            client=self._client,
            trace_id=self.observation.trace_id,
            type=type,
            name=name,
            parent_observation_id=self.observation.id,
            **kwargs
        )

    def update(self, **kwargs) -> None:
        """
        Update observation fields.

        This creates a new version in the backend (immutable event pattern).

        Args:
            **kwargs: Fields to update

        Example:
            >>> obs.update(metadata={"updated": "true"})
            >>> obs.update(output={"result": "success"})
        """
        for key, value in kwargs.items():
            if hasattr(self.observation, key):
                setattr(self.observation, key, value)

    def set_level(self, level: ObservationLevel, status_message: Optional[str] = None) -> None:
        """
        Set observation level and optional status message.

        Args:
            level: Observation level (DEBUG, DEFAULT, WARNING, ERROR)
            status_message: Optional status/error message

        Example:
            >>> obs.set_level(ObservationLevel.ERROR, "Connection failed")
        """
        self.observation.level = level
        if status_message is not None:
            self.observation.status_message = status_message

    def score(
        self,
        name: str,
        value: Optional[float] = None,
        string_value: Optional[str] = None,
        data_type: Optional[ScoreDataType] = None,
        **kwargs
    ) -> str:
        """
        Add quality score to observation.

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
            observation_id=self.observation.id,
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
        Complete observation and submit to backend.

        Args:
            output: Optional output data
            **kwargs: Additional fields to update before submission

        Example:
            >>> obs.end(output={"result": "success"})
        """
        if self._submitted:
            return  # Already submitted, ignore duplicate calls

        # Set end time
        self.observation.end_time = datetime.now(timezone.utc)

        # Update final fields
        if output is not None:
            self.observation.output = output

        for key, value in kwargs.items():
            if hasattr(self.observation, key):
                setattr(self.observation, key, value)

        # Submit to backend via batch API
        from ..types.telemetry import TelemetryEventType

        self._client.submit_batch_event(
            TelemetryEventType.OBSERVATION,
            self.observation.model_dump(mode="json", exclude_none=True)
        )

        self._submitted = True
        # Pop from context
        pop_observation()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with auto-end."""
        if exc_type is not None:
            # Error occurred, set error level and message
            self.observation.level = ObservationLevel.ERROR
            self.observation.status_message = str(exc_val)

        self.end()


__all__ = ["ObservationClient"]
