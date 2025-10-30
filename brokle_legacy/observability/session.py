"""
Session management for user journey grouping.

Provides session client for grouping multiple traces together.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, Optional

from ..types.observability import Session
from ..types.telemetry import TelemetryEventType
from .._utils.ulid import generate_ulid

if TYPE_CHECKING:
    from ..client import Brokle, AsyncBrokle
    from .trace import TraceClient


class SessionClient:
    """
    Client-side session management for grouping traces.

    Sessions group related traces together (e.g., chat conversations, workflow executions).

    Example:
        >>> session = client.session(
        ...     user_id="user_123",
        ...     metadata={"conversation_type": "support"}
        ... )
        >>> trace = session.trace("first-query")
        >>> trace.end()
        >>> session.update(metadata={"messages": "5"})
    """

    def __init__(
        self,
        client: 'Brokle',
        id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Initialize session client.

        Args:
            client: Brokle client instance
            id: Optional session ULID (generates new if not provided)
            user_id: Optional user ULID
            metadata: String key-value metadata
            **kwargs: Additional session fields
        """
        # Generate ULID if not provided
        session_id = id or generate_ulid()

        # Create session entity
        self.session = Session(
            id=session_id,
            user_id=user_id,
            metadata=metadata or {},
            created_at=datetime.now(timezone.utc),
            **kwargs
        )

        self._client = client
        self._submitted = False

        # Auto-submit session creation
        self._submit()

    def _submit(self) -> None:
        """Submit session to backend."""
        from ..types.telemetry import TelemetryEventType

        self._client.submit_batch_event(
            TelemetryEventType.SESSION,
            self.session.model_dump(mode="json", exclude_none=True)
        )
        self._submitted = True

    def trace(self, name: str, **kwargs) -> 'TraceClient':
        """
        Create a trace within this session.

        Args:
            name: Human-readable trace name
            **kwargs: Additional trace fields

        Returns:
            TraceClient with session_id set

        Example:
            >>> session = client.session(user_id="user_123")
            >>> trace = session.trace("user-query")
            >>> trace.end()
        """
        from .trace import TraceClient

        # Ensure session_id is set on trace
        kwargs['session_id'] = self.session.id

        return TraceClient(
            client=self._client,
            name=name,
            **kwargs
        )

    def update(self, **kwargs) -> None:
        """
        Update session fields.

        This creates a new version in the backend (immutable event pattern).

        Args:
            **kwargs: Fields to update (metadata, bookmarked, public)

        Example:
            >>> session.update(metadata={"messages": "10"})
            >>> session.update(bookmarked=True)
        """
        for key, value in kwargs.items():
            if hasattr(self.session, key):
                setattr(self.session, key, value)

        # Re-submit with updated fields
        self._submit()

    def score(
        self,
        name: str,
        value: Optional[float] = None,
        string_value: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Add quality score to session.

        Args:
            name: Score name/metric
            value: Numeric value (for NUMERIC/BOOLEAN types)
            string_value: String value (for CATEGORICAL type)
            **kwargs: Additional score fields

        Returns:
            Score ID (ULID)

        Example:
            >>> session.score("overall_quality", value=0.92)
        """
        from .score import score_session

        return score_session(
            client=self._client,
            session_id=self.session.id,
            name=name,
            value=value,
            string_value=string_value,
            **kwargs
        )


class AsyncSessionClient:
    """
    Async session client for AsyncBrokle.

    Provides the same API as SessionClient but for async contexts.

    Example:
        >>> session = await async_client.session(user_id="user_123")
        >>> trace = await session.trace("query")
        >>> await trace.end()
    """

    def __init__(
        self,
        client: 'AsyncBrokle',
        id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Initialize async session client.

        Args:
            client: AsyncBrokle client instance
            id: Optional session ULID
            user_id: Optional user ULID
            metadata: String key-value metadata
            **kwargs: Additional session fields
        """
        session_id = id or generate_ulid()

        self.session = Session(
            id=session_id,
            user_id=user_id,
            metadata=metadata or {},
            created_at=datetime.now(timezone.utc),
            **kwargs
        )

        self._client = client
        self._submitted = False

        # Auto-submit session creation
        self._submit()

    def _submit(self) -> None:
        """Submit session to backend (background processor handles async)."""
        self._client.submit_batch_event(
            TelemetryEventType.SESSION,
            self.session.model_dump(mode="json", exclude_none=True)
        )
        self._submitted = True

    def trace(self, name: str, **kwargs) -> 'TraceClient':
        """Create a trace within this session."""
        from .trace import AsyncTraceClient

        kwargs['session_id'] = self.session.id

        return AsyncTraceClient(
            client=self._client,
            name=name,
            **kwargs
        )

    def update(self, **kwargs) -> None:
        """Update session fields."""
        for key, value in kwargs.items():
            if hasattr(self.session, key):
                setattr(self.session, key, value)

        self._submit()

    def score(
        self,
        name: str,
        value: Optional[float] = None,
        string_value: Optional[str] = None,
        **kwargs
    ) -> str:
        """Add quality score to session."""
        from .score import score_session

        return score_session(
            client=self._client,
            session_id=self.session.id,
            name=name,
            value=value,
            string_value=string_value,
            **kwargs
        )


__all__ = ["SessionClient", "AsyncSessionClient"]
