"""
Score management for quality evaluation.

Provides utility functions for adding scores to traces, spans, and sessions.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from ..types.observability import Score, ScoreDataType, ScoreSource
from ..types.telemetry import TelemetryEventType
from .._utils.ulid import generate_ulid

if TYPE_CHECKING:
    from ..client import Brokle, AsyncBrokle


def score_trace(
    client: 'Brokle',
    trace_id: str,
    name: str,
    value: Optional[float] = None,
    string_value: Optional[str] = None,
    data_type: ScoreDataType = ScoreDataType.NUMERIC,
    source: ScoreSource = ScoreSource.API,
    comment: Optional[str] = None,
    **kwargs
) -> str:
    """
    Add quality score to a trace.

    Args:
        client: Brokle client instance
        trace_id: Trace ULID
        name: Score name/metric (e.g., "quality", "accuracy")
        value: Numeric value (for NUMERIC/BOOLEAN types)
        string_value: String value (for CATEGORICAL type)
        data_type: Score data type (NUMERIC, CATEGORICAL, BOOLEAN)
        source: Score source (API, SDK, HUMAN, EVAL)
        comment: Optional comment/explanation
        **kwargs: Additional score fields

    Returns:
        Score ID (ULID)

    Example:
        >>> score_trace(
        ...     client=client,
        ...     trace_id="01ABCDEFGHIJKLMNOPQRSTUVWX",
        ...     name="quality",
        ...     value=0.95
        ... )
    """
    score_id = kwargs.pop('id', None) or generate_ulid()

    score = Score(
        id=score_id,
        trace_id=trace_id,
        name=name,
        value=value,
        string_value=string_value,
        data_type=data_type,
        source=source,
        comment=comment,
        timestamp=datetime.now(timezone.utc),
        **kwargs
    )

    # Submit to backend via batch API
    client.submit_batch_event(
        TelemetryEventType.QUALITY_SCORE,
        score.model_dump(mode="json", exclude_none=True)
    )

    return score_id


def score_span(
    client: 'Brokle',
    span_id: str,
    name: str,
    value: Optional[float] = None,
    string_value: Optional[str] = None,
    data_type: ScoreDataType = ScoreDataType.NUMERIC,
    source: ScoreSource = ScoreSource.API,
    comment: Optional[str] = None,
    **kwargs
) -> str:
    """
    Add quality score to an span.

    Args:
        client: Brokle client instance
        span_id: Span ULID
        name: Score name/metric
        value: Numeric value (for NUMERIC/BOOLEAN types)
        string_value: String value (for CATEGORICAL type)
        data_type: Score data type (NUMERIC, CATEGORICAL, BOOLEAN)
        source: Score source (API, SDK, HUMAN, EVAL)
        comment: Optional comment/explanation
        **kwargs: Additional score fields

    Returns:
        Score ID (ULID)

    Example:
        >>> score_span(
        ...     client=client,
        ...     span_id="01ABCDEFGHIJKLMNOPQRSTUVWX",
        ...     name="accuracy",
        ...     value=0.88
        ... )
    """
    score_id = kwargs.pop('id', None) or generate_ulid()

    score = Score(
        id=score_id,
        span_id=span_id,
        name=name,
        value=value,
        string_value=string_value,
        data_type=data_type,
        source=source,
        comment=comment,
        timestamp=datetime.now(timezone.utc),
        **kwargs
    )

    # Submit to backend via batch API
    client.submit_batch_event(
        TelemetryEventType.QUALITY_SCORE,
        score.model_dump(mode="json", exclude_none=True)
    )

    return score_id


def score_session(
    client: 'Brokle',
    session_id: str,
    name: str,
    value: Optional[float] = None,
    string_value: Optional[str] = None,
    data_type: ScoreDataType = ScoreDataType.NUMERIC,
    source: ScoreSource = ScoreSource.API,
    comment: Optional[str] = None,
    **kwargs
) -> str:
    """
    Add quality score to a session.

    Args:
        client: Brokle client instance
        session_id: Session ULID
        name: Score name/metric
        value: Numeric value (for NUMERIC/BOOLEAN types)
        string_value: String value (for CATEGORICAL type)
        data_type: Score data type (NUMERIC, CATEGORICAL, BOOLEAN)
        source: Score source (API, SDK, HUMAN, EVAL)
        comment: Optional comment/explanation
        **kwargs: Additional score fields

    Returns:
        Score ID (ULID)

    Example:
        >>> score_session(
        ...     client=client,
        ...     session_id="01ABCDEFGHIJKLMNOPQRSTUVWX",
        ...     name="overall_quality",
        ...     value=0.92
        ... )
    """
    score_id = kwargs.pop('id', None) or generate_ulid()

    score = Score(
        id=score_id,
        session_id=session_id,
        name=name,
        value=value,
        string_value=string_value,
        data_type=data_type,
        source=source,
        comment=comment,
        timestamp=datetime.now(timezone.utc),
        **kwargs
    )

    # Submit to backend via batch API
    client.submit_batch_event(
        TelemetryEventType.QUALITY_SCORE,
        score.model_dump(mode="json", exclude_none=True)
    )

    return score_id


__all__ = ["score_trace", "score_span", "score_session"]
