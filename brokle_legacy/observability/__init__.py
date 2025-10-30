"""
Public observability helpers.

This module intentionally exposes only the stable pieces needed by the three
integration patterns (wrappers, decorator, native SDK). Anything else should
remain private to avoid API churn.
"""

from ..types.observability import (
    Observation,
    ObservationLevel,
    ObservationType,
    Score,
    ScoreDataType,
    ScoreSource,
    Session,
    Trace,
)
from .context import (
    clear_context,
    client_context,
    get_client,
    get_client_context,
    get_context,
    get_context_info,
    get_current_observation_id,
    get_current_trace_id,
    get_session_id,
    pop_observation,
    pop_trace,
    push_observation,
    push_trace,
    set_client,
    set_session_id,
)
from .observation import ObservationClient
from .score import score_observation, score_session, score_trace
from .session import SessionClient
from .trace import TraceClient

__all__ = [
    # Domain types
    "Observation",
    "ObservationLevel",
    "ObservationType",
    "Score",
    "ScoreDataType",
    "ScoreSource",
    "Session",
    "Trace",
    # Context helpers
    "client_context",
    "clear_context",
    "get_client",
    "get_client_context",
    "get_context",
    "get_context_info",
    "get_current_observation_id",
    "get_current_trace_id",
    "get_session_id",
    "pop_observation",
    "pop_trace",
    "push_observation",
    "push_trace",
    "set_client",
    "set_session_id",
    # Clients
    "ObservationClient",
    "SessionClient",
    "TraceClient",
    # Score helpers
    "score_observation",
    "score_session",
    "score_trace",
]
