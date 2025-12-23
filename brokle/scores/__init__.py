"""
Scores Module

Provides score submission for Brokle evaluations.

Usage:
    >>> from brokle import Brokle
    >>> from brokle.scorers import ExactMatch
    >>>
    >>> client = Brokle(api_key="bk_...")
    >>>
    >>> # Direct score
    >>> client.scores.submit(
    ...     trace_id="abc123",
    ...     name="quality",
    ...     value=0.9,
    ... )
    >>>
    >>> # With scorer
    >>> exact = ExactMatch()
    >>> client.scores.submit(
    ...     trace_id="abc123",
    ...     scorer=exact,
    ...     output="Paris",
    ...     expected="Paris",
    ... )
    >>>
    >>> # Batch scores
    >>> client.scores.batch([
    ...     {"trace_id": "abc123", "name": "accuracy", "value": 0.9},
    ...     {"trace_id": "abc123", "name": "relevance", "value": 0.85},
    ... ])
"""

from ._managers import AsyncScoresManager, ScoresManager
from .exceptions import ScoreError, ScorerError
from .types import (
    Scorer,
    ScorerArgs,
    ScorerProtocol,
    ScoreResult,
    ScoreSource,
    ScoreType,
    ScoreValue,
)

__all__ = [
    # Managers
    "ScoresManager",
    "AsyncScoresManager",
    # Types
    "ScoreType",
    "ScoreSource",
    "ScoreResult",
    "ScoreValue",
    "ScorerProtocol",
    "Scorer",
    "ScorerArgs",
    # Exceptions
    "ScoreError",
    "ScorerError",
]
