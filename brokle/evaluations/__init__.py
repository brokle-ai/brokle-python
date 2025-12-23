"""
Brokle Evaluations Module

Provides evaluation and scoring functionality accessed via client.evaluations.

Supports two scoring modes:
1. Direct score: Pass name + value directly
2. Scorer function: Pass a scorer callable with output/expected

Example (Sync):
    >>> from brokle import Brokle, ScoreType
    >>> from brokle.scorers import ExactMatch
    >>>
    >>> client = Brokle(api_key="bk_...")
    >>>
    >>> # Direct score
    >>> client.evaluations.score(
    ...     trace_id="abc123",
    ...     name="quality",
    ...     value=0.9,
    ... )
    >>>
    >>> # With built-in scorer
    >>> exact = ExactMatch(name="answer_match")
    >>> client.evaluations.score(
    ...     trace_id="abc123",
    ...     scorer=exact,
    ...     output="Paris",
    ...     expected="Paris",
    ... )

Example (Async):
    >>> async with AsyncBrokle(api_key="bk_...") as client:
    ...     await client.evaluations.score(
    ...         trace_id="abc123",
    ...         name="quality",
    ...         value=0.9,
    ...     )
"""

from ._managers import AsyncEvaluationsManager, EvaluationsManager
from .exceptions import EvaluationError, ScoreError, ScorerError
from .types import ScoreResult, ScoreSource, ScoreType, ScoreValue, ScorerProtocol

__all__ = [
    # Managers
    "EvaluationsManager",
    "AsyncEvaluationsManager",
    # Types
    "ScoreType",
    "ScoreSource",
    "ScoreResult",
    "ScoreValue",
    "ScorerProtocol",
    # Exceptions
    "EvaluationError",
    "ScoreError",
    "ScorerError",
]
