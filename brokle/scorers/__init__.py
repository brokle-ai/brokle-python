"""
Brokle Scorers Module

Provides built-in scorers and decorators for creating custom evaluation functions.

Built-in Scorers:
- ExactMatch: Exact string comparison
- Contains: Substring matching
- RegexMatch: Regex pattern matching
- JSONValid: JSON validity check
- LengthCheck: String length validation

Decorators:
- @scorer: Create custom scorers from functions
- @multi_scorer: Create scorers that return multiple scores

Usage:
    >>> from brokle import Brokle
    >>> from brokle.scorers import ExactMatch, Contains, scorer, ScoreResult
    >>>
    >>> client = Brokle(api_key="bk_...")
    >>>
    >>> # Built-in scorer
    >>> exact = ExactMatch(name="answer_match")
    >>> client.scores.submit(
    ...     trace_id="abc123",
    ...     scorer=exact,
    ...     output="Paris",
    ...     expected="Paris",
    ... )
    >>>
    >>> # Custom scorer
    >>> @scorer
    ... def similarity(output, expected=None, **kwargs):
    ...     return 0.85  # Auto-wrapped as ScoreResult
    >>>
    >>> client.scores.submit(
    ...     trace_id="abc123",
    ...     scorer=similarity,
    ...     output="result",
    ...     expected="expected",
    ... )
"""

from .base import Contains, ExactMatch, JSONValid, LengthCheck, RegexMatch
from .decorator import multi_scorer, scorer

# Re-export ScoreResult for convenience in custom scorers
from ..scores.types import ScoreResult, ScoreType

__all__ = [
    # Built-in scorers
    "ExactMatch",
    "Contains",
    "RegexMatch",
    "JSONValid",
    "LengthCheck",
    # Decorators
    "scorer",
    "multi_scorer",
    # Types (for custom scorers)
    "ScoreResult",
    "ScoreType",
]
