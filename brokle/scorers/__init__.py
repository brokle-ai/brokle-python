"""
Brokle Scorers Module

Provides built-in scorers, LLM-as-Judge scorers, and decorators for creating
custom evaluation functions.

Built-in Scorers:
- ExactMatch: Exact string comparison
- Contains: Substring matching
- RegexMatch: Regex pattern matching
- JSONValid: JSON validity check
- LengthCheck: String length validation

LLM-as-Judge Scorers:
- LLMScorer: Use LLM models to evaluate outputs with project credentials

Decorators:
- @scorer: Create custom scorers from functions
- @multi_scorer: Create scorers that return multiple scores

Usage:
    >>> from brokle import Brokle
    >>> from brokle.scorers import ExactMatch, Contains, LLMScorer, scorer, ScoreResult
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
    >>> # LLM-as-Judge scorer
    >>> relevance = LLMScorer(
    ...     client=client,
    ...     name="relevance",
    ...     prompt="Rate relevance 0-10: {{output}}",
    ...     model="gpt-4o",
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

# Re-export ScoreResult for convenience in custom scorers
from ..scores.types import ScoreResult, ScoreType
from .base import Contains, ExactMatch, JSONValid, LengthCheck, RegexMatch
from .decorator import multi_scorer, scorer
from .llm_scorer import LLMScorer

__all__ = [
    # Built-in scorers
    "ExactMatch",
    "Contains",
    "RegexMatch",
    "JSONValid",
    "LengthCheck",
    # LLM-as-Judge scorers
    "LLMScorer",
    # Decorators
    "scorer",
    "multi_scorer",
    # Types (for custom scorers)
    "ScoreResult",
    "ScoreType",
]
