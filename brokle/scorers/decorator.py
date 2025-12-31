"""
Scorer Decorator for Custom Evaluations

The @scorer decorator allows you to create custom scoring functions
that integrate seamlessly with the Brokle evaluation system.

Usage:
    >>> from brokle.scorers import scorer
    >>> from brokle.scores import ScoreResult
    >>>
    >>> # Simple scorer returning float (auto-wrapped)
    >>> @scorer
    ... def similarity(output, expected=None, **kwargs):
    ...     # Your custom logic here
    ...     return 0.85  # Wrapped as ScoreResult(name="similarity", value=0.85)
    >>>
    >>> # Scorer returning ScoreResult (full control)
    >>> @scorer
    ... def detailed_scorer(output, expected=None, **kwargs):
    ...     score = compute_score(output, expected)
    ...     return ScoreResult(
    ...         name="detailed",
    ...         value=score,
    ...         reason=f"Computed similarity: {score:.2f}",
    ...         metadata={"algorithm": "custom"},
    ...     )
    >>>
    >>> # Use with client
    >>> client.scores.submit(
    ...     trace_id="abc123",
    ...     scorer=similarity,
    ...     output="result",
    ...     expected="expected",
    ... )
"""

import functools
from typing import Any, Callable, List, TypeVar, Union, cast

from ..scores.types import ScoreResult, ScoreType, ScoreValue, ScorerProtocol

F = TypeVar("F", bound=Callable[..., ScoreValue])


def scorer(func: F) -> ScorerProtocol:
    """
    Decorator to mark a function as a scorer.

    The decorated function can return:
    - float/int: Auto-wrapped in ScoreResult with function name as score name
    - bool: Converted to 1.0/0.0 with BOOLEAN type
    - ScoreResult: Used directly
    - List[ScoreResult]: Multiple scores from one evaluation

    Args:
        func: The scoring function to decorate. Must accept output and
              optional expected kwargs.

    Returns:
        A ScorerProtocol-compatible callable

    Example:
        >>> @scorer
        ... def my_scorer(output, expected=None, **kwargs):
        ...     return 0.9  # Wrapped as ScoreResult(name="my_scorer", value=0.9)

        >>> @scorer
        ... def bool_scorer(output, expected=None, **kwargs):
        ...     return output == expected  # Wrapped as BOOLEAN

        >>> @scorer
        ... def full_scorer(output, expected=None, **kwargs):
        ...     return ScoreResult(name="custom", value=0.8, reason="Good")
    """

    @functools.wraps(func)
    def wrapper(
        output: Any = None,
        expected: Any = None,
        **kwargs: Any,
    ) -> ScoreValue:
        result = func(output=output, expected=expected, **kwargs)

        if isinstance(result, ScoreResult):
            return result
        if isinstance(result, list):
            return result

        if isinstance(result, bool):
            return ScoreResult(
                name=func.__name__,
                value=1.0 if result else 0.0,
                type=ScoreType.BOOLEAN,
            )
        elif isinstance(result, (int, float)):
            return ScoreResult(
                name=func.__name__,
                value=float(result),
                type=ScoreType.NUMERIC,
            )
        else:
            raise TypeError(
                f"Scorer {func.__name__} must return ScoreResult, "
                f"List[ScoreResult], float, int, or bool, "
                f"got {type(result).__name__}"
            )

    # Preserve the original function name for scorer identification
    wrapper.name = func.__name__  # type: ignore[attr-defined]

    # Mark as scorer for introspection
    wrapper._is_scorer = True  # type: ignore[attr-defined]

    return cast(ScorerProtocol, wrapper)


def multi_scorer(func: Callable[..., List[ScoreResult]]) -> ScorerProtocol:
    """
    Decorator for scorers that return multiple scores.

    Use this when a single evaluation produces multiple scores
    (e.g., accuracy, fluency, and relevance from one function).

    Args:
        func: The scoring function. Must return List[ScoreResult].

    Returns:
        A ScorerProtocol-compatible callable

    Example:
        >>> @multi_scorer
        ... def quality_metrics(output, expected=None, **kwargs):
        ...     return [
        ...         ScoreResult(name="accuracy", value=0.9),
        ...         ScoreResult(name="fluency", value=0.85),
        ...         ScoreResult(name="relevance", value=0.95),
        ...     ]
    """

    @functools.wraps(func)
    def wrapper(
        output: Any = None,
        expected: Any = None,
        **kwargs: Any,
    ) -> List[ScoreResult]:
        result = func(output=output, expected=expected, **kwargs)

        if not isinstance(result, list):
            raise TypeError(
                f"Multi-scorer {func.__name__} must return List[ScoreResult], "
                f"got {type(result).__name__}"
            )

        for item in result:
            if not isinstance(item, ScoreResult):
                raise TypeError(
                    f"Multi-scorer {func.__name__} must return List[ScoreResult], "
                    f"but list contains {type(item).__name__}"
                )

        return result

    # Preserve the original function name for scorer identification
    wrapper.name = func.__name__  # type: ignore[attr-defined]

    # Mark as scorer for introspection
    wrapper._is_scorer = True  # type: ignore[attr-defined]
    wrapper._is_multi_scorer = True  # type: ignore[attr-defined]

    return cast(ScorerProtocol, wrapper)
