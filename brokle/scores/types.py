"""
Type definitions for the scores module.

This module provides the core types used for scoring:
- ScoreType: Enum for score data types (NUMERIC, CATEGORICAL, BOOLEAN)
- ScoreSource: Enum for score sources (code, llm, human)
- ScoreResult: Dataclass representing a score returned by scorers
- ScoreValue: Union type for flexible scorer return types
- ScorerProtocol: Protocol defining the scorer callable interface
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Union


class ScoreType(str, Enum):
    """Score data type classification."""

    NUMERIC = "NUMERIC"
    CATEGORICAL = "CATEGORICAL"
    BOOLEAN = "BOOLEAN"


class ScoreSource(str, Enum):
    """Source of the score (how it was generated)."""

    CODE = "code"
    LLM = "llm"
    HUMAN = "human"


@dataclass
class ScoreResult:
    """
    Return type for all scorers.

    Attributes:
        name: The name of the score (e.g., "accuracy", "relevance")
        value: The numeric score value (typically 0.0-1.0)
        type: The score data type (NUMERIC, CATEGORICAL, BOOLEAN)
        string_value: String value for CATEGORICAL scores
        reason: Human-readable explanation for the score
        metadata: Additional metadata about the scoring
        scoring_failed: Flag indicating if the scorer failed to execute
    """

    name: str
    value: float
    type: ScoreType = ScoreType.NUMERIC
    string_value: Optional[str] = None
    reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    scoring_failed: bool = False


ScoreValue = Union[float, int, bool, ScoreResult, List[ScoreResult], None]


class ScorerProtocol(Protocol):
    """
    Protocol defining the interface for scorer callables.

    Any callable matching this signature can be used as a scorer:
    - Built-in scorer classes (ExactMatch, Contains, etc.)
    - Functions decorated with @scorer
    - Custom classes implementing __call__

    Example:
        @scorer
        def my_scorer(output, expected=None, **kwargs) -> ScoreResult:
            return ScoreResult(name="custom", value=0.9)

        class MyScorer:
            def __call__(self, output, expected=None, **kwargs) -> float:
                return 0.9  # Auto-wrapped in ScoreResult
    """

    def __call__(
        self,
        output: Any,
        expected: Any = None,
        **kwargs: Any,
    ) -> ScoreValue:
        """
        Execute the scorer on the given output.

        Args:
            output: The actual output to evaluate
            expected: The expected/reference output (optional)
            **kwargs: Additional arguments (e.g., input, context)

        Returns:
            ScoreValue: The score result (float, bool, ScoreResult, or list)
        """
        ...


# Type aliases for API compatibility
Scorer = ScorerProtocol
ScorerArgs = Dict[str, Any]
