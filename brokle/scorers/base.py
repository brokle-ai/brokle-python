"""
Built-in Scorers for Brokle Evaluations

Provides ready-to-use scorers for common evaluation patterns:
- ExactMatch: Exact string comparison
- Contains: Substring matching
- RegexMatch: Regex pattern matching
- JSONValid: JSON validity check
- LengthCheck: String length validation

All scorers follow the ScorerProtocol and return ScoreResult.

Usage:
    >>> from brokle import Brokle
    >>> from brokle.scorers import ExactMatch, Contains
    >>>
    >>> client = Brokle(api_key="bk_...")
    >>>
    >>> # Exact match
    >>> exact = ExactMatch(name="answer_match", case_sensitive=False)
    >>> client.evaluations.score(
    ...     trace_id="abc123",
    ...     scorer=exact,
    ...     output="Paris",
    ...     expected="paris",
    ... )
    >>>
    >>> # Contains check
    >>> contains = Contains(name="keyword_present")
    >>> client.evaluations.score(
    ...     trace_id="abc123",
    ...     scorer=contains,
    ...     output="The capital of France is Paris",
    ...     expected="Paris",
    ... )
"""

import json
import re
from typing import Any, Optional, Union

from ..evaluations.types import ScoreResult, ScoreType


class ExactMatch:
    """
    Exact string comparison scorer.

    Compares output and expected values as strings after stripping whitespace.
    Supports case-sensitive and case-insensitive comparison.

    Args:
        name: Score name (default: "exact_match")
        case_sensitive: Whether to perform case-sensitive comparison (default: True)

    Example:
        >>> exact = ExactMatch(name="answer_match", case_sensitive=False)
        >>> result = exact(output="Paris", expected="paris")
        >>> result.value  # 1.0 (match)

        >>> result = exact(output="London", expected="paris")
        >>> result.value  # 0.0 (no match)
    """

    def __init__(self, name: str = "exact_match", case_sensitive: bool = True):
        self.name = name
        self.case_sensitive = case_sensitive

    def __call__(self, output: Any, expected: Any, **kwargs: Any) -> ScoreResult:
        """
        Compare output and expected values.

        Args:
            output: The actual output to evaluate
            expected: The expected/reference value
            **kwargs: Additional arguments (ignored)

        Returns:
            ScoreResult with value 1.0 (match) or 0.0 (no match)
        """
        out_str = str(output).strip()
        exp_str = str(expected).strip()

        if not self.case_sensitive:
            out_str = out_str.lower()
            exp_str = exp_str.lower()

        match = out_str == exp_str

        return ScoreResult(
            name=self.name,
            value=1.0 if match else 0.0,
            type=ScoreType.BOOLEAN,
        )


class Contains:
    """
    Substring matching scorer.

    Checks if the expected value is contained within the output.
    Supports case-sensitive and case-insensitive comparison.

    Args:
        name: Score name (default: "contains")
        case_sensitive: Whether to perform case-sensitive comparison (default: True)

    Example:
        >>> contains = Contains(name="has_keyword")
        >>> result = contains(output="The capital is Paris", expected="Paris")
        >>> result.value  # 1.0 (contains)

        >>> result = contains(output="Hello world", expected="foo")
        >>> result.value  # 0.0 (not found)
    """

    def __init__(self, name: str = "contains", case_sensitive: bool = True):
        self.name = name
        self.case_sensitive = case_sensitive

    def __call__(self, output: Any, expected: Any, **kwargs: Any) -> ScoreResult:
        """
        Check if expected is contained in output.

        Args:
            output: The actual output to evaluate
            expected: The substring to find
            **kwargs: Additional arguments (ignored)

        Returns:
            ScoreResult with value 1.0 (found) or 0.0 (not found)
        """
        out_str = str(output)
        exp_str = str(expected)

        if not self.case_sensitive:
            out_str = out_str.lower()
            exp_str = exp_str.lower()

        match = exp_str in out_str

        return ScoreResult(
            name=self.name,
            value=1.0 if match else 0.0,
            type=ScoreType.BOOLEAN,
        )


class RegexMatch:
    """
    Regex pattern matching scorer.

    Checks if the output matches a given regex pattern.
    The pattern can be a string or a pre-compiled regex.

    Args:
        pattern: Regex pattern (string or compiled re.Pattern)
        name: Score name (default: "regex_match")

    Example:
        >>> # Match email pattern
        >>> email_check = RegexMatch(r"[a-z]+@[a-z]+\\.[a-z]+", name="has_email")
        >>> result = email_check(output="Contact: test@example.com")
        >>> result.value  # 1.0 (match found)

        >>> # Match phone pattern
        >>> phone_check = RegexMatch(r"\\d{3}-\\d{3}-\\d{4}")
        >>> result = phone_check(output="Call 555-123-4567")
        >>> result.value  # 1.0
    """

    def __init__(
        self, pattern: Union[str, re.Pattern[str]], name: str = "regex_match"
    ):
        self.name = name
        self.pattern = re.compile(pattern) if isinstance(pattern, str) else pattern

    def __call__(self, output: Any, **kwargs: Any) -> ScoreResult:
        """
        Check if output matches the regex pattern.

        Args:
            output: The actual output to evaluate
            **kwargs: Additional arguments (ignored, including expected)

        Returns:
            ScoreResult with value 1.0 (match) or 0.0 (no match)
        """
        match = bool(self.pattern.search(str(output)))

        return ScoreResult(
            name=self.name,
            value=1.0 if match else 0.0,
            type=ScoreType.BOOLEAN,
        )


class JSONValid:
    """
    JSON validity check scorer.

    Validates whether the output is valid JSON.

    Args:
        name: Score name (default: "json_valid")

    Example:
        >>> json_check = JSONValid()
        >>> result = json_check(output='{"key": "value"}')
        >>> result.value  # 1.0 (valid JSON)

        >>> result = json_check(output='{invalid json}')
        >>> result.value  # 0.0 (invalid JSON)
    """

    def __init__(self, name: str = "json_valid"):
        self.name = name

    def __call__(self, output: Any, **kwargs: Any) -> ScoreResult:
        """
        Check if output is valid JSON.

        Args:
            output: The actual output to evaluate
            **kwargs: Additional arguments (ignored)

        Returns:
            ScoreResult with value 1.0 (valid) or 0.0 (invalid)
        """
        try:
            json.loads(str(output))
            valid = True
        except (json.JSONDecodeError, TypeError):
            valid = False

        return ScoreResult(
            name=self.name,
            value=1.0 if valid else 0.0,
            type=ScoreType.BOOLEAN,
        )


class LengthCheck:
    """
    String length validation scorer.

    Validates that the output length falls within specified bounds.
    At least one of min_length or max_length must be specified.

    Args:
        min_length: Minimum allowed length (inclusive)
        max_length: Maximum allowed length (inclusive)
        name: Score name (default: "length_check")

    Example:
        >>> # Check for reasonable response length
        >>> length = LengthCheck(min_length=10, max_length=1000)
        >>> result = length(output="Hello world!")
        >>> result.value  # 1.0 (12 chars, within bounds)

        >>> # Ensure minimum length
        >>> min_check = LengthCheck(min_length=50)
        >>> result = min_check(output="Too short")
        >>> result.value  # 0.0 (below minimum)
    """

    def __init__(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        name: str = "length_check",
    ):
        self.name = name
        self.min_length = min_length
        self.max_length = max_length

    def __call__(self, output: Any, **kwargs: Any) -> ScoreResult:
        """
        Check if output length is within bounds.

        Args:
            output: The actual output to evaluate
            **kwargs: Additional arguments (ignored)

        Returns:
            ScoreResult with value 1.0 (valid) or 0.0 (invalid),
            includes actual length in metadata
        """
        length = len(str(output))
        valid = True

        if self.min_length is not None and length < self.min_length:
            valid = False
        if self.max_length is not None and length > self.max_length:
            valid = False

        return ScoreResult(
            name=self.name,
            value=1.0 if valid else 0.0,
            type=ScoreType.BOOLEAN,
            metadata={"length": length},
        )
