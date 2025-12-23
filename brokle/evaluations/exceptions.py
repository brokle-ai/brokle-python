"""
Exception classes for the evaluations module.

This module provides a hierarchy of exceptions for evaluation-related errors:
- EvaluationError: Base exception for all evaluation errors
- ScoreError: Error submitting a score to the API
- ScorerError: Error executing a scorer function
"""


class EvaluationError(Exception):
    """Base exception for all evaluation-related errors."""

    pass


class ScoreError(EvaluationError):
    """
    Error submitting a score to the API.

    Raised when the score API request fails due to:
    - Network errors
    - Authentication/authorization issues
    - Invalid score data
    - Server errors
    """

    pass


class ScorerError(EvaluationError):
    """
    Error executing a scorer function.

    Raised when a scorer fails to execute properly.
    Note: By default, scorer errors are captured gracefully and
    returned as a score with scoring_failed=True in metadata.
    This exception is raised only when explicit error handling is needed.
    """

    pass
