"""
Exception classes for the scores module.

This module provides exception classes for score-related errors.
"""


class ScoreError(Exception):
    """
    Error submitting a score to the API.

    Raised when the score API request fails due to:
    - Network errors
    - Authentication/authorization issues
    - Invalid score data
    - Server errors
    """

    pass


class ScorerError(Exception):
    """
    Error executing a scorer function.

    Raised when a scorer fails to execute properly.
    Note: By default, scorer errors are captured gracefully and
    returned as a score with scoring_failed=True in metadata.
    This exception is raised only when explicit error handling is needed.
    """

    pass
