"""
Exception classes for the experiments module.

This module provides exception classes for experiment-related errors:
- EvaluationError: Base exception for evaluation operations
- TaskError: Task function execution failed
- ScorerExecutionError: Scorer execution failed
"""

from typing import Optional


class EvaluationError(Exception):
    """
    Error with evaluation operations.

    Raised when evaluation API requests fail due to:
    - Network errors
    - Authentication/authorization issues
    - Invalid experiment data
    - Server errors
    """

    pass


class TaskError(EvaluationError):
    """
    Task function execution failed.

    Raised when the user-provided task function throws an exception
    during evaluation. The original exception is preserved.

    Attributes:
        dataset_item_id: ID of the dataset item being processed
        original_error: The original exception from the task function
    """

    def __init__(
        self,
        message: str,
        dataset_item_id: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.dataset_item_id = dataset_item_id
        self.original_error = original_error


class ScorerExecutionError(EvaluationError):
    """
    Scorer execution failed.

    Raised when a scorer function throws an exception during evaluation.
    The evaluation continues with scoring_failed=True for that score.

    Attributes:
        scorer_name: Name of the scorer that failed
        dataset_item_id: ID of the dataset item being processed
        original_error: The original exception from the scorer
    """

    def __init__(
        self,
        message: str,
        scorer_name: Optional[str] = None,
        dataset_item_id: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.scorer_name = scorer_name
        self.dataset_item_id = dataset_item_id
        self.original_error = original_error
