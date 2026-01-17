"""
Brokle Experiments Module

Provides experiment management for running evaluations against datasets.

Usage:
    >>> from brokle import Brokle
    >>> from brokle.scorers import ExactMatch
    >>>
    >>> client = Brokle(api_key="bk_...")
    >>> dataset = client.datasets.get("dataset_id")
    >>>
    >>> def my_task(input):
    ...     return call_llm(input["prompt"])
    >>>
    >>> results = client.experiments.run(
    ...     name="gpt4-test",
    ...     dataset=dataset,
    ...     task=my_task,
    ...     scorers=[ExactMatch()],
    ... )
    >>>
    >>> for name, stats in results.summary.items():
    ...     print(f"{name}: mean={stats['mean']:.3f}")

Async Usage:
    >>> async with AsyncBrokle(api_key="bk_...") as client:
    ...     results = await client.experiments.run(
    ...         name="test",
    ...         dataset=dataset,
    ...         task=my_task,
    ...         scorers=[ExactMatch()],
    ...     )
"""

from ._manager import AsyncExperimentsManager, ExperimentsManager
from .exceptions import EvaluationError, ScorerExecutionError, TaskError
from .types import (
    AsyncTaskFunction,
    ComparisonResult,
    EvaluationItem,
    EvaluationResults,
    Experiment,
    ExperimentSummary,
    ProgressCallback,
    ScoreAggregation,
    ScoreDiff,
    SummaryStats,
    TaskFunction,
)

__all__ = [
    # Managers
    "ExperimentsManager",
    "AsyncExperimentsManager",
    # Types
    "EvaluationResults",
    "EvaluationItem",
    "SummaryStats",
    "Experiment",
    "TaskFunction",
    "AsyncTaskFunction",
    "ProgressCallback",
    # Comparison types
    "ComparisonResult",
    "ScoreAggregation",
    "ScoreDiff",
    "ExperimentSummary",
    # Exceptions
    "EvaluationError",
    "TaskError",
    "ScorerExecutionError",
]
