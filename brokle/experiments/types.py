"""
Type definitions for the experiments module.

Provides types for running evaluation experiments:
- Experiment: Metadata for list/get operations
- EvaluationItem: Single evaluation item result
- SummaryStats: Per-scorer summary statistics
- EvaluationResults: Complete evaluation results from run()
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, TypedDict

from ..scores.types import ScoreResult


@dataclass
class Experiment:
    """
    Experiment metadata (for list/get operations).

    Attributes:
        id: Unique experiment identifier
        name: Human-readable experiment name
        dataset_id: ID of the dataset used
        status: Current status (running, completed, failed)
        metadata: Additional experiment metadata
        created_at: ISO timestamp when created
        updated_at: ISO timestamp when last updated
    """

    id: str
    name: str
    dataset_id: str
    status: str  # "running", "completed", "failed"
    created_at: str
    updated_at: str
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experiment":
        """Create Experiment from API response dict."""
        return cls(
            id=data["id"],
            name=data["name"],
            dataset_id=data["dataset_id"],
            status=data["status"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            metadata=data.get("metadata"),
        )


@dataclass
class EvaluationItem:
    """
    Single evaluation item result.

    Represents the result of running a task and scorers on one dataset item.

    Attributes:
        dataset_item_id: ID of the source dataset item
        input: Input data passed to the task
        output: Output returned by the task
        expected: Expected output from the dataset (optional)
        scores: List of score results from all scorers
        trial_number: Trial number (1-based, for multi-trial experiments)
        error: Error message if task failed (optional)
    """

    dataset_item_id: str
    input: Dict[str, Any]
    output: Any
    scores: List[ScoreResult] = field(default_factory=list)
    expected: Optional[Any] = None
    trial_number: int = 1
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for API submission."""
        result: Dict[str, Any] = {
            "dataset_item_id": self.dataset_item_id,
            "input": self.input,
            "output": self.output,
            "trial_number": self.trial_number,
            "scores": [
                {
                    "name": s.name,
                    "value": s.value,
                    "type": s.type.value if hasattr(s.type, "value") else str(s.type),
                    "string_value": s.string_value,
                    "reason": s.reason,
                    "metadata": s.metadata,
                    "scoring_failed": s.scoring_failed,
                }
                for s in self.scores
            ],
        }
        if self.expected is not None:
            result["expected"] = self.expected
        if self.error is not None:
            result["error"] = self.error
        return result


class SummaryStats(TypedDict):
    """
    Per-scorer summary statistics.

    Computed across all evaluation items for a single scorer.
    Only non-failed scores are included in mean/std_dev/min/max.

    Attributes:
        mean: Average score value
        std_dev: Standard deviation of scores
        min: Minimum score value
        max: Maximum score value
        count: Total number of scores (including failed)
        pass_rate: Percentage of non-failed scores (0.0-1.0)
    """

    mean: float
    std_dev: float
    min: float
    max: float
    count: int
    pass_rate: float


@dataclass
class EvaluationResults:
    """
    Complete evaluation results from run().

    Contains all evaluation items, summary statistics, and experiment metadata.

    Attributes:
        experiment_id: ID of the created experiment
        experiment_name: Name of the experiment
        dataset_id: ID of the dataset used
        url: Dashboard URL to view the experiment (optional)
        summary: Per-scorer summary statistics
        items: List of all evaluation item results
    """

    experiment_id: str
    experiment_name: str
    dataset_id: str
    summary: Dict[str, SummaryStats]
    items: List[EvaluationItem]
    url: Optional[str] = None

    def __repr__(self) -> str:
        """String representation."""
        return f"EvaluationResults(experiment='{self.experiment_name}', items={len(self.items)})"

    def __len__(self) -> int:
        """Return number of evaluation items."""
        return len(self.items)


# Type aliases for task functions
TaskFunction = Callable[[Dict[str, Any]], Any]
"""Synchronous task function: (input) -> output"""

AsyncTaskFunction = Callable[[Dict[str, Any]], Coroutine[Any, Any, Any]]
"""Asynchronous task function: async (input) -> output"""

ProgressCallback = Callable[[int, int], None]
"""Progress callback: (completed, total) -> None"""
