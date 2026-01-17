"""
Type definitions for the experiments module.

Provides types for running evaluation experiments:
- Experiment: Metadata for list/get operations
- EvaluationItem: Single evaluation item result
- SummaryStats: Per-scorer summary statistics
- EvaluationResults: Complete evaluation results from run()

Span-based evaluation (THE WEDGE):
- SpanExtractInput: Function to extract input from a queried span
- SpanExtractOutput: Function to extract output from a queried span
- SpanExtractExpected: Function to extract expected output from a queried span
"""

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    TypedDict,
)

from ..scores.types import ScoreResult

if TYPE_CHECKING:
    from ..query.types import QueriedSpan


@dataclass
class Experiment:
    """
    Experiment metadata (for list/get operations).

    Attributes:
        id: Unique experiment identifier
        name: Human-readable experiment name
        status: Current status (running, completed, failed)
        created_at: ISO timestamp when created
        updated_at: ISO timestamp when last updated
        dataset_id: ID of the dataset used (None for span-based experiments)
        metadata: Additional experiment metadata
    """

    id: str
    name: str
    status: str  # "running", "completed", "failed"
    created_at: str
    updated_at: str
    dataset_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experiment":
        """Create Experiment from API response dict."""
        return cls(
            id=data["id"],
            name=data["name"],
            status=data["status"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            dataset_id=data.get("dataset_id"),
            metadata=data.get("metadata"),
        )


@dataclass
class EvaluationItem:
    """
    Single evaluation item result.

    Represents the result of running a task and scorers on one item.
    Can be from a dataset (dataset_item_id set) or from spans (span_id set).

    Attributes:
        input: Input data passed to the task
        output: Output returned by the task
        expected: Expected output from the dataset (optional)
        scores: List of score results from all scorers
        trial_number: Trial number (1-based, for multi-trial experiments)
        error: Error message if task failed (optional)
        dataset_item_id: ID of the source dataset item (for dataset-based)
        span_id: ID of the source span (for span-based - THE WEDGE)
    """

    input: Dict[str, Any]
    output: Any
    scores: List[ScoreResult] = field(default_factory=list)
    expected: Optional[Any] = None
    trial_number: int = 1
    error: Optional[str] = None
    dataset_item_id: Optional[str] = None  # For dataset-based evaluation
    span_id: Optional[str] = None  # For span-based evaluation (THE WEDGE)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for API submission."""
        result: Dict[str, Any] = {
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
        if self.dataset_item_id is not None:
            result["dataset_item_id"] = self.dataset_item_id
        if self.span_id is not None:
            result["span_id"] = self.span_id
        if self.expected is not None:
            result["expected"] = self.expected
        if self.error is not None:
            result["error"] = self.error
        return result


@dataclass
class SummaryStats:
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
    Can be from a dataset-based or span-based (THE WEDGE) evaluation.

    Attributes:
        experiment_id: ID of the created experiment
        experiment_name: Name of the experiment
        summary: Per-scorer summary statistics
        items: List of all evaluation item results
        url: Dashboard URL to view the experiment (optional)
        dataset_id: ID of the dataset used (for dataset-based)
        source: Source type ('dataset' or 'spans')
        experiment: Experiment object with full metadata (property)
    """

    experiment_id: str
    experiment_name: str
    summary: Dict[str, SummaryStats]
    items: List[EvaluationItem]
    url: Optional[str] = None
    dataset_id: Optional[str] = None  # For dataset-based evaluation
    source: str = "dataset"  # 'dataset' or 'spans'
    status: str = "completed"  # Experiment status
    created_at: Optional[str] = None  # ISO timestamp
    updated_at: Optional[str] = None  # ISO timestamp
    metadata: Optional[Dict[str, Any]] = None  # Experiment metadata

    @property
    def experiment(self) -> Experiment:
        """
        Get the experiment as an Experiment object.

        Provides convenient access to experiment metadata via results.experiment.id,
        results.experiment.name, etc.

        Returns:
            Experiment object with full metadata
        """
        return Experiment(
            id=self.experiment_id,
            name=self.experiment_name,
            status=self.status,
            created_at=self.created_at or "",
            updated_at=self.updated_at or "",
            dataset_id=self.dataset_id,
            metadata=self.metadata,
        )

    def get_summary(self, scorer_name: str) -> Optional[SummaryStats]:
        """
        Get summary statistics for a specific scorer.

        Args:
            scorer_name: Name of the scorer

        Returns:
            SummaryStats for the scorer, or None if not found
        """
        return self.summary.get(scorer_name)

    def __repr__(self) -> str:
        """String representation."""
        return f"EvaluationResults(experiment='{self.experiment_name}', items={len(self.items)}, source='{self.source}')"

    def __len__(self) -> int:
        """Return number of evaluation items."""
        return len(self.items)


class ScoreAggregation(TypedDict):
    """
    Score aggregation statistics from experiment comparison.

    Attributes:
        mean: Average score value
        std_dev: Standard deviation of scores
        min: Minimum score value
        max: Maximum score value
        count: Total number of scores
    """

    mean: float
    std_dev: float
    min: float
    max: float
    count: int


class ScoreDiff(TypedDict, total=False):
    """
    Score difference from baseline in experiment comparison.

    Attributes:
        type: Diff type (e.g., "percentage", "absolute")
        difference: Numeric difference value
        direction: Direction of change ("up", "down", "same")
    """

    type: str
    difference: float
    direction: str


class ExperimentSummary(TypedDict):
    """
    Experiment summary in comparison results.

    Attributes:
        name: Experiment name
        status: Experiment status
    """

    name: str
    status: str


@dataclass
class ComparisonResult:
    """
    Result of comparing multiple experiments.

    Provides score aggregations and diffs across experiments.

    Attributes:
        experiments: Map of experiment ID to summary info
        scores: Map of scorer name -> experiment ID -> aggregation stats
        diffs: Map of scorer name -> experiment ID -> diff from baseline (optional)
    """

    experiments: Dict[str, ExperimentSummary]
    scores: Dict[str, Dict[str, ScoreAggregation]]
    diffs: Optional[Dict[str, Dict[str, ScoreDiff]]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComparisonResult":
        """Create ComparisonResult from API response dict."""
        return cls(
            experiments=data.get("experiments", {}),
            scores=data.get("scores", {}),
            diffs=data.get("diffs"),
        )

    def get_score_summary(
        self, scorer_name: str, experiment_id: str
    ) -> Optional[ScoreAggregation]:
        """
        Get score aggregation for a specific scorer and experiment.

        Args:
            scorer_name: Name of the scorer
            experiment_id: ID of the experiment

        Returns:
            ScoreAggregation if found, None otherwise
        """
        if scorer_name in self.scores:
            return self.scores[scorer_name].get(experiment_id)
        return None

    def get_diff(
        self, scorer_name: str, experiment_id: str
    ) -> Optional[ScoreDiff]:
        """
        Get score diff from baseline for a specific scorer and experiment.

        Args:
            scorer_name: Name of the scorer
            experiment_id: ID of the experiment

        Returns:
            ScoreDiff if found, None otherwise
        """
        if self.diffs and scorer_name in self.diffs:
            return self.diffs[scorer_name].get(experiment_id)
        return None


# Type aliases for task functions
TaskFunction = Callable[[Dict[str, Any]], Any]
"""Synchronous task function: (input) -> output"""

AsyncTaskFunction = Callable[[Dict[str, Any]], Coroutine[Any, Any, Any]]
"""Asynchronous task function: async (input) -> output"""

ProgressCallback = Callable[[int, int], None]
"""Progress callback: (completed, total) -> None"""


# Type aliases for span extraction functions (THE WEDGE)
# These are used with span-based evaluation to extract data from queried spans
SpanExtractInput = Callable[["QueriedSpan"], Dict[str, Any]]
"""Extract input from a queried span: (span) -> input_dict"""

SpanExtractOutput = Callable[["QueriedSpan"], Any]
"""Extract output from a queried span: (span) -> output"""

SpanExtractExpected = Callable[["QueriedSpan"], Any]
"""Extract expected output from a queried span: (span) -> expected"""
