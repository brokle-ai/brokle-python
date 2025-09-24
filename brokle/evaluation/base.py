"""
Base evaluation classes and types for Brokle evaluation framework.

Comprehensive evaluation framework with Brokle-specific enhancements.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
from enum import Enum

logger = logging.getLogger(__name__)


class EvaluationStatus(Enum):
    """Status of an evaluation run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class EvaluationResult:
    """
    Result of a single evaluation.

    Comprehensive evaluation result structure with
    Brokle-specific enhancements for AI platform metrics.
    """
    key: str
    score: float
    value: Any = None
    comment: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Brokle-specific fields
    cost_impact: Optional[float] = None
    latency_impact: Optional[float] = None
    quality_score: Optional[float] = None
    routing_decision: Optional[str] = None
    cache_hit: Optional[bool] = None

    # Standard fields
    timestamp: datetime = field(default_factory=datetime.utcnow)
    evaluator_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API serialization."""
        return {
            "key": self.key,
            "score": self.score,
            "value": self.value,
            "comment": self.comment,
            "metadata": self.metadata,
            "cost_impact": self.cost_impact,
            "latency_impact": self.latency_impact,
            "quality_score": self.quality_score,
            "routing_decision": self.routing_decision,
            "cache_hit": self.cache_hit,
            "timestamp": self.timestamp.isoformat(),
            "evaluator_info": self.evaluator_info,
        }


@dataclass
class EvaluationConfig:
    """
    Configuration for evaluation runs.

    Provides fine-grained control over evaluation behavior.
    """
    # Core settings
    max_concurrency: int = 10
    timeout_seconds: float = 300.0
    retry_attempts: int = 3
    fail_fast: bool = False

    # Sampling and filtering
    sample_rate: float = 1.0
    max_samples: Optional[int] = None
    filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None

    # Output settings
    output_format: str = "json"
    include_metadata: bool = True
    include_inputs: bool = True
    include_outputs: bool = True

    # Brokle-specific settings
    track_cost_impact: bool = True
    track_latency_impact: bool = True
    track_quality_scores: bool = True
    track_routing_decisions: bool = True
    track_cache_performance: bool = True

    # Integration settings
    send_to_brokle: bool = True
    experiment_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "max_concurrency": self.max_concurrency,
            "timeout_seconds": self.timeout_seconds,
            "retry_attempts": self.retry_attempts,
            "fail_fast": self.fail_fast,
            "sample_rate": self.sample_rate,
            "max_samples": self.max_samples,
            "output_format": self.output_format,
            "include_metadata": self.include_metadata,
            "include_inputs": self.include_inputs,
            "include_outputs": self.include_outputs,
            "track_cost_impact": self.track_cost_impact,
            "track_latency_impact": self.track_latency_impact,
            "track_quality_scores": self.track_quality_scores,
            "track_routing_decisions": self.track_routing_decisions,
            "track_cache_performance": self.track_cache_performance,
            "send_to_brokle": self.send_to_brokle,
            "experiment_name": self.experiment_name,
            "tags": self.tags,
        }


class BaseEvaluator(ABC):
    """
    Base class for all evaluators.

    Comprehensive evaluator framework with Brokle enhancements.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.name = name or self.__class__.__name__
        self.description = description or f"Evaluator: {self.name}"
        self.version = version
        self.metadata = metadata or {}

        # Evaluation tracking
        self._evaluation_count = 0
        self._total_score = 0.0
        self._start_time: Optional[datetime] = None

    @abstractmethod
    def evaluate(
        self,
        prediction: Any,
        reference: Any = None,
        input_data: Any = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate a single prediction.

        Args:
            prediction: The model's prediction/output
            reference: The expected/reference output (optional)
            input_data: The input that generated the prediction (optional)
            **kwargs: Additional evaluation parameters

        Returns:
            EvaluationResult with score and metadata
        """
        pass

    async def aevaluate(
        self,
        prediction: Any,
        reference: Any = None,
        input_data: Any = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Async version of evaluate.

        Default implementation runs sync evaluate in executor.
        Override for true async evaluation.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.evaluate,
            prediction,
            reference,
            input_data,
            **kwargs
        )

    def batch_evaluate(
        self,
        predictions: List[Any],
        references: Optional[List[Any]] = None,
        inputs: Optional[List[Any]] = None,
        config: Optional[EvaluationConfig] = None,
        **kwargs
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple predictions in batch.

        Args:
            predictions: List of predictions to evaluate
            references: List of reference outputs (optional)
            inputs: List of inputs (optional)
            config: Evaluation configuration
            **kwargs: Additional parameters

        Returns:
            List of EvaluationResult objects
        """
        config = config or EvaluationConfig()
        self._start_time = datetime.utcnow()

        results = []
        for i, prediction in enumerate(predictions):
            try:
                reference = references[i] if references else None
                input_data = inputs[i] if inputs else None

                result = self.evaluate(
                    prediction=prediction,
                    reference=reference,
                    input_data=input_data,
                    **kwargs
                )

                results.append(result)
                self._update_stats(result.score)

            except Exception as e:
                logger.error(f"Evaluation failed for item {i}: {e}")
                if config.fail_fast:
                    raise

                # Create failed result
                failed_result = EvaluationResult(
                    key=self.name,
                    score=0.0,
                    comment=f"Evaluation failed: {str(e)}",
                    metadata={"error": str(e), "item_index": i}
                )
                results.append(failed_result)

        return results

    async def abatch_evaluate(
        self,
        predictions: List[Any],
        references: Optional[List[Any]] = None,
        inputs: Optional[List[Any]] = None,
        config: Optional[EvaluationConfig] = None,
        **kwargs
    ) -> List[EvaluationResult]:
        """
        Async batch evaluation with concurrency control.

        Args:
            predictions: List of predictions to evaluate
            references: List of reference outputs (optional)
            inputs: List of inputs (optional)
            config: Evaluation configuration
            **kwargs: Additional parameters

        Returns:
            List of EvaluationResult objects
        """
        config = config or EvaluationConfig()
        self._start_time = datetime.utcnow()

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(config.max_concurrency)

        async def evaluate_single(i: int, prediction: Any) -> EvaluationResult:
            async with semaphore:
                try:
                    reference = references[i] if references else None
                    input_data = inputs[i] if inputs else None

                    # Use asyncio timeout
                    result = await asyncio.wait_for(
                        self.aevaluate(
                            prediction=prediction,
                            reference=reference,
                            input_data=input_data,
                            **kwargs
                        ),
                        timeout=config.timeout_seconds
                    )

                    self._update_stats(result.score)
                    return result

                except asyncio.TimeoutError:
                    logger.error(f"Evaluation timeout for item {i}")
                    return EvaluationResult(
                        key=self.name,
                        score=0.0,
                        comment=f"Evaluation timeout after {config.timeout_seconds}s",
                        metadata={"error": "timeout", "item_index": i}
                    )

                except Exception as e:
                    logger.error(f"Evaluation failed for item {i}: {e}")
                    if config.fail_fast:
                        raise

                    return EvaluationResult(
                        key=self.name,
                        score=0.0,
                        comment=f"Evaluation failed: {str(e)}",
                        metadata={"error": str(e), "item_index": i}
                    )

        # Create tasks for all evaluations
        tasks = [
            evaluate_single(i, prediction)
            for i, prediction in enumerate(predictions)
        ]

        # Execute with concurrency control
        results = await asyncio.gather(*tasks, return_exceptions=False)

        return results

    def _update_stats(self, score: float) -> None:
        """Update internal evaluation statistics."""
        self._evaluation_count += 1
        self._total_score += score

    def get_stats(self) -> Dict[str, Any]:
        """Get evaluation statistics."""
        avg_score = (
            self._total_score / self._evaluation_count
            if self._evaluation_count > 0
            else 0.0
        )

        duration = None
        if self._start_time:
            duration = (datetime.utcnow() - self._start_time).total_seconds()

        return {
            "name": self.name,
            "version": self.version,
            "evaluation_count": self._evaluation_count,
            "average_score": avg_score,
            "total_score": self._total_score,
            "duration_seconds": duration,
            "evaluations_per_second": (
                self._evaluation_count / duration
                if duration and duration > 0
                else 0.0
            )
        }

    def reset_stats(self) -> None:
        """Reset evaluation statistics."""
        self._evaluation_count = 0
        self._total_score = 0.0
        self._start_time = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', version='{self.version}')"