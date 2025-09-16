"""
AI quality scoring configuration and backend integration.

This module provides client-side quality configuration that integrates
with Brokle's backend quality scoring service with simple fallbacks
when the backend is unavailable.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class QualityMetric(Enum):
    """Quality metrics supported by Brokle platform."""
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    FLUENCY = "fluency"
    COMPLETENESS = "completeness"
    FACTUALITY = "factuality"
    SAFETY = "safety"
    BIAS = "bias"
    TOXICITY = "toxicity"
    CUSTOM = "custom"


class QualityThreshold(Enum):
    """Quality threshold levels."""
    STRICT = "strict"      # 90%+
    HIGH = "high"          # 80%+
    MEDIUM = "medium"      # 70%+
    LOW = "low"            # 60%+
    CUSTOM = "custom"      # User-defined


@dataclass
class QualityScore:
    """Quality score result with detailed breakdown."""
    overall_score: float
    metric_scores: Dict[str, float] = field(default_factory=dict)

    # Metadata
    evaluation_time_ms: Optional[float] = None
    model_used: Optional[str] = None
    confidence: Optional[float] = None

    # Detailed analysis
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    # Context
    input_text: Optional[str] = None
    output_text: Optional[str] = None
    reference_text: Optional[str] = None

    @property
    def passed_threshold(self) -> Optional[bool]:
        """Check if score meets threshold (if set)."""
        return getattr(self, '_threshold_passed', None)

    def set_threshold_result(self, passed: bool) -> None:
        """Set threshold result."""
        self._threshold_passed = passed


@dataclass
class MetricConfig:
    """Configuration for individual quality metrics."""
    enabled: bool = True
    weight: float = 1.0
    threshold: float = 0.7

    # Metric-specific settings
    use_reference: bool = False
    custom_prompt: Optional[str] = None
    model_override: Optional[str] = None

    # Performance settings
    timeout_seconds: float = 10.0
    cache_results: bool = True


@dataclass
class EvaluationConfig:
    """Configuration for quality evaluation automation."""
    enabled: bool = True

    # Evaluation triggers
    evaluate_all_requests: bool = False
    sample_rate: float = 0.1  # 10% of requests
    evaluate_on_threshold: bool = True
    evaluate_on_error: bool = True

    # Real-time evaluation
    enable_realtime: bool = True
    realtime_timeout_ms: float = 500.0

    # Batch evaluation
    enable_batch: bool = True
    batch_size: int = 100
    batch_interval_minutes: int = 60

    # Quality gates
    enable_quality_gates: bool = False
    block_on_low_quality: bool = False
    min_quality_score: float = 0.6


@dataclass
class QualityConfig:
    """Complete quality configuration for Brokle AI platform."""
    enabled: bool = True

    # Metrics configuration
    metrics: Dict[QualityMetric, MetricConfig] = field(default_factory=dict)

    # Thresholds
    overall_threshold: float = 0.7
    threshold_mode: QualityThreshold = QualityThreshold.MEDIUM

    # Evaluation settings
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Performance
    parallel_evaluation: bool = True
    max_concurrent_evaluations: int = 10

    # Fallback behavior
    fallback_to_local: bool = True
    local_evaluation_model: str = "gpt-3.5-turbo"

    # Analytics
    track_quality_trends: bool = True
    quality_alerts: bool = True

    def add_metric(
        self,
        metric: QualityMetric,
        weight: float = 1.0,
        threshold: float = 0.7,
        **kwargs
    ) -> None:
        """Add quality metric configuration."""
        self.metrics[metric] = MetricConfig(
            weight=weight,
            threshold=threshold,
            **kwargs
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "enabled": self.enabled,
            "metrics": {
                metric.value: {
                    "enabled": config.enabled,
                    "weight": config.weight,
                    "threshold": config.threshold,
                    "use_reference": config.use_reference,
                    "custom_prompt": config.custom_prompt,
                    "model_override": config.model_override,
                    "timeout_seconds": config.timeout_seconds,
                    "cache_results": config.cache_results
                }
                for metric, config in self.metrics.items()
            },
            "overall_threshold": self.overall_threshold,
            "threshold_mode": self.threshold_mode.value,
            "evaluation": {
                "enabled": self.evaluation.enabled,
                "evaluate_all_requests": self.evaluation.evaluate_all_requests,
                "sample_rate": self.evaluation.sample_rate,
                "evaluate_on_threshold": self.evaluation.evaluate_on_threshold,
                "evaluate_on_error": self.evaluation.evaluate_on_error,
                "enable_realtime": self.evaluation.enable_realtime,
                "realtime_timeout_ms": self.evaluation.realtime_timeout_ms,
                "enable_batch": self.evaluation.enable_batch,
                "batch_size": self.evaluation.batch_size,
                "batch_interval_minutes": self.evaluation.batch_interval_minutes,
                "enable_quality_gates": self.evaluation.enable_quality_gates,
                "block_on_low_quality": self.evaluation.block_on_low_quality,
                "min_quality_score": self.evaluation.min_quality_score
            },
            "parallel_evaluation": self.parallel_evaluation,
            "max_concurrent_evaluations": self.max_concurrent_evaluations,
            "fallback_to_local": self.fallback_to_local,
            "local_evaluation_model": self.local_evaluation_model,
            "track_quality_trends": self.track_quality_trends,
            "quality_alerts": self.quality_alerts
        }


class QualityEvaluator:
    """
    Client-side quality configuration manager.

    Manages quality configuration and coordinates with Brokle's backend
    quality scoring service, providing simple fallbacks when offline.
    """

    def __init__(self):
        self.config = QualityConfig()
        self._backend_available = True
        self._evaluation_cache: Dict[str, QualityScore] = {}

    def configure(self, config: QualityConfig) -> None:
        """Configure quality evaluation settings."""
        self.config = config
        logger.info(f"Quality evaluation configured with {len(config.metrics)} metrics")

    def get_config(self) -> QualityConfig:
        """Get current quality configuration."""
        return self.config

    async def evaluate_response(
        self,
        input_text: str,
        output_text: str,
        reference_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> QualityScore:
        """Evaluate AI response quality."""
        if not self.config.enabled:
            return self._create_default_score(input_text, output_text)

        # Try backend evaluation first
        if self._backend_available:
            try:
                score = await self._evaluate_with_backend(
                    input_text, output_text, reference_text, metadata
                )
                if score:
                    return score
            except Exception as e:
                logger.warning(f"Backend quality evaluation failed: {e}")
                self._backend_available = False

        # Fallback to simple local evaluation
        if self.config.fallback_to_local:
            return self._create_fallback_score(input_text, output_text, reference_text)

        return self._create_default_score(input_text, output_text)

    async def _evaluate_with_backend(
        self,
        input_text: str,
        output_text: str,
        reference_text: Optional[str],
        metadata: Optional[Dict[str, Any]]
    ) -> Optional[QualityScore]:
        """Evaluate using backend quality scoring service."""
        try:
            # TODO: Implement actual backend API call
            # Example: POST /api/v1/quality/evaluate with config and text
            payload = {
                "input_text": input_text,
                "output_text": output_text,
                "reference_text": reference_text,
                "config": self.config.to_dict(),
                "metadata": metadata or {}
            }

            # Placeholder for actual HTTP request
            logger.debug(f"Would call backend API: POST /api/v1/quality/evaluate")
            logger.debug(f"Payload size: {len(str(payload))} chars")

            # Return None to trigger fallback
            return None

        except Exception as e:
            logger.warning(f"Backend quality evaluation failed: {e}")
            return None

    def _create_fallback_score(
        self,
        input_text: str,
        output_text: str,
        reference_text: Optional[str] = None
    ) -> QualityScore:
        """Create simple fallback quality score when backend unavailable."""
        start_time = datetime.now()

        # Very basic quality heuristics
        if not output_text.strip():
            overall_score = 0.0
        elif len(output_text) < 10:
            overall_score = 0.3
        elif reference_text and output_text.lower() in reference_text.lower():
            overall_score = 0.9  # Good match with reference
        else:
            overall_score = 0.7  # Reasonable default

        # Simple metric breakdown
        metric_scores = {
            "overall": overall_score,
            "completeness": 0.8 if len(output_text) > 20 else 0.5,
            "basic_quality": overall_score
        }

        evaluation_time = (datetime.now() - start_time).total_seconds() * 1000

        quality_score = QualityScore(
            overall_score=overall_score,
            metric_scores=metric_scores,
            evaluation_time_ms=evaluation_time,
            model_used="fallback",
            input_text=input_text,
            output_text=output_text,
            reference_text=reference_text
        )

        # Check threshold
        passed = overall_score >= self.config.overall_threshold
        quality_score.set_threshold_result(passed)

        return quality_score

    def _create_default_score(self, input_text: str, output_text: str) -> QualityScore:
        """Create default quality score when evaluation is disabled."""
        return QualityScore(
            overall_score=0.8,  # Neutral score
            input_text=input_text,
            output_text=output_text
        )

    def create_accuracy_focused_config(self) -> QualityConfig:
        """Create accuracy-focused quality configuration."""
        config = QualityConfig()
        config.add_metric(QualityMetric.ACCURACY, weight=2.0, threshold=0.8)
        config.add_metric(QualityMetric.FACTUALITY, weight=1.5, threshold=0.85)
        config.add_metric(QualityMetric.COMPLETENESS, weight=1.0, threshold=0.7)
        config.overall_threshold = 0.8
        return config

    def create_safety_focused_config(self) -> QualityConfig:
        """Create safety-focused quality configuration."""
        config = QualityConfig()
        config.add_metric(QualityMetric.SAFETY, weight=3.0, threshold=0.95)
        config.add_metric(QualityMetric.BIAS, weight=2.0, threshold=0.9)
        config.add_metric(QualityMetric.TOXICITY, weight=2.0, threshold=0.95)
        config.overall_threshold = 0.9
        config.evaluation.block_on_low_quality = True
        return config

    def create_comprehensive_config(self) -> QualityConfig:
        """Create comprehensive quality configuration."""
        config = QualityConfig()
        config.add_metric(QualityMetric.ACCURACY, weight=1.5, threshold=0.8)
        config.add_metric(QualityMetric.RELEVANCE, weight=1.5, threshold=0.8)
        config.add_metric(QualityMetric.COHERENCE, weight=1.0, threshold=0.75)
        config.add_metric(QualityMetric.FLUENCY, weight=1.0, threshold=0.8)
        config.add_metric(QualityMetric.COMPLETENESS, weight=1.0, threshold=0.7)
        config.add_metric(QualityMetric.SAFETY, weight=2.0, threshold=0.9)
        config.overall_threshold = 0.75
        return config

    def get_quality_stats(self) -> Dict[str, Any]:
        """Get quality evaluation statistics."""
        return {
            "config": self.config.to_dict(),
            "backend_available": self._backend_available,
            "cache_size": len(self._evaluation_cache),
            "metrics_enabled": len([m for m in self.config.metrics.values() if m.enabled])
        }


# Global quality evaluator instance
_quality_evaluator: Optional[QualityEvaluator] = None


def get_quality_evaluator() -> QualityEvaluator:
    """Get global quality evaluator instance."""
    global _quality_evaluator

    if _quality_evaluator is None:
        _quality_evaluator = QualityEvaluator()

    return _quality_evaluator


def configure_quality(config: QualityConfig) -> None:
    """Configure global quality evaluation settings."""
    evaluator = get_quality_evaluator()
    evaluator.configure(config)


def get_quality_config() -> QualityConfig:
    """Get current quality configuration."""
    evaluator = get_quality_evaluator()
    return evaluator.get_config()


def create_accuracy_focused_quality() -> QualityConfig:
    """Create accuracy-focused quality configuration."""
    evaluator = get_quality_evaluator()
    return evaluator.create_accuracy_focused_config()


def create_safety_focused_quality() -> QualityConfig:
    """Create safety-focused quality configuration."""
    evaluator = get_quality_evaluator()
    return evaluator.create_safety_focused_config()


def create_comprehensive_quality() -> QualityConfig:
    """Create comprehensive quality configuration."""
    evaluator = get_quality_evaluator()
    return evaluator.create_comprehensive_config()


async def evaluate_quality(
    input_text: str,
    output_text: str,
    reference_text: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> QualityScore:
    """Evaluate AI response quality."""
    evaluator = get_quality_evaluator()
    return await evaluator.evaluate_response(input_text, output_text, reference_text, metadata)


def get_quality_stats() -> Dict[str, Any]:
    """Get quality evaluation statistics."""
    evaluator = get_quality_evaluator()
    return evaluator.get_quality_stats()