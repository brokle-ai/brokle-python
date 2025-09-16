"""
LangSmith-inspired evaluation framework for Brokle SDK.

This module provides comprehensive evaluation capabilities for AI applications,
following LangSmith patterns with Brokle-specific enhancements.
"""

from .base import BaseEvaluator, EvaluationResult, EvaluationConfig
from .evaluate import evaluate, aevaluate
from .metrics import (
    AccuracyEvaluator,
    RelevanceEvaluator,
    CostEfficiencyEvaluator,
    LatencyEvaluator,
    QualityEvaluator,
)

__all__ = [
    # Core evaluation functions
    "evaluate",
    "aevaluate",
    # Base classes
    "BaseEvaluator",
    "EvaluationResult",
    "EvaluationConfig",
    # Built-in evaluators
    "AccuracyEvaluator",
    "RelevanceEvaluator",
    "CostEfficiencyEvaluator",
    "LatencyEvaluator",
    "QualityEvaluator",
]