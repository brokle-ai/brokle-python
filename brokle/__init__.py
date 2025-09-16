""".. include:: ../README.md"""

# Core imports following LangFuse pattern
from .client import Brokle, get_client
from .config import Config, configure, get_config, reset_config
from .auth import AuthManager
from .decorators import observe
from ._client.attributes import BrokleOtelSpanAttributes
from .integrations import auto_instrument, print_status, get_status, get_registry
from ._version import __version__

# Evaluation framework exports
from .evaluation import (
    evaluate,
    aevaluate,
    BaseEvaluator,
    EvaluationResult,
    EvaluationConfig,
    AccuracyEvaluator,
    RelevanceEvaluator,
    CostEfficiencyEvaluator,
    LatencyEvaluator,
    QualityEvaluator,
)

# Main exports - clean and minimal like LangFuse
__all__ = [
    # Core client and observability
    "Brokle",
    "get_client",
    "observe",
    "Config",
    "configure",
    "get_config",
    "reset_config",
    "AuthManager",
    "BrokleOtelSpanAttributes",
    # Integrations
    "auto_instrument",
    "print_status",
    "get_status",
    "get_registry",
    # Evaluation framework
    "evaluate",
    "aevaluate",
    "BaseEvaluator",
    "EvaluationResult",
    "EvaluationConfig",
    "AccuracyEvaluator",
    "RelevanceEvaluator",
    "CostEfficiencyEvaluator",
    "LatencyEvaluator",
    "QualityEvaluator",
    # Version
    "__version__",
]