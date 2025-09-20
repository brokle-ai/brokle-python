""".. include:: ../README.md"""

from .client import Brokle, get_client
from .config import Config
from .auth import AuthManager
from ._client.attributes import BrokleOtelSpanAttributes
from .integrations import auto_instrument, print_status, get_status, get_registry
from ._version import __version__

# Auto-instrumentation convenience imports
# Note: These imports don't auto-instrument by default - you need to explicitly import the modules
# For auto-instrumentation, use:
#   import brokle.openai  # Auto-instruments OpenAI
#   import brokle.integrations.openai  # Alternative import for auto-instrumentation

# Exception classes
from .exceptions import (
    BrokleError,
    AuthenticationError,
    RateLimitError,
    ConfigurationError,
    APIError,
    NetworkError,
    ValidationError,
    TimeoutError,
    UnsupportedOperationError,
    QuotaExceededError,
    ProviderError,
    CacheError,
    EvaluationError,
)

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

# AI Platform exports
from .ai_platform import (
    # Core AI client
    AIClient, get_ai_client, configure_ai_platform, generate, generate_stream,

    # Routing
    RoutingStrategy, RoutingConfig, ProviderConfig, ProviderTier,
    create_cost_optimized_routing, create_quality_optimized_routing,

    # Caching
    CacheStrategy, CacheConfig, SemanticCacheConfig,
    create_semantic_cache_config, get_cache_stats,

    # Quality
    QualityMetric, QualityConfig, QualityScore,
    create_comprehensive_quality, evaluate_quality,

    # Optimization
    OptimizationStrategy, CostOptimizationConfig, BudgetConfig,
    create_balanced_optimization, get_cost_breakdown,

    # Providers
    ProviderStatus, ProviderHealth, get_provider_health,
    get_healthy_providers, get_provider_rankings,
)

# Main exports
__all__ = [
    # Core client and observability
    "Brokle",
    "get_client",
    "Config",
    "AuthManager",
    "BrokleOtelSpanAttributes",
    # Exceptions
    "BrokleError",
    "AuthenticationError",
    "RateLimitError",
    "ConfigurationError",
    "APIError",
    "NetworkError",
    "ValidationError",
    "TimeoutError",
    "UnsupportedOperationError",
    "QuotaExceededError",
    "ProviderError",
    "CacheError",
    "EvaluationError",
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
    # AI Platform - Core
    "AIClient",
    "get_ai_client",
    "configure_ai_platform",
    "generate",
    "generate_stream",
    # AI Platform - Routing
    "RoutingStrategy",
    "RoutingConfig",
    "ProviderConfig",
    "ProviderTier",
    "create_cost_optimized_routing",
    "create_quality_optimized_routing",
    # AI Platform - Caching
    "CacheStrategy",
    "CacheConfig",
    "SemanticCacheConfig",
    "create_semantic_cache_config",
    "get_cache_stats",
    # AI Platform - Quality
    "QualityMetric",
    "QualityConfig",
    "QualityScore",
    "create_comprehensive_quality",
    "evaluate_quality",
    # AI Platform - Optimization
    "OptimizationStrategy",
    "CostOptimizationConfig",
    "BudgetConfig",
    "create_balanced_optimization",
    "get_cost_breakdown",
    # AI Platform - Providers
    "ProviderStatus",
    "ProviderHealth",
    "get_provider_health",
    "get_healthy_providers",
    "get_provider_rankings",
    # Version
    "__version__",
]