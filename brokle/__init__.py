"""
Brokle SDK - The Open-Source AI Control Plane

Comprehensive Python SDK providing intelligent routing, cost optimization,
semantic caching, and observability for AI applications.

Three Integration Patterns:

1. **Native SDK** (Full AI Platform Features):
   ```python
   from brokle import Brokle, get_client

   client = Brokle(api_key="ak_...")
   response = await client.chat.create(
       model="gpt-4",
       messages=[{"role": "user", "content": "Hello!"}]
   )
   ```

2. **Drop-in Replacement** (Pure Observability):
   ```python
   # Instead of: from openai import OpenAI
   from brokle.openai import OpenAI

   client = OpenAI(api_key="sk-...")
   response = client.chat.completions.create(...)
   ```

3. **Universal Decorator** (Framework-Agnostic):
   ```python
   from brokle import observe

   @observe()
   def my_ai_workflow(user_query: str) -> str:
       return llm.generate(user_query)
   ```
"""

# Core client and configuration
from .client import Brokle, get_client
from .config import Config
from .auth import AuthManager
from ._client.attributes import BrokleOtelSpanAttributes

# Universal decorator pattern
from .decorators import (
    observe,
    trace_workflow,
    observe_llm,
    observe_retrieval,
    ObserveConfig,
)

from ._version import __version__

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

# AI Platform exports (Native SDK with full features)
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

# Main exports - Clean 3-Pattern Architecture
__all__ = [
    # === PATTERN 1: NATIVE SDK (Full AI Platform Features) ===
    "Brokle",                    # Main client class
    "get_client",                # Singleton client accessor
    "Config",                    # Configuration management
    "AuthManager",               # Authentication handling
    "BrokleOtelSpanAttributes",  # Telemetry attributes

    # === PATTERN 2: UNIVERSAL DECORATOR (Framework-Agnostic) ===
    "observe",                   # Universal @observe() decorator
    "trace_workflow",            # Workflow context manager
    "observe_llm",               # LLM-specific decorator
    "observe_retrieval",         # Retrieval-specific decorator
    "ObserveConfig",             # Decorator configuration

    # === PATTERN 3: DROP-IN REPLACEMENTS (Pure Observability) ===
    # Note: Import separately from brokle.openai, brokle.anthropic, etc.
    # Example: from brokle.openai import OpenAI

    # === SHARED: EXCEPTION CLASSES ===
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

    # === SHARED: EVALUATION FRAMEWORK ===
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

    # === NATIVE SDK: AI PLATFORM FEATURES ===
    # Core
    "AIClient",
    "get_ai_client",
    "configure_ai_platform",
    "generate",
    "generate_stream",
    # Routing
    "RoutingStrategy",
    "RoutingConfig",
    "ProviderConfig",
    "ProviderTier",
    "create_cost_optimized_routing",
    "create_quality_optimized_routing",
    # Caching
    "CacheStrategy",
    "CacheConfig",
    "SemanticCacheConfig",
    "create_semantic_cache_config",
    "get_cache_stats",
    # Quality
    "QualityMetric",
    "QualityConfig",
    "QualityScore",
    "create_comprehensive_quality",
    "evaluate_quality",
    # Optimization
    "OptimizationStrategy",
    "CostOptimizationConfig",
    "BudgetConfig",
    "create_balanced_optimization",
    "get_cost_breakdown",
    # Providers
    "ProviderStatus",
    "ProviderHealth",
    "get_provider_health",
    "get_healthy_providers",
    "get_provider_rankings",

    # === METADATA ===
    "__version__",
]