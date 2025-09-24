"""
Brokle SDK 2.0 - The Open-Source AI Control Plane

BREAKING CHANGES in v2.0.0:
- Removed drop-in replacements (brokle.openai.OpenAI, brokle.anthropic.Anthropic)
- Added explicit wrapper functions (wrap_openai, wrap_anthropic)
- Enhanced 3-pattern integration system

Migration Guide:
1.x: from brokle.openai import OpenAI
2.x: from openai import OpenAI; from brokle import wrap_openai; client = wrap_openai(OpenAI())

Three Integration Patterns:

1. **Wrapper Functions** (LangSmith/Optik Style):
   ```python
   from openai import OpenAI
   from anthropic import Anthropic
   from brokle import wrap_openai, wrap_anthropic

   openai_client = wrap_openai(OpenAI(api_key="sk-..."))
   anthropic_client = wrap_anthropic(Anthropic(api_key="sk-ant-..."))
   response = openai_client.chat.completions.create(...)
   ```

2. **Universal Decorator** (Framework-Agnostic):
   ```python
   from brokle import observe

   @observe()
   def my_ai_workflow(user_query: str) -> str:
       return llm.generate(user_query)
   ```

3. **Native SDK** (Full AI Platform Features):
   ```python
   from brokle import Brokle, get_client

   client = Brokle(api_key="ak_...")
   response = await client.chat.create(
       model="gpt-4",
       messages=[{"role": "user", "content": "Hello!"}]
   )
   ```
"""

# === PATTERN 1: WRAPPER FUNCTIONS (NEW IN 2.0) ===
from .wrappers import (
    wrap_openai,
    wrap_anthropic,
    wrap_google,    # Future
    wrap_cohere,    # Future
)

# === PATTERN 2: UNIVERSAL DECORATOR (UNCHANGED) ===
from .decorators import (
    observe,
    trace_workflow,
    observe_llm,
    observe_retrieval,
    ObserveConfig,
)

# === PATTERN 3: NATIVE SDK (UNCHANGED) ===
from .client import Brokle, get_client
from .config import Config
from .auth import AuthManager
from ._client.attributes import BrokleOtelSpanAttributes

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

# Main exports - Clean 3-Pattern Architecture (v2.0.0)
__all__ = [
    # === PATTERN 1: WRAPPER FUNCTIONS (LangSmith/Optik Style) ===
    "wrap_openai",               # OpenAI client wrapper
    "wrap_anthropic",            # Anthropic client wrapper
    "wrap_google",               # Google AI wrapper (future)
    "wrap_cohere",               # Cohere wrapper (future)

    # === PATTERN 2: UNIVERSAL DECORATOR (Framework-Agnostic) ===
    "observe",                   # Universal @observe() decorator
    "trace_workflow",            # Workflow context manager
    "observe_llm",               # LLM-specific decorator
    "observe_retrieval",         # Retrieval-specific decorator
    "ObserveConfig",             # Decorator configuration

    # === PATTERN 3: NATIVE SDK (Full AI Platform Features) ===
    "Brokle",                    # Main client class
    "get_client",                # Singleton client accessor
    "Config",                    # Configuration management
    "AuthManager",               # Authentication handling
    "BrokleOtelSpanAttributes",  # Telemetry attributes

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


# Deprecation warnings for old imports (BREAKING CHANGE)
def __getattr__(name: str):
    """Handle deprecated imports with helpful error messages."""
    deprecated_imports = {
        'openai': (
            'BREAKING CHANGE in Brokle 2.0: brokle.openai is deprecated.\n'
            'Migration: from openai import OpenAI; from brokle import wrap_openai; client = wrap_openai(OpenAI())\n'
            'See migration guide: https://docs.brokle.ai/migration/v2'
        ),
        'anthropic': (
            'BREAKING CHANGE in Brokle 2.0: brokle.anthropic is deprecated.\n'
            'Migration: from anthropic import Anthropic; from brokle import wrap_anthropic; client = wrap_anthropic(Anthropic())\n'
            'See migration guide: https://docs.brokle.ai/migration/v2'
        ),
    }

    if name in deprecated_imports:
        raise ImportError(deprecated_imports[name])

    raise AttributeError(f"module 'brokle' has no attribute '{name}'")