"""
Brokle AI Platform client features.

This module provides client-side configuration and integration with
Brokle's AI platform services including intelligent routing, semantic caching,
quality scoring, and cost optimization.

The SDK acts as a thin wrapper that:
- Configures user preferences and passes them to backend
- Provides fallback modes when backend is unavailable
- Offers local lightweight features for resilience
- Maintains full OTEL tracing context
"""

from .routing import (
    RoutingStrategy,
    RoutingConfig,
    ProviderConfig,
    FallbackConfig,
    ProviderTier,
    configure_routing,
    get_routing_config,
    create_cost_optimized_routing,
    create_quality_optimized_routing,
    create_latency_optimized_routing,
    add_provider
)

from .caching import (
    CacheStrategy,
    CacheLevel,
    CacheConfig,
    SemanticCacheConfig,
    LocalCacheConfig,
    configure_caching,
    get_cache_config,
    create_semantic_cache_config,
    create_exact_cache_config,
    get_cache_stats
)

from .quality import (
    QualityMetric,
    QualityThreshold,
    QualityConfig,
    QualityScore,
    EvaluationConfig,
    configure_quality,
    get_quality_config,
    create_accuracy_focused_quality,
    create_safety_focused_quality,
    create_comprehensive_quality,
    evaluate_quality,
    get_quality_stats
)

from .optimization import (
    OptimizationStrategy,
    BudgetPeriod,
    CostUnit,
    BudgetAlert,
    CostOptimizationConfig,
    BudgetConfig,
    UsageConfig,
    CostBreakdown,
    configure_optimization,
    get_optimization_config,
    track_request_cost,
    get_cost_breakdown,
    create_aggressive_optimization,
    create_balanced_optimization,
    create_conservative_optimization,
    get_optimization_stats
)

from .providers import (
    ProviderStatus,
    HealthCheckType,
    ProviderHealth,
    ProviderMetrics,
    get_provider_status,
    get_provider_health,
    record_provider_request,
    get_all_provider_health,
    get_healthy_providers,
    get_provider_rankings,
    reset_provider_metrics
)

from .client import (
    AIRequest,
    AIResponse,
    AIClient,
    get_ai_client,
    configure_ai_platform,
    generate,
    generate_stream
)

__all__ = [
    # Routing
    "RoutingStrategy",
    "RoutingConfig",
    "ProviderConfig",
    "FallbackConfig",
    "ProviderTier",
    "configure_routing",
    "get_routing_config",
    "create_cost_optimized_routing",
    "create_quality_optimized_routing",
    "create_latency_optimized_routing",
    "add_provider",

    # Caching
    "CacheStrategy",
    "CacheLevel",
    "CacheConfig",
    "SemanticCacheConfig",
    "LocalCacheConfig",
    "configure_caching",
    "get_cache_config",
    "create_semantic_cache_config",
    "create_exact_cache_config",
    "get_cache_stats",

    # Quality
    "QualityMetric",
    "QualityThreshold",
    "QualityConfig",
    "QualityScore",
    "EvaluationConfig",
    "configure_quality",
    "get_quality_config",
    "create_accuracy_focused_quality",
    "create_safety_focused_quality",
    "create_comprehensive_quality",
    "evaluate_quality",
    "get_quality_stats",

    # Optimization
    "OptimizationStrategy",
    "BudgetPeriod",
    "CostUnit",
    "BudgetAlert",
    "CostOptimizationConfig",
    "BudgetConfig",
    "UsageConfig",
    "CostBreakdown",
    "configure_optimization",
    "get_optimization_config",
    "track_request_cost",
    "get_cost_breakdown",
    "create_aggressive_optimization",
    "create_balanced_optimization",
    "create_conservative_optimization",
    "get_optimization_stats",

    # Providers
    "ProviderStatus",
    "HealthCheckType",
    "ProviderHealth",
    "ProviderMetrics",
    "get_provider_status",
    "get_provider_health",
    "record_provider_request",
    "get_all_provider_health",
    "get_healthy_providers",
    "get_provider_rankings",
    "reset_provider_metrics",

    # Client
    "AIRequest",
    "AIResponse",
    "AIClient",
    "get_ai_client",
    "configure_ai_platform",
    "generate",
    "generate_stream"
]