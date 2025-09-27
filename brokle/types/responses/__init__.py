"""
Response models package for Brokle SDK.

This package contains modular response models organized by domain following
industry standard patterns with clean namespace separation via response.brokle.*
"""

# Import and re-export core models using industry standard pattern

# Import base classes and mixins
from .base import (
    BrokleResponseBase,
    TimestampMixin,
    MetadataMixin,
    TokenUsageMixin,
    CostTrackingMixin,
    PaginationMixin,
    ProviderMixin,
    RequestTrackingMixin,
    OrganizationContextMixin,
    StatusMixin,
    TimestampedResponse,
    ProviderResponse,
    PaginatedResponse,
    TrackedResponse,
    FullContextResponse,

    # Industry standard response models
    BaseResponse,
    BrokleMetadata,
)

# Import core AI models
from .core import (
    # Core Response Models
    ChatCompletionResponse,
    EmbeddingResponse,
    CompletionResponse,

    # Supporting Models
    ChatCompletionMessage,
    ChatCompletionChoice,
    EmbeddingData,
    CompletionChoice,

)

# Import telemetry models
from .telemetry import (
    # Telemetry Response Models
    TelemetryTraceResponse,
    TelemetrySpanResponse,
    TelemetryEventBatchResponse,


    # New telemetry models
    TelemetryPerformanceResponse,

    # Telemetry mixins
    TelemetryTimingMixin,
)

# Import billing models
from .billing import (
    # Billing & Cost Response Models
    CostCalculationResponse,
    CostTrackingResponse,
    BudgetResponse,
    CostComparisonResponse,
    CostTrendResponse,
    UsageRecordingResponse,
    QuotaCheckResponse,
    BillingMetricsResponse,


    # Billing-specific mixins
    BudgetPeriodMixin,
    UsageStatsMixin,
)

# Import observability models
from .observability import (
    # Observability Response Models
    ObservabilityTraceResponse,
    ObservabilityObservationResponse,
    ObservabilityQualityScoreResponse,
    ObservabilityStatsResponse,
    ObservabilityListResponse,
    ObservabilityBatchResponse,

    # Analytics & Evaluation Models
    AnalyticsResponse,
    EvaluationResponse,

    # Supporting models
    AnalyticsMetric,
    EvaluationScore,

    # Observability mixins
    ObservabilityTimingMixin,
    QualityScoreMixin,
    ObservabilityStatsCore,
)

# Import remaining models
from .remaining import (
    # Error Handling
    ErrorResponse,
    APIResponse,

    # Caching
    CacheResponse,
    CacheStatsResponse,

    # Embeddings & Search
    EmbeddingGenerationResponse,
    SemanticSearchResponse,

    # ML & Routing
    MLRoutingResponse,
    MLModelInfoResponse,

    # Configuration
    ConfigResponse,
    FeatureFlagResponse,

    # Usage & Billing
    SubscriptionLimitResponse,

    # Notifications
    NotificationResponse,
    NotificationStatusResponse,
    NotificationHistoryResponse,


    # Remaining mixins
    CacheMetricsMixin,
    NotificationDeliveryMixin,
    SearchResultMixin,
)


__all__ = [
    # Phase 1: Base classes and mixins
    'BrokleResponseBase',
    'TimestampMixin',
    'MetadataMixin',
    'TokenUsageMixin',
    'CostTrackingMixin',
    'PaginationMixin',
    'ProviderMixin',
    'RequestTrackingMixin',
    'OrganizationContextMixin',
    'StatusMixin',
    'TimestampedResponse',
    'ProviderResponse',
    'PaginatedResponse',
    'TrackedResponse',
    'FullContextResponse',

    # Phase 2: Core AI Response Models
    'ChatCompletionResponse',
    'EmbeddingResponse',
    'CompletionResponse',
    'ChatCompletionMessage',
    'ChatCompletionChoice',
    'EmbeddingData',
    'CompletionChoice',


    # Phase 3: Telemetry & Tracing Models
    'TelemetryTraceResponse',
    'TelemetrySpanResponse',
    'TelemetryEventBatchResponse',

    'TelemetryPerformanceResponse',

    # Telemetry mixins
    'TelemetryTimingMixin',

    # Phase 4: Billing & Cost Models
    'CostCalculationResponse',
    'CostTrackingResponse',
    'BudgetResponse',
    'CostComparisonResponse',
    'CostTrendResponse',
    'UsageRecordingResponse',
    'QuotaCheckResponse',
    'BillingMetricsResponse',


    # Billing mixins
    'BudgetPeriodMixin',
    'UsageStatsMixin',

    # Phase 5: Observability & Analytics Models
    'ObservabilityTraceResponse',
    'ObservabilityObservationResponse',
    'ObservabilityQualityScoreResponse',
    'ObservabilityStatsResponse',
    'ObservabilityListResponse',
    'ObservabilityBatchResponse',
    'AnalyticsResponse',
    'EvaluationResponse',

    # Supporting models
    'AnalyticsMetric',
    'EvaluationScore',


    # Observability mixins
    'ObservabilityTimingMixin',
    'QualityScoreMixin',
    'ObservabilityStatsCore',

    # Phase 6: Remaining Models
    'ErrorResponse', 'APIResponse',
    'CacheResponse', 'CacheStatsResponse',
    'EmbeddingGenerationResponse', 'SemanticSearchResponse',
    'MLRoutingResponse', 'MLModelInfoResponse',
    'ConfigResponse', 'FeatureFlagResponse',
    'SubscriptionLimitResponse',
    'NotificationResponse', 'NotificationStatusResponse', 'NotificationHistoryResponse',


    # Remaining mixins
    'CacheMetricsMixin',
    'NotificationDeliveryMixin',
    'SearchResultMixin',

    # Industry standard response models
    'BaseResponse',
    'BrokleMetadata',
]