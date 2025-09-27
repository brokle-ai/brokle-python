"""
Response models package for Brokle SDK.

This package contains modular response models organized by domain following
industry standard patterns with clean namespace separation via response.brokle.*
"""

# Import and re-export core models using industry standard pattern

# Import base classes and mixins
from .base import (  # Industry standard response models
    BaseResponse,
    BrokleMetadata,
    BrokleResponseBase,
    CostTrackingMixin,
    FullContextResponse,
    MetadataMixin,
    OrganizationContextMixin,
    PaginatedResponse,
    PaginationMixin,
    ProviderMixin,
    ProviderResponse,
    RequestTrackingMixin,
    StatusMixin,
    TimestampedResponse,
    TimestampMixin,
    TokenUsageMixin,
    TrackedResponse,
)

# Import billing models
from .billing import (  # Billing & Cost Response Models; Billing-specific mixins
    BillingMetricsResponse,
    BudgetPeriodMixin,
    BudgetResponse,
    CostCalculationResponse,
    CostComparisonResponse,
    CostTrackingResponse,
    CostTrendResponse,
    QuotaCheckResponse,
    UsageRecordingResponse,
    UsageStatsMixin,
)

# Import core AI models
from .core import (  # Core Response Models; Supporting Models
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionResponse,
    CompletionChoice,
    CompletionResponse,
    EmbeddingData,
    EmbeddingResponse,
)

# Import observability models
from .observability import (  # Observability Response Models; Analytics & Evaluation Models; Supporting models; Observability mixins
    AnalyticsMetric,
    AnalyticsResponse,
    EvaluationResponse,
    EvaluationScore,
    ObservabilityBatchResponse,
    ObservabilityListResponse,
    ObservabilityObservationResponse,
    ObservabilityQualityScoreResponse,
    ObservabilityStatsCore,
    ObservabilityStatsResponse,
    ObservabilityTimingMixin,
    ObservabilityTraceResponse,
    QualityScoreMixin,
)

# Import remaining models
from .remaining import (  # Error Handling; Caching; Embeddings & Search; ML & Routing; Configuration; Usage & Billing; Notifications; Remaining mixins
    APIResponse,
    CacheMetricsMixin,
    CacheResponse,
    CacheStatsResponse,
    ConfigResponse,
    EmbeddingGenerationResponse,
    ErrorResponse,
    FeatureFlagResponse,
    MLModelInfoResponse,
    MLRoutingResponse,
    NotificationDeliveryMixin,
    NotificationHistoryResponse,
    NotificationResponse,
    NotificationStatusResponse,
    SearchResultMixin,
    SemanticSearchResponse,
    SubscriptionLimitResponse,
)

# Import telemetry models
from .telemetry import (  # Telemetry Response Models; New telemetry models; Telemetry mixins
    TelemetryEventBatchResponse,
    TelemetryPerformanceResponse,
    TelemetrySpanResponse,
    TelemetryTimingMixin,
    TelemetryTraceResponse,
)

__all__ = [
    # Phase 1: Base classes and mixins
    "BrokleResponseBase",
    "TimestampMixin",
    "MetadataMixin",
    "TokenUsageMixin",
    "CostTrackingMixin",
    "PaginationMixin",
    "ProviderMixin",
    "RequestTrackingMixin",
    "OrganizationContextMixin",
    "StatusMixin",
    "TimestampedResponse",
    "ProviderResponse",
    "PaginatedResponse",
    "TrackedResponse",
    "FullContextResponse",
    # Phase 2: Core AI Response Models
    "ChatCompletionResponse",
    "EmbeddingResponse",
    "CompletionResponse",
    "ChatCompletionMessage",
    "ChatCompletionChoice",
    "EmbeddingData",
    "CompletionChoice",
    # Phase 3: Telemetry & Tracing Models
    "TelemetryTraceResponse",
    "TelemetrySpanResponse",
    "TelemetryEventBatchResponse",
    "TelemetryPerformanceResponse",
    # Telemetry mixins
    "TelemetryTimingMixin",
    # Phase 4: Billing & Cost Models
    "CostCalculationResponse",
    "CostTrackingResponse",
    "BudgetResponse",
    "CostComparisonResponse",
    "CostTrendResponse",
    "UsageRecordingResponse",
    "QuotaCheckResponse",
    "BillingMetricsResponse",
    # Billing mixins
    "BudgetPeriodMixin",
    "UsageStatsMixin",
    # Phase 5: Observability & Analytics Models
    "ObservabilityTraceResponse",
    "ObservabilityObservationResponse",
    "ObservabilityQualityScoreResponse",
    "ObservabilityStatsResponse",
    "ObservabilityListResponse",
    "ObservabilityBatchResponse",
    "AnalyticsResponse",
    "EvaluationResponse",
    # Supporting models
    "AnalyticsMetric",
    "EvaluationScore",
    # Observability mixins
    "ObservabilityTimingMixin",
    "QualityScoreMixin",
    "ObservabilityStatsCore",
    # Phase 6: Remaining Models
    "ErrorResponse",
    "APIResponse",
    "CacheResponse",
    "CacheStatsResponse",
    "EmbeddingGenerationResponse",
    "SemanticSearchResponse",
    "MLRoutingResponse",
    "MLModelInfoResponse",
    "ConfigResponse",
    "FeatureFlagResponse",
    "SubscriptionLimitResponse",
    "NotificationResponse",
    "NotificationStatusResponse",
    "NotificationHistoryResponse",
    # Remaining mixins
    "CacheMetricsMixin",
    "NotificationDeliveryMixin",
    "SearchResultMixin",
    # Industry standard response models
    "BaseResponse",
    "BrokleMetadata",
]
