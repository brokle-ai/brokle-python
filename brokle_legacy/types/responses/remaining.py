"""
Remaining response models for Brokle SDK.

This module contains the final set of response models:
- Error Handling: ErrorResponse, APIResponse
- Caching: CacheResponse, CacheStatsResponse
- Embeddings & Search: EmbeddingGenerationResponse, SemanticSearchResponse
- ML & Routing: MLRoutingResponse, MLModelInfoResponse
- Configuration: ConfigResponse, FeatureFlagResponse
- Usage & Billing: SubscriptionLimitResponse
- Notifications: NotificationResponse, NotificationStatusResponse, NotificationHistoryResponse

Models are refactored to use mixins for reduced duplication while maintaining
complete backward compatibility.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Import our mixins and base classes
from .base import (
    BrokleResponseBase,
    MetadataMixin,
    PaginationMixin,
    StatusMixin,
    TimestampMixin,
)


class CacheMetricsMixin(BaseModel):
    """Mixin for cache performance metrics."""

    hit_rate: float = Field(description="Cache hit rate (0.0-1.0)")
    miss_rate: float = Field(description="Cache miss rate (0.0-1.0)")
    total_hits: int = Field(description="Total cache hits")
    total_misses: int = Field(description="Total cache misses")


class NotificationDeliveryMixin(BaseModel):
    """Mixin for notification delivery tracking."""

    sent_at: datetime = Field(description="Sent timestamp")
    delivered_at: Optional[datetime] = Field(
        default=None, description="Delivery timestamp"
    )
    read_at: Optional[datetime] = Field(default=None, description="Read timestamp")
    error_message: Optional[str] = Field(
        default=None, description="Error message if failed"
    )
    retry_count: int = Field(description="Retry attempts")


class SearchResultMixin(BaseModel):
    """Mixin for search result fields."""

    score: float = Field(description="Search relevance score")
    rank: int = Field(description="Search result rank")


# Error Handling Models
class ErrorResponse(BaseModel):
    """
    Error response model.

    Maintains complete backward compatibility with original.
    """

    error: Dict[str, Any] = Field(description="Error details")
    success: bool = Field(default=False, description="Success status")
    request_id: Optional[str] = Field(default=None, description="Request ID")
    timestamp: Optional[datetime] = Field(default=None, description="Error timestamp")


class APIResponse(BaseModel):
    """
    Generic API response wrapper.

    Maintains complete backward compatibility with original.
    """

    success: bool = Field(description="Success status")
    data: Optional[Any] = Field(default=None, description="Response data")
    error: Optional[Dict[str, Any]] = Field(default=None, description="Error details")
    meta: Optional[Dict[str, Any]] = Field(
        default=None, description="Response metadata"
    )


# Caching Models
class CacheResponse(BaseModel):
    """
    Cache response.

    Maintains complete backward compatibility with original.
    """

    hit: bool = Field(description="Cache hit status")
    key: Optional[str] = Field(default=None, description="Cache key")
    value: Optional[Dict[str, Any]] = Field(default=None, description="Cache value")
    ttl: Optional[int] = Field(default=None, description="Time to live remaining")
    similarity_score: Optional[float] = Field(
        default=None, description="Similarity score for semantic cache"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Cache metadata"
    )
    created_at: Optional[datetime] = Field(
        default=None, description="Cache creation timestamp"
    )
    accessed_at: Optional[datetime] = Field(
        default=None, description="Last access timestamp"
    )


class CacheStatsResponse(BaseModel):
    """
    Cache statistics response.

    Maintains complete backward compatibility with original.
    """

    total_entries: int = Field(description="Total cache entries")
    hit_rate: float = Field(description="Cache hit rate (0.0-1.0)")
    miss_rate: float = Field(description="Cache miss rate (0.0-1.0)")
    total_hits: int = Field(description="Total cache hits")
    total_misses: int = Field(description="Total cache misses")
    average_access_time_ms: Optional[float] = Field(
        default=None, description="Average access time"
    )
    memory_usage_mb: Optional[float] = Field(
        default=None, description="Memory usage in MB"
    )
    cache_size_mb: Optional[float] = Field(default=None, description="Cache size in MB")
    eviction_count: int = Field(description="Number of evictions")


# Embeddings & Search Models
class EmbeddingGenerationResponse(BaseModel):
    """
    Embedding generation response.

    Maintains complete backward compatibility with original.
    """

    embedding_id: str = Field(description="Embedding ID")
    text: str = Field(description="Input text")
    embedding: List[float] = Field(description="Generated embedding vector")
    model: str = Field(description="Model used")
    dimensions: int = Field(description="Embedding dimensions")
    token_count: Optional[int] = Field(default=None, description="Input token count")
    processing_time_ms: Optional[float] = Field(
        default=None, description="Processing time"
    )
    cost_usd: Optional[float] = Field(default=None, description="Cost in USD")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Generation metadata"
    )
    created_at: datetime = Field(description="Creation timestamp")


class SemanticSearchResponse(BaseModel):
    """
    Semantic search response.

    Maintains complete backward compatibility with original.
    """

    query: str = Field(description="Search query")
    results: List[Dict[str, Any]] = Field(description="Search results")
    total_results: int = Field(description="Total number of results")
    search_time_ms: float = Field(description="Search time in milliseconds")
    filters_applied: Optional[Dict[str, Any]] = Field(
        default=None, description="Applied filters"
    )
    similarity_threshold: Optional[float] = Field(
        default=None, description="Similarity threshold used"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Search metadata"
    )


# ML & Routing Models
class MLRoutingResponse(BaseModel):
    """
    ML routing response.

    Maintains complete backward compatibility with original.
    """

    request_id: str = Field(description="Request ID")
    selected_provider: str = Field(description="Selected provider")
    selected_model: str = Field(description="Selected model")
    routing_strategy: str = Field(description="Routing strategy used")
    confidence_score: float = Field(description="Routing confidence score")
    alternative_providers: Optional[List[str]] = Field(
        default=None, description="Alternative providers considered"
    )
    routing_metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Routing metadata"
    )
    decision_time_ms: float = Field(description="Decision time in milliseconds")
    cost_estimate_usd: Optional[float] = Field(
        default=None, description="Estimated cost"
    )
    quality_prediction: Optional[float] = Field(
        default=None, description="Predicted quality score"
    )


class MLModelInfoResponse(BaseModel):
    """
    ML model information response.

    Maintains complete backward compatibility with original.
    """

    model_name: str = Field(description="Model name")
    provider: str = Field(description="Model provider")
    model_type: str = Field(description="Model type")
    capabilities: List[str] = Field(description="Model capabilities")
    input_cost_per_token: Optional[float] = Field(
        default=None, description="Input cost per token"
    )
    output_cost_per_token: Optional[float] = Field(
        default=None, description="Output cost per token"
    )
    context_length: Optional[int] = Field(default=None, description="Context length")
    max_output_tokens: Optional[int] = Field(
        default=None, description="Max output tokens"
    )
    supports_streaming: bool = Field(description="Streaming support")
    supports_function_calling: bool = Field(description="Function calling support")
    latency_p50_ms: Optional[float] = Field(
        default=None, description="50th percentile latency"
    )
    latency_p95_ms: Optional[float] = Field(
        default=None, description="95th percentile latency"
    )
    availability: float = Field(description="Model availability (0.0-1.0)")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Model metadata"
    )


# Configuration Models
class ConfigResponse(BaseModel):
    """
    Configuration response.

    Maintains complete backward compatibility with original.
    """

    config_key: str = Field(description="Configuration key")
    config_value: Any = Field(description="Configuration value")
    data_type: str = Field(description="Value data type")
    description: Optional[str] = Field(
        default=None, description="Configuration description"
    )
    is_sensitive: bool = Field(description="Whether config is sensitive")
    scope: str = Field(description="Configuration scope")
    last_updated: Optional[datetime] = Field(
        default=None, description="Last update timestamp"
    )
    updated_by: Optional[str] = Field(default=None, description="Updated by user")


class FeatureFlagResponse(BaseModel):
    """
    Feature flag response.

    Maintains complete backward compatibility with original.
    """

    flag_key: str = Field(description="Feature flag key")
    enabled: bool = Field(description="Flag enabled status")
    description: Optional[str] = Field(default=None, description="Flag description")
    rollout_percentage: Optional[float] = Field(
        default=None, description="Rollout percentage"
    )
    target_groups: Optional[List[str]] = Field(
        default=None, description="Target user groups"
    )
    conditions: Optional[Dict[str, Any]] = Field(
        default=None, description="Flag conditions"
    )
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: Optional[datetime] = Field(
        default=None, description="Last update timestamp"
    )


# Usage & Billing Models
class SubscriptionLimitResponse(BaseModel):
    """
    Subscription limit response.

    Maintains complete backward compatibility with original.
    """

    organization_id: str = Field(description="Organization ID")
    subscription_tier: str = Field(description="Subscription tier")
    current_usage: int = Field(description="Current usage")
    usage_limit: int = Field(description="Usage limit")
    usage_percentage: float = Field(description="Usage percentage")
    limit_type: str = Field(description="Type of limit")
    reset_date: Optional[datetime] = Field(default=None, description="Limit reset date")
    overage_allowed: bool = Field(description="Overage allowed")
    overage_cost_per_unit: Optional[float] = Field(
        default=None, description="Overage cost per unit"
    )


# Notification Models
class NotificationResponse(BaseModel):
    """
    Notification response.

    Maintains complete backward compatibility with original.
    """

    notification_id: str = Field(description="Notification ID")
    status: str = Field(description="Notification status")
    recipient: str = Field(description="Recipient")
    channel: str = Field(description="Notification channel")
    sent_at: datetime = Field(description="Sent timestamp")
    delivered_at: Optional[datetime] = Field(
        default=None, description="Delivery timestamp"
    )
    read_at: Optional[datetime] = Field(default=None, description="Read timestamp")
    error_message: Optional[str] = Field(
        default=None, description="Error message if failed"
    )
    retry_count: int = Field(description="Retry attempts")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Notification metadata"
    )


class NotificationStatusResponse(BaseModel):
    """
    Notification status response.

    Maintains complete backward compatibility with original.
    """

    notification_id: str = Field(description="Notification ID")
    status: str = Field(description="Current status")
    total_recipients: int = Field(description="Total recipients")
    delivered_count: int = Field(description="Successfully delivered")
    failed_count: int = Field(description="Failed deliveries")
    pending_count: int = Field(description="Pending deliveries")
    delivery_rate: float = Field(description="Delivery success rate")
    last_updated: datetime = Field(description="Last status update")


class NotificationHistoryResponse(BaseModel):
    """
    Notification history response.

    Maintains complete backward compatibility with original.
    """

    organization_id: str = Field(description="Organization ID")
    total_notifications: int = Field(description="Total notifications sent")
    date_range: Dict[str, str] = Field(description="Date range for history")
    notifications: List[Dict[str, Any]] = Field(
        description="Notification history entries"
    )
    summary: Dict[str, Any] = Field(description="History summary statistics")
    page: int = Field(description="Current page")
    per_page: int = Field(description="Items per page")
    total_pages: int = Field(description="Total pages")


# Re-export for backward compatibility
__all__ = [
    # Backward compatible models
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
    # Remaining-specific mixins
    "CacheMetricsMixin",
    "NotificationDeliveryMixin",
    "SearchResultMixin",
]
