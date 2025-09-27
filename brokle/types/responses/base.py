"""
Base classes and mixins for Brokle response models.

This module provides common field patterns extracted from the original
monolithic responses.py file to reduce duplication and improve maintainability.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TimestampMixin(BaseModel):
    """Mixin for common timestamp fields."""

    created_at: datetime = Field(description="Creation timestamp")
    updated_at: Optional[datetime] = Field(
        default=None, description="Last update timestamp"
    )


class MetadataMixin(BaseModel):
    """Mixin for metadata and tags fields."""

    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata"
    )
    tags: Optional[Dict[str, Any]] = Field(
        default=None, description="User-defined tags"
    )


class TokenUsageMixin(BaseModel):
    """Mixin for token usage tracking fields."""

    prompt_tokens: Optional[int] = Field(
        default=None, description="Input/prompt tokens used"
    )
    completion_tokens: Optional[int] = Field(
        default=None, description="Output/completion tokens used"
    )
    total_tokens: Optional[int] = Field(default=None, description="Total tokens used")


class CostTrackingMixin(BaseModel):
    """Mixin for cost tracking fields."""

    input_cost: Optional[float] = Field(
        default=None, description="Cost for input tokens in USD"
    )
    output_cost: Optional[float] = Field(
        default=None, description="Cost for output tokens in USD"
    )
    total_cost_usd: Optional[float] = Field(
        default=None, description="Total cost in USD"
    )


class PaginationMixin(BaseModel):
    """Mixin for paginated response fields."""

    total_count: int = Field(description="Total number of items")
    page: int = Field(description="Current page number (0-indexed)")
    page_size: int = Field(description="Number of items per page")


class ProviderMixin(BaseModel):
    """Mixin for AI provider identification fields."""

    provider: Optional[str] = Field(default=None, description="AI provider name")
    model: Optional[str] = Field(default=None, description="AI model identifier")


class RequestTrackingMixin(BaseModel):
    """Mixin for request and session tracking fields."""

    request_id: Optional[str] = Field(
        default=None, description="Unique request identifier"
    )
    user_id: Optional[str] = Field(default=None, description="User identifier")
    session_id: Optional[str] = Field(default=None, description="Session identifier")


class OrganizationContextMixin(BaseModel):
    """Mixin for organization context fields."""

    organization_id: Optional[str] = Field(
        default=None, description="Organization identifier"
    )
    environment: Optional[str] = Field(default=None, description="Environment tag")


class StatusMixin(BaseModel):
    """Mixin for status and state tracking fields."""

    status: str = Field(description="Current status")
    status_message: Optional[str] = Field(
        default=None, description="Status description"
    )


# Base response classes using mixins
class BrokleResponseBase(BaseModel):
    """
    Base response class with common Brokle platform fields.

    Use this as a base for new response models that need basic platform integration.
    Follow industry standard patterns with clean namespace separation.
    """

    model_config = ConfigDict(
        # Use alias generator and validation by name
        validate_by_name=True,
        populate_by_name=True,
    )


class TimestampedResponse(BrokleResponseBase, TimestampMixin):
    """Base response with timestamp tracking."""

    pass


class ProviderResponse(
    BrokleResponseBase, ProviderMixin, TokenUsageMixin, CostTrackingMixin
):
    """Base response for AI provider interactions with usage/cost tracking."""

    pass


class PaginatedResponse(BrokleResponseBase, PaginationMixin):
    """Base response for paginated data."""

    pass


class TrackedResponse(BrokleResponseBase, RequestTrackingMixin, TimestampMixin):
    """Base response with request tracking and timestamps."""

    pass


class FullContextResponse(
    BrokleResponseBase,
    RequestTrackingMixin,
    OrganizationContextMixin,
    TimestampMixin,
    MetadataMixin,
):
    """Base response with full context tracking."""

    pass


# Industry standard platform metadata
class BrokleMetadata(BaseModel):
    """
    Complete Brokle platform metadata following industry standards.

    Contains all platform insights, analytics, and tracking data.
    Follows AWS SDK ResponseMetadata and Google Cloud metadata patterns.
    """

    # Request tracking
    request_id: Optional[str] = Field(default=None, description="Unique request ID")

    # Provider and routing info
    provider: Optional[str] = Field(
        default=None, description="AI provider used (openai, anthropic, etc.)"
    )
    model_used: Optional[str] = Field(
        default=None, description="Actual model used by provider"
    )
    routing_strategy: Optional[str] = Field(
        default=None, description="Routing strategy applied"
    )
    routing_reason: Optional[str] = Field(
        default=None, description="Why this provider was chosen"
    )
    routing_decision: Optional[Dict[str, Any]] = Field(
        default=None, description="Detailed routing decision data"
    )

    # Performance metrics
    latency_ms: Optional[float] = Field(
        default=None, description="Total response time in milliseconds"
    )

    # Complete cost tracking
    cost_usd: Optional[float] = Field(
        default=None, description="Total cost for this request"
    )
    cost_per_token: Optional[float] = Field(default=None, description="Cost per token")
    input_cost_usd: Optional[float] = Field(
        default=None, description="Input cost in USD"
    )
    output_cost_usd: Optional[float] = Field(
        default=None, description="Output cost in USD"
    )

    # Token usage tracking
    input_tokens: Optional[int] = Field(
        default=None, description="Input/prompt tokens used"
    )
    output_tokens: Optional[int] = Field(
        default=None, description="Output/completion tokens generated"
    )
    total_tokens: Optional[int] = Field(default=None, description="Total tokens used")

    # Caching info
    cache_hit: Optional[bool] = Field(
        default=None, description="Whether response came from cache"
    )
    cache_similarity_score: Optional[float] = Field(
        default=None, description="Semantic similarity score if cached"
    )
    cached: Optional[bool] = Field(
        default=None, description="Whether response was cached (alternative field)"
    )

    # Quality assessment and evaluation
    quality_score: Optional[float] = Field(
        default=None, description="AI response quality score (0.0-1.0)"
    )
    evaluation_scores: Optional[Dict[str, float]] = Field(
        default=None, description="Detailed evaluation scores"
    )

    # Platform insights and optimization
    optimization_applied: Optional[List[str]] = Field(
        default=None, description="Optimizations applied automatically"
    )
    cost_savings_usd: Optional[float] = Field(
        default=None, description="Cost saved through optimization"
    )

    # Metadata and timestamps
    created_at: Optional[datetime] = Field(
        default=None, description="Response creation timestamp"
    )
    custom_tags: Optional[Dict[str, Any]] = Field(
        default=None, description="User-defined custom tags"
    )


class BaseResponse(BaseModel):
    """
    Base response model following industry standard pattern.

    Uses clean namespace separation like AWS SDK (ResponseMetadata),
    Google Cloud (metadata), and Stripe (last_response).

    All Brokle platform insights are accessed via response.brokle.*
    """

    # Brokle platform metadata (industry standard namespace pattern)
    brokle: Optional[BrokleMetadata] = Field(
        default=None, description="Brokle platform insights and metadata"
    )

    # Backward compatibility properties (deprecated)
    # These forward to response.brokle.* for legacy code support

    @property
    def request_id(self) -> Optional[str]:
        """DEPRECATED: Use response.brokle.request_id instead."""
        import warnings

        warnings.warn(
            "Accessing response.request_id is deprecated. Use response.brokle.request_id instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.brokle.request_id if self.brokle else None

    @property
    def provider(self) -> Optional[str]:
        """DEPRECATED: Use response.brokle.provider instead."""
        import warnings

        warnings.warn(
            "Accessing response.provider is deprecated. Use response.brokle.provider instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.brokle.provider if self.brokle else None

    @property
    def cost_usd(self) -> Optional[float]:
        """DEPRECATED: Use response.brokle.cost_usd instead."""
        import warnings

        warnings.warn(
            "Accessing response.cost_usd is deprecated. Use response.brokle.cost_usd instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.brokle.cost_usd if self.brokle else None

    @property
    def cache_hit(self) -> Optional[bool]:
        """DEPRECATED: Use response.brokle.cache_hit instead."""
        import warnings

        warnings.warn(
            "Accessing response.cache_hit is deprecated. Use response.brokle.cache_hit instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.brokle.cache_hit if self.brokle else None

    @property
    def quality_score(self) -> Optional[float]:
        """DEPRECATED: Use response.brokle.quality_score instead."""
        import warnings

        warnings.warn(
            "Accessing response.quality_score is deprecated. Use response.brokle.quality_score instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.brokle.quality_score if self.brokle else None

    @property
    def input_tokens(self) -> Optional[int]:
        """DEPRECATED: Use response.brokle.input_tokens instead."""
        import warnings

        warnings.warn(
            "Accessing response.input_tokens is deprecated. Use response.brokle.input_tokens instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.brokle.input_tokens if self.brokle else None

    @property
    def output_tokens(self) -> Optional[int]:
        """DEPRECATED: Use response.brokle.output_tokens instead."""
        import warnings

        warnings.warn(
            "Accessing response.output_tokens is deprecated. Use response.brokle.output_tokens instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.brokle.output_tokens if self.brokle else None

    @property
    def total_tokens(self) -> Optional[int]:
        """DEPRECATED: Use response.brokle.total_tokens instead."""
        import warnings

        warnings.warn(
            "Accessing response.total_tokens is deprecated. Use response.brokle.total_tokens instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.brokle.total_tokens if self.brokle else None

    @property
    def latency_ms(self) -> Optional[float]:
        """DEPRECATED: Use response.brokle.latency_ms instead."""
        import warnings

        warnings.warn(
            "Accessing response.latency_ms is deprecated. Use response.brokle.latency_ms instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.brokle.latency_ms if self.brokle else None

    @property
    def routing_reason(self) -> Optional[str]:
        """DEPRECATED: Use response.brokle.routing_reason instead."""
        import warnings

        warnings.warn(
            "Accessing response.routing_reason is deprecated. Use response.brokle.routing_reason instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.brokle.routing_reason if self.brokle else None


# Export all models
__all__ = [
    # Mixins
    "TimestampMixin",
    "MetadataMixin",
    "TokenUsageMixin",
    "CostTrackingMixin",
    "PaginationMixin",
    "ProviderMixin",
    "RequestTrackingMixin",
    "OrganizationContextMixin",
    "StatusMixin",
    # Base classes
    "BrokleResponseBase",
    "TimestampedResponse",
    "ProviderResponse",
    "PaginatedResponse",
    "TrackedResponse",
    "FullContextResponse",
    # Industry standard response models
    "BrokleMetadata",
    "BaseResponse",
]
