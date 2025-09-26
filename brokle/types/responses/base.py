"""
Base classes and mixins for Brokle response models.

This module provides common field patterns extracted from the original
monolithic responses.py file to reduce duplication and improve maintainability.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


class TimestampMixin(BaseModel):
    """Mixin for common timestamp fields."""

    created_at: datetime = Field(description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Last update timestamp")


class MetadataMixin(BaseModel):
    """Mixin for metadata and tags fields."""

    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    tags: Optional[Dict[str, Any]] = Field(default=None, description="User-defined tags")


class TokenUsageMixin(BaseModel):
    """Mixin for token usage tracking fields."""

    prompt_tokens: Optional[int] = Field(default=None, description="Input/prompt tokens used")
    completion_tokens: Optional[int] = Field(default=None, description="Output/completion tokens used")
    total_tokens: Optional[int] = Field(default=None, description="Total tokens used")


class CostTrackingMixin(BaseModel):
    """Mixin for cost tracking fields."""

    input_cost: Optional[float] = Field(default=None, description="Cost for input tokens in USD")
    output_cost: Optional[float] = Field(default=None, description="Cost for output tokens in USD")
    total_cost_usd: Optional[float] = Field(default=None, description="Total cost in USD")


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

    request_id: Optional[str] = Field(default=None, description="Unique request identifier")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    session_id: Optional[str] = Field(default=None, description="Session identifier")


class OrganizationContextMixin(BaseModel):
    """Mixin for organization and project context fields."""

    organization_id: Optional[str] = Field(default=None, description="Organization identifier")
    project_id: Optional[str] = Field(default=None, description="Project identifier")
    environment: Optional[str] = Field(default=None, description="Environment tag")


class StatusMixin(BaseModel):
    """Mixin for status and state tracking fields."""

    status: str = Field(description="Current status")
    status_message: Optional[str] = Field(default=None, description="Status description")


# Base response classes using mixins
class BrokleResponseBase(BaseModel):
    """
    Base response class with common Brokle platform fields.

    Use this as a base for new response models that need basic platform integration.
    For backward compatibility, use the existing BaseResponse from responses.py.
    """

    model_config = ConfigDict(
        # Use alias generator and validation by name
        validate_by_name=True,
        populate_by_name=True,
    )


class TimestampedResponse(BrokleResponseBase, TimestampMixin):
    """Base response with timestamp tracking."""
    pass


class ProviderResponse(BrokleResponseBase, ProviderMixin, TokenUsageMixin, CostTrackingMixin):
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
    MetadataMixin
):
    """Base response with full context tracking."""
    pass


# Legacy response models for backward compatibility
class BrokleMetadata(BaseModel):
    """Brokle platform metadata automatically added to all responses."""

    # Request tracking
    request_id: Optional[str] = Field(default=None, description="Unique request ID")

    # Provider and routing info
    provider: Optional[str] = Field(default=None, description="AI provider used (openai, anthropic, etc.)")
    model_used: Optional[str] = Field(default=None, description="Actual model used by provider")
    routing_strategy: Optional[str] = Field(default=None, description="Routing strategy applied")
    routing_reason: Optional[str] = Field(default=None, description="Why this provider was chosen")

    # Performance metrics
    latency_ms: Optional[float] = Field(default=None, description="Total response time in milliseconds")

    # Cost tracking (automatic)
    cost_usd: Optional[float] = Field(default=None, description="Total cost for this request")
    cost_per_token: Optional[float] = Field(default=None, description="Cost per token")

    # Caching info (automatic)
    cache_hit: Optional[bool] = Field(default=None, description="Whether response came from cache")
    cache_similarity_score: Optional[float] = Field(default=None, description="Semantic similarity score if cached")

    # Quality assessment (automatic)
    quality_score: Optional[float] = Field(default=None, description="AI response quality score (0.0-1.0)")

    # Platform insights
    optimization_applied: Optional[List[str]] = Field(default=None, description="Optimizations applied automatically")
    cost_savings_usd: Optional[float] = Field(default=None, description="Cost saved through optimization")


class BaseResponse(BaseModel):
    """Base response model with automatic Brokle platform insights."""

    # Brokle platform metadata (automatically populated)
    brokle: Optional[BrokleMetadata] = Field(default=None, description="Brokle platform insights")

    # Legacy fields for backward compatibility (will be moved to brokle.*)
    request_id: Optional[str] = Field(default=None, description="Request ID")
    provider: Optional[str] = Field(default=None, description="Provider used")
    provider_model_id: Optional[str] = Field(default=None, description="Provider model ID")
    latency_ms: Optional[float] = Field(default=None, description="Total latency in milliseconds")
    cost_usd: Optional[float] = Field(default=None, description="Total cost in USD")
    input_tokens: Optional[int] = Field(default=None, description="Input tokens used")
    output_tokens: Optional[int] = Field(default=None, description="Output tokens generated")
    total_tokens: Optional[int] = Field(default=None, description="Total tokens used")


# Export all models
__all__ = [
    # Mixins
    'TimestampMixin',
    'MetadataMixin',
    'TokenUsageMixin',
    'CostTrackingMixin',
    'PaginationMixin',
    'ProviderMixin',
    'RequestTrackingMixin',
    'OrganizationContextMixin',
    'StatusMixin',

    # Base classes
    'BrokleResponseBase',
    'TimestampedResponse',
    'ProviderResponse',
    'PaginatedResponse',
    'TrackedResponse',
    'FullContextResponse',

    # Legacy models (for backward compatibility)
    'BrokleMetadata',
    'BaseResponse',
]