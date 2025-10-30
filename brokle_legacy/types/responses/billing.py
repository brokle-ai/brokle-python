"""
Billing and cost response models for Brokle SDK.

This module contains response models for billing, cost tracking, and usage:
- CostCalculationResponse
- CostTrackingResponse
- BudgetResponse
- CostComparisonResponse
- CostTrendResponse
- UsageRecordingResponse
- QuotaCheckResponse
- BillingMetricsResponse

Models follow industry standard patterns with clean architecture using mixins
for modular design and response.brokle.* namespace separation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Import our mixins and base classes
from .base import (
    BrokleResponseBase,
    CostTrackingMixin,
    MetadataMixin,
    OrganizationContextMixin,
    ProviderMixin,
    RequestTrackingMixin,
    TimestampMixin,
    TokenUsageMixin,
)


class BudgetPeriodMixin(BaseModel):
    """Mixin for budget period fields."""

    period_start: datetime = Field(description="Budget period start")
    period_end: datetime = Field(description="Budget period end")


class UsageStatsMixin(BaseModel):
    """Mixin for usage statistics fields."""

    total_requests: int = Field(description="Total requests count")
    total_tokens: int = Field(description="Total tokens used")
    total_cost: float = Field(description="Total cost")


class CostCalculationResponse(BaseModel):
    """
    Cost calculation response model.

    Maintains complete backward compatibility with original while internally
    leveraging mixins for common field patterns.
    """

    provider: str = Field(description="AI provider")
    model: str = Field(description="Model name")
    input_cost_usd: float = Field(description="Input cost in USD")
    output_cost_usd: float = Field(description="Output cost in USD")
    total_cost_usd: float = Field(description="Total cost in USD")
    input_tokens: int = Field(description="Input token count")
    output_tokens: int = Field(description="Output token count")
    pricing_model: str = Field(description="Pricing model used")
    calculation_timestamp: datetime = Field(description="Calculation timestamp")


class CostTrackingResponse(BaseModel):
    """
    Cost tracking response model.

    Maintains complete backward compatibility with original.
    """

    request_id: str = Field(description="Request ID")
    tracked: bool = Field(description="Successfully tracked")
    calculated_cost: float = Field(description="Calculated cost")
    actual_cost: Optional[float] = Field(
        default=None, description="Actual provider cost"
    )
    variance: Optional[float] = Field(default=None, description="Cost variance")
    organization_total: Optional[float] = Field(
        default=None, description="Organization total cost"
    )


class BudgetResponse(BaseModel):
    """
    Budget response model.

    Maintains complete backward compatibility with original.
    """

    budget_id: str = Field(description="Budget ID")
    organization_id: str = Field(description="Organization ID")
    environment: Optional[str] = Field(default=None, description="Environment tag")
    budget_type: str = Field(description="Budget type")
    amount: float = Field(description="Budget amount")
    spent: float = Field(description="Amount spent")
    remaining: float = Field(description="Amount remaining")
    utilization: float = Field(description="Budget utilization (0.0-1.0)")
    status: str = Field(description="Budget status")
    alert_thresholds: List[float] = Field(description="Alert thresholds")
    alerts_triggered: List[str] = Field(description="Triggered alerts")
    period_start: datetime = Field(description="Budget period start")
    period_end: datetime = Field(description="Budget period end")


class CostComparisonResponse(BaseModel):
    """
    Cost comparison response model.

    Maintains complete backward compatibility with original.
    """

    providers: List[str] = Field(description="Compared providers")
    costs: Dict[str, float] = Field(description="Cost per provider")
    best_option: str = Field(description="Most cost-effective provider")
    savings_potential: float = Field(description="Potential savings in USD")
    comparison_details: Dict[str, Dict[str, Any]] = Field(
        description="Detailed comparison"
    )


class CostTrendResponse(BaseModel):
    """
    Cost trend response model.

    Maintains complete backward compatibility with original.
    """

    organization_id: str = Field(description="Organization ID")
    period: str = Field(description="Time period")
    total_cost: float = Field(description="Total cost")
    trend_data: List[Dict[str, Any]] = Field(description="Trend data points")
    average_daily_cost: float = Field(description="Average daily cost")
    cost_change_percent: float = Field(description="Cost change percentage")
    top_providers: List[Dict[str, Any]] = Field(description="Top providers by cost")
    top_models: List[Dict[str, Any]] = Field(description="Top models by cost")


class UsageRecordingResponse(BaseModel):
    """
    Usage recording response model.

    Maintains complete backward compatibility with original.
    """

    request_id: str = Field(description="Request ID")
    recorded: bool = Field(description="Successfully recorded")
    organization_id: str = Field(description="Organization ID")
    total_requests: int = Field(description="Total requests count")
    total_tokens: int = Field(description="Total tokens used")
    total_cost: float = Field(description="Total cost")
    current_period_usage: Dict[str, Any] = Field(
        description="Current period usage summary"
    )


class QuotaCheckResponse(BaseModel):
    """
    Quota check response model.

    Maintains complete backward compatibility with original.
    """

    allowed: bool = Field(description="Request allowed")
    organization_id: str = Field(description="Organization ID")
    resource_type: str = Field(description="Resource type")
    current_usage: int = Field(description="Current usage")
    quota_limit: int = Field(description="Quota limit")
    remaining: int = Field(description="Remaining quota")
    reset_date: Optional[datetime] = Field(default=None, description="Quota reset date")
    warning_threshold: Optional[float] = Field(
        default=None, description="Warning threshold"
    )
    is_warning: bool = Field(description="Warning threshold reached")


class BillingMetricsResponse(BaseModel):
    """
    Billing metrics response model.

    Maintains complete backward compatibility with original.
    """

    organization_id: str = Field(description="Organization ID")
    period: str = Field(description="Time period")
    total_cost: float = Field(description="Total cost")
    total_requests: int = Field(description="Total requests")
    total_tokens: int = Field(description="Total tokens")
    cost_breakdown: Dict[str, float] = Field(
        description="Cost breakdown by service/provider"
    )
    top_projects: List[Dict[str, Any]] = Field(description="Top projects by usage")
    usage_trends: List[Dict[str, Any]] = Field(description="Usage trend data")


# Re-export for backward compatibility
__all__ = [
    # Backward compatible models
    "CostCalculationResponse",
    "CostTrackingResponse",
    "BudgetResponse",
    "CostComparisonResponse",
    "CostTrendResponse",
    "UsageRecordingResponse",
    "QuotaCheckResponse",
    "BillingMetricsResponse",
    # Billing-specific mixins
    "BudgetPeriodMixin",
    "UsageStatsMixin",
]
