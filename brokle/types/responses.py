"""
Response models for Brokle SDK.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime


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
    cache_hit: Optional[bool] = Field(default=None, description="Whether cache was hit")
    quality_score: Optional[float] = Field(default=None, description="Overall quality score")
    
    # Additional legacy fields (deprecated - use brokle.* instead)
    provider_latency_ms: Optional[float] = Field(default=None, description="Provider latency in milliseconds")
    gateway_latency_ms: Optional[float] = Field(default=None, description="Platform latency in milliseconds")
    input_cost_usd: Optional[float] = Field(default=None, description="Input cost in USD")
    output_cost_usd: Optional[float] = Field(default=None, description="Output cost in USD")
    routing_strategy: Optional[str] = Field(default=None, description="Routing strategy used")
    routing_reason: Optional[str] = Field(default=None, description="Routing decision reason")
    routing_decision: Optional[Dict[str, Any]] = Field(default=None, description="Detailed routing decision")
    cached: Optional[bool] = Field(default=None, description="Whether response was cached")
    cache_strategy: Optional[str] = Field(default=None, description="Cache strategy used")
    cache_similarity_score: Optional[float] = Field(default=None, description="Cache similarity score")
    evaluation_scores: Optional[Dict[str, float]] = Field(default=None, description="Evaluation scores")
    ab_test_variant: Optional[str] = Field(default=None, description="A/B test variant")
    ab_test_experiment: Optional[str] = Field(default=None, description="A/B test experiment")
    created_at: Optional[datetime] = Field(default=None, description="Creation timestamp")
    custom_tags: Optional[Dict[str, Any]] = Field(default=None, description="Custom tags")


class CompletionChoice(BaseModel):
    """Completion choice model."""
    
    text: str = Field(description="Generated text")
    index: int = Field(description="Choice index")
    logprobs: Optional[Dict[str, Any]] = Field(default=None, description="Log probabilities")
    finish_reason: Optional[str] = Field(default=None, description="Finish reason")


class CompletionResponse(BaseResponse):
    """Completion response model."""
    
    id: str = Field(description="Response ID")
    object: str = Field(description="Object type")
    created: int = Field(description="Creation timestamp")
    model: str = Field(description="Model used")
    choices: List[CompletionChoice] = Field(description="Generated choices")
    usage: Optional[Dict[str, int]] = Field(default=None, description="Token usage")


class ChatCompletionMessage(BaseModel):
    """Chat completion message model."""
    
    role: str = Field(description="Message role")
    content: Optional[str] = Field(default=None, description="Message content")
    function_call: Optional[Dict[str, Any]] = Field(default=None, description="Function call")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(default=None, description="Tool calls")
    name: Optional[str] = Field(default=None, description="Function name")


class ChatCompletionChoice(BaseModel):
    """Chat completion choice model."""
    
    index: int = Field(description="Choice index")
    message: ChatCompletionMessage = Field(description="Message")
    finish_reason: Optional[str] = Field(default=None, description="Finish reason")
    logprobs: Optional[Dict[str, Any]] = Field(default=None, description="Log probabilities")


class ChatCompletionResponse(BaseResponse):
    """Chat completion response model."""
    
    id: str = Field(description="Response ID")
    object: str = Field(description="Object type")
    created: int = Field(description="Creation timestamp")
    model: str = Field(description="Model used")
    choices: List[ChatCompletionChoice] = Field(description="Generated choices")
    usage: Optional[Dict[str, int]] = Field(default=None, description="Token usage")
    system_fingerprint: Optional[str] = Field(default=None, description="System fingerprint")


class EmbeddingData(BaseModel):
    """Embedding data model."""
    
    object: str = Field(description="Object type")
    embedding: List[float] = Field(description="Embedding vector")
    index: int = Field(description="Embedding index")


class EmbeddingResponse(BaseResponse):
    """Embedding response model."""
    
    object: str = Field(description="Object type")
    data: List[EmbeddingData] = Field(description="Embedding data")
    model: str = Field(description="Model used")
    usage: Optional[Dict[str, int]] = Field(default=None, description="Token usage")


class AnalyticsMetric(BaseModel):
    """Analytics metric model."""
    
    name: str = Field(description="Metric name")
    value: Union[int, float, str] = Field(description="Metric value")
    timestamp: Optional[datetime] = Field(default=None, description="Metric timestamp")
    dimensions: Optional[Dict[str, str]] = Field(default=None, description="Metric dimensions")


class AnalyticsResponse(BaseModel):
    """Analytics response model."""
    
    metrics: List[AnalyticsMetric] = Field(description="Analytics metrics")
    total_count: Optional[int] = Field(default=None, description="Total count")
    start_date: Optional[str] = Field(default=None, description="Start date")
    end_date: Optional[str] = Field(default=None, description="End date")
    granularity: Optional[str] = Field(default=None, description="Time granularity")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Applied filters")


class EvaluationScore(BaseModel):
    """Evaluation score model."""
    
    metric: str = Field(description="Evaluation metric")
    score: float = Field(description="Score value")
    threshold: Optional[float] = Field(default=None, description="Score threshold")
    passed: Optional[bool] = Field(default=None, description="Whether score passed threshold")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional details")


class EvaluationResponse(BaseModel):
    """Evaluation response model."""
    
    response_id: Optional[str] = Field(default=None, description="Response ID")
    evaluation_id: str = Field(description="Evaluation ID")
    scores: List[EvaluationScore] = Field(description="Evaluation scores")
    overall_score: Optional[float] = Field(default=None, description="Overall score")
    feedback: Optional[str] = Field(default=None, description="Feedback text")
    created_at: datetime = Field(description="Creation timestamp")
    processing_time_ms: Optional[float] = Field(default=None, description="Processing time in milliseconds")


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: Dict[str, Any] = Field(description="Error details")
    success: bool = Field(default=False, description="Success status")
    request_id: Optional[str] = Field(default=None, description="Request ID")
    timestamp: Optional[datetime] = Field(default=None, description="Error timestamp")


class APIResponse(BaseModel):
    """Generic API response wrapper."""
    
    success: bool = Field(description="Success status")
    data: Optional[Any] = Field(default=None, description="Response data")
    error: Optional[Dict[str, Any]] = Field(default=None, description="Error details")
    meta: Optional[Dict[str, Any]] = Field(default=None, description="Response metadata")
    
    class Config:
        extra = "allow"


# Telemetry Service Responses
class TelemetryTraceResponse(BaseModel):
    """Telemetry trace response."""
    
    trace_id: str = Field(description="Trace ID")
    name: str = Field(description="Trace name")
    status: str = Field(description="Trace status")
    start_time: datetime = Field(description="Trace start time")
    end_time: Optional[datetime] = Field(default=None, description="Trace end time")
    duration_ms: Optional[float] = Field(default=None, description="Duration in milliseconds")
    span_count: Optional[int] = Field(default=None, description="Number of spans")
    user_id: Optional[str] = Field(default=None, description="User ID")
    session_id: Optional[str] = Field(default=None, description="Session ID")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Trace metadata")
    tags: Optional[List[str]] = Field(default=None, description="Trace tags")


class TelemetrySpanResponse(BaseModel):
    """Telemetry span response."""
    
    span_id: str = Field(description="Span ID")
    trace_id: str = Field(description="Parent trace ID")
    parent_span_id: Optional[str] = Field(default=None, description="Parent span ID")
    name: str = Field(description="Span name")
    span_type: str = Field(description="Span type")
    status: str = Field(description="Span status")
    start_time: datetime = Field(description="Span start time")
    end_time: Optional[datetime] = Field(default=None, description="Span end time")
    duration_ms: Optional[float] = Field(default=None, description="Duration in milliseconds")
    attributes: Optional[Dict[str, Any]] = Field(default=None, description="Span attributes")
    events: Optional[List[Dict[str, Any]]] = Field(default=None, description="Span events")


class TelemetryEventBatchResponse(BaseModel):
    """Batch telemetry events response."""
    
    processed_count: int = Field(description="Number of events processed")
    failed_count: int = Field(description="Number of events failed")
    batch_id: str = Field(description="Batch ID")
    processing_time_ms: float = Field(description="Processing time in milliseconds")
    errors: Optional[List[Dict[str, Any]]] = Field(default=None, description="Processing errors")


# Cache Service Responses
class CacheResponse(BaseModel):
    """Cache response."""
    
    hit: bool = Field(description="Cache hit status")
    key: Optional[str] = Field(default=None, description="Cache key")
    value: Optional[Dict[str, Any]] = Field(default=None, description="Cache value")
    ttl: Optional[int] = Field(default=None, description="Time to live remaining")
    similarity_score: Optional[float] = Field(default=None, description="Similarity score for semantic cache")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Cache metadata")
    created_at: Optional[datetime] = Field(default=None, description="Cache creation timestamp")
    accessed_at: Optional[datetime] = Field(default=None, description="Last access timestamp")


class CacheStatsResponse(BaseModel):
    """Cache statistics response."""
    
    total_entries: int = Field(description="Total cache entries")
    hit_rate: float = Field(description="Cache hit rate (0.0-1.0)")
    miss_rate: float = Field(description="Cache miss rate (0.0-1.0)")
    total_hits: int = Field(description="Total cache hits")
    total_misses: int = Field(description="Total cache misses")
    memory_usage_mb: Optional[float] = Field(default=None, description="Memory usage in MB")
    average_ttl: Optional[float] = Field(default=None, description="Average TTL in seconds")
    provider_stats: Optional[Dict[str, Dict[str, Any]]] = Field(default=None, description="Per-provider statistics")


class EmbeddingGenerationResponse(BaseModel):
    """Embedding generation response."""
    
    embeddings: List[List[float]] = Field(description="Generated embeddings")
    model: str = Field(description="Model used")
    provider: str = Field(description="Provider used")
    dimensions: int = Field(description="Embedding dimensions")
    tokens_used: Optional[int] = Field(default=None, description="Tokens used")
    processing_time_ms: float = Field(description="Processing time in milliseconds")
    cost_usd: Optional[float] = Field(default=None, description="Cost in USD")


class SemanticSearchResponse(BaseModel):
    """Semantic search response."""
    
    results: List[Dict[str, Any]] = Field(description="Search results")
    query: str = Field(description="Search query")
    total_results: int = Field(description="Total number of results")
    search_time_ms: float = Field(description="Search time in milliseconds")
    similarity_scores: List[float] = Field(description="Similarity scores for results")


# Cost Tracking Service Responses
class CostCalculationResponse(BaseModel):
    """Cost calculation response."""
    
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
    """Cost tracking response."""
    
    request_id: str = Field(description="Request ID")
    tracked: bool = Field(description="Successfully tracked")
    calculated_cost: float = Field(description="Calculated cost")
    actual_cost: Optional[float] = Field(default=None, description="Actual provider cost")
    variance: Optional[float] = Field(default=None, description="Cost variance")
    organization_total: Optional[float] = Field(default=None, description="Organization total cost")
    project_total: Optional[float] = Field(default=None, description="Project total cost")


class BudgetResponse(BaseModel):
    """Budget response."""
    
    budget_id: str = Field(description="Budget ID")
    organization_id: str = Field(description="Organization ID")
    project_id: Optional[str] = Field(default=None, description="Project ID")
    environment_id: Optional[str] = Field(default=None, description="Environment ID")
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
    """Cost comparison response."""
    
    providers: List[str] = Field(description="Compared providers")
    costs: Dict[str, float] = Field(description="Cost per provider")
    best_option: str = Field(description="Most cost-effective provider")
    savings_potential: float = Field(description="Potential savings in USD")
    comparison_details: Dict[str, Dict[str, Any]] = Field(description="Detailed comparison")


class CostTrendResponse(BaseModel):
    """Cost trend response."""
    
    organization_id: str = Field(description="Organization ID")
    period: str = Field(description="Time period")
    total_cost: float = Field(description="Total cost")
    trend_data: List[Dict[str, Any]] = Field(description="Trend data points")
    average_daily_cost: float = Field(description="Average daily cost")
    cost_change_percent: float = Field(description="Cost change percentage")
    top_providers: List[Dict[str, Any]] = Field(description="Top providers by cost")
    top_models: List[Dict[str, Any]] = Field(description="Top models by cost")


# ML Service Responses
class MLRoutingResponse(BaseModel):
    """ML routing recommendation response."""
    
    recommended_provider: str = Field(description="Recommended provider")
    confidence_score: float = Field(description="Confidence score (0.0-1.0)")
    routing_reason: str = Field(description="Routing decision reason")
    alternative_providers: List[Dict[str, Any]] = Field(description="Alternative provider options")
    expected_performance: Dict[str, float] = Field(description="Expected performance metrics")
    cost_estimate: float = Field(description="Estimated cost")
    latency_estimate: float = Field(description="Estimated latency in ms")
    model_mapping: Optional[Dict[str, str]] = Field(default=None, description="Provider-specific model mapping")


class MLModelInfoResponse(BaseModel):
    """ML model information response."""
    
    model_name: str = Field(description="Model name")
    model_version: str = Field(description="Model version")
    training_data_size: int = Field(description="Training data size")
    last_training: datetime = Field(description="Last training timestamp")
    accuracy_metrics: Dict[str, float] = Field(description="Model accuracy metrics")
    feature_importance: Dict[str, float] = Field(description="Feature importance scores")
    performance_stats: Dict[str, Any] = Field(description="Performance statistics")


# Configuration Service Responses
class ConfigResponse(BaseModel):
    """Configuration response."""
    
    organization_id: Optional[str] = Field(default=None, description="Organization ID")
    project_id: Optional[str] = Field(default=None, description="Project ID")
    environment: Optional[str] = Field(default=None, description="Environment")
    config: Dict[str, Any] = Field(description="Configuration data")
    version: str = Field(description="Configuration version")
    last_updated: datetime = Field(description="Last update timestamp")


class FeatureFlagResponse(BaseModel):
    """Feature flag response."""
    
    feature_name: str = Field(description="Feature name")
    enabled: bool = Field(description="Feature enabled status")
    tier: str = Field(description="Subscription tier")
    organization_id: str = Field(description="Organization ID")
    configuration: Optional[Dict[str, Any]] = Field(default=None, description="Feature configuration")
    rollout_percentage: Optional[float] = Field(default=None, description="Rollout percentage")


class SubscriptionLimitResponse(BaseModel):
    """Subscription limit response."""
    
    limit_type: str = Field(description="Limit type")
    tier: str = Field(description="Subscription tier")
    limit_value: Union[int, float] = Field(description="Limit value")
    current_usage: Union[int, float] = Field(description="Current usage")
    remaining: Union[int, float] = Field(description="Remaining quota")
    utilization: float = Field(description="Utilization percentage (0.0-1.0)")
    reset_date: Optional[datetime] = Field(default=None, description="Quota reset date")


# Billing Service Responses
class UsageRecordingResponse(BaseModel):
    """Usage recording response."""
    
    request_id: str = Field(description="Request ID")
    recorded: bool = Field(description="Successfully recorded")
    organization_id: str = Field(description="Organization ID")
    project_id: str = Field(description="Project ID")
    total_requests: int = Field(description="Total requests count")
    total_tokens: int = Field(description="Total tokens used")
    total_cost: float = Field(description="Total cost")
    current_period_usage: Dict[str, Any] = Field(description="Current period usage summary")


class QuotaCheckResponse(BaseModel):
    """Quota check response."""
    
    allowed: bool = Field(description="Request allowed")
    organization_id: str = Field(description="Organization ID")
    resource_type: str = Field(description="Resource type")
    current_usage: int = Field(description="Current usage")
    quota_limit: int = Field(description="Quota limit")
    remaining: int = Field(description="Remaining quota")
    reset_date: Optional[datetime] = Field(default=None, description="Quota reset date")
    warning_threshold: Optional[float] = Field(default=None, description="Warning threshold")
    is_warning: bool = Field(description="Warning threshold reached")


class BillingMetricsResponse(BaseModel):
    """Billing metrics response."""
    
    organization_id: str = Field(description="Organization ID")
    period: str = Field(description="Billing period")
    total_cost: float = Field(description="Total cost")
    total_requests: int = Field(description="Total requests")
    total_tokens: int = Field(description="Total tokens")
    cost_breakdown: Dict[str, float] = Field(description="Cost breakdown by service/provider")
    top_projects: List[Dict[str, Any]] = Field(description="Top projects by usage")
    usage_trends: List[Dict[str, Any]] = Field(description="Usage trend data")


# Notification Service Responses
class NotificationResponse(BaseModel):
    """Notification response."""
    
    notification_id: str = Field(description="Notification ID")
    status: str = Field(description="Notification status")
    recipient: str = Field(description="Recipient")
    channel: str = Field(description="Notification channel")
    sent_at: datetime = Field(description="Sent timestamp")
    delivered_at: Optional[datetime] = Field(default=None, description="Delivery timestamp")
    read_at: Optional[datetime] = Field(default=None, description="Read timestamp")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    retry_count: int = Field(description="Retry attempts")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Notification metadata")


class NotificationStatusResponse(BaseModel):
    """Notification status response."""
    
    notification_id: str = Field(description="Notification ID")
    status: str = Field(description="Current status")
    status_history: List[Dict[str, Any]] = Field(description="Status change history")
    delivery_attempts: int = Field(description="Delivery attempts")
    last_attempt: Optional[datetime] = Field(default=None, description="Last delivery attempt")
    next_retry: Optional[datetime] = Field(default=None, description="Next retry time")


class NotificationHistoryResponse(BaseModel):
    """Notification history response."""
    
    notifications: List[NotificationResponse] = Field(description="Notification list")
    total_count: int = Field(description="Total notification count")
    page: int = Field(description="Current page")
    page_size: int = Field(description="Page size")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Applied filters")