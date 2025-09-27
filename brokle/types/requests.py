"""
Request models for Brokle SDK.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class BaseRequest(BaseModel):
    """Base request model."""

    # Brokle specific parameters
    routing_strategy: Optional[str] = Field(
        default=None,
        description="Routing strategy: 'cost_optimized', 'quality_optimized', 'latency_optimized'",
    )
    cache_strategy: Optional[str] = Field(
        default=None, description="Cache strategy: 'semantic', 'exact', 'disabled'"
    )
    cache_similarity_threshold: Optional[float] = Field(
        default=None, description="Similarity threshold for semantic caching (0.0-1.0)"
    )
    max_cost_usd: Optional[float] = Field(
        default=None, description="Maximum cost limit in USD"
    )
    evaluation_metrics: Optional[List[str]] = Field(
        default=None, description="Evaluation metrics to compute"
    )
    custom_tags: Optional[Dict[str, Any]] = Field(
        default=None, description="Custom tags for tracking"
    )

    # A/B testing parameters
    ab_test_enabled: Optional[bool] = Field(
        default=None, description="Enable A/B testing"
    )
    ab_test_config: Optional[Dict[str, Any]] = Field(
        default=None, description="A/B testing configuration"
    )

    # Observability parameters
    trace_id: Optional[str] = Field(
        default=None, description="Trace ID for distributed tracing"
    )
    parent_span_id: Optional[str] = Field(default=None, description="Parent span ID")
    user_id: Optional[str] = Field(default=None, description="User ID for tracking")
    session_id: Optional[str] = Field(
        default=None, description="Session ID for tracking"
    )


class CompletionRequest(BaseRequest):
    """Completion request model."""

    model: str = Field(description="Model name")
    prompt: str = Field(description="Prompt text")
    max_tokens: Optional[int] = Field(
        default=None, description="Maximum tokens to generate"
    )
    temperature: Optional[float] = Field(
        default=None, description="Temperature for sampling"
    )
    top_p: Optional[float] = Field(default=None, description="Top-p for sampling")
    frequency_penalty: Optional[float] = Field(
        default=None, description="Frequency penalty"
    )
    presence_penalty: Optional[float] = Field(
        default=None, description="Presence penalty"
    )
    stop: Optional[Union[str, List[str]]] = Field(
        default=None, description="Stop sequences"
    )
    stream: Optional[bool] = Field(default=False, description="Enable streaming")
    n: Optional[int] = Field(default=1, description="Number of completions")
    logprobs: Optional[int] = Field(
        default=None, description="Number of log probabilities"
    )
    echo: Optional[bool] = Field(default=False, description="Echo the prompt")
    suffix: Optional[str] = Field(default=None, description="Suffix for insertion")
    user: Optional[str] = Field(default=None, description="User identifier")


class ChatCompletionRequest(BaseRequest):
    """Chat completion request model."""

    model: str = Field(description="Model name")
    messages: List[Dict[str, str]] = Field(description="Chat messages")
    max_tokens: Optional[int] = Field(
        default=None, description="Maximum tokens to generate"
    )
    temperature: Optional[float] = Field(
        default=None, description="Temperature for sampling"
    )
    top_p: Optional[float] = Field(default=None, description="Top-p for sampling")
    frequency_penalty: Optional[float] = Field(
        default=None, description="Frequency penalty"
    )
    presence_penalty: Optional[float] = Field(
        default=None, description="Presence penalty"
    )
    stop: Optional[Union[str, List[str]]] = Field(
        default=None, description="Stop sequences"
    )
    stream: Optional[bool] = Field(default=False, description="Enable streaming")
    n: Optional[int] = Field(default=1, description="Number of completions")
    logit_bias: Optional[Dict[str, float]] = Field(
        default=None, description="Logit bias"
    )
    user: Optional[str] = Field(default=None, description="User identifier")

    # Function calling
    functions: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Function definitions"
    )
    function_call: Optional[Union[str, Dict[str, str]]] = Field(
        default=None, description="Function call"
    )

    # Tool calling (newer OpenAI API)
    tools: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Tool definitions"
    )
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(
        default=None, description="Tool choice"
    )

    # Response format
    response_format: Optional[Dict[str, str]] = Field(
        default=None, description="Response format"
    )
    seed: Optional[int] = Field(default=None, description="Random seed")


class EmbeddingRequest(BaseRequest):
    """Embedding request model."""

    model: str = Field(description="Model name")
    input: Union[str, List[str]] = Field(description="Input text(s)")
    encoding_format: Optional[str] = Field(
        default="float", description="Encoding format"
    )
    dimensions: Optional[int] = Field(default=None, description="Number of dimensions")
    user: Optional[str] = Field(default=None, description="User identifier")


class AnalyticsRequest(BaseModel):
    """Analytics request model."""

    start_date: Optional[str] = Field(
        default=None, description="Start date (ISO format)"
    )
    end_date: Optional[str] = Field(default=None, description="End date (ISO format)")
    group_by: Optional[List[str]] = Field(default=None, description="Group by fields")
    filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Filter conditions"
    )
    metrics: Optional[List[str]] = Field(
        default=None, description="Metrics to retrieve"
    )
    granularity: Optional[str] = Field(default="daily", description="Time granularity")
    limit: Optional[int] = Field(default=100, description="Maximum results")
    offset: Optional[int] = Field(default=0, description="Results offset")


class EvaluationRequest(BaseModel):
    """Evaluation request model."""

    response_id: Optional[str] = Field(
        default=None, description="Response ID to evaluate"
    )
    input_text: Optional[str] = Field(default=None, description="Input text")
    output_text: Optional[str] = Field(default=None, description="Output text")
    metrics: List[str] = Field(description="Evaluation metrics")
    reference_text: Optional[str] = Field(
        default=None, description="Reference text for comparison"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional context"
    )
    feedback_type: Optional[str] = Field(default=None, description="Feedback type")
    feedback_value: Optional[Union[str, float, bool]] = Field(
        default=None, description="Feedback value"
    )
    comment: Optional[str] = Field(default=None, description="Optional comment")


# Telemetry Service Requests
class TelemetryTraceRequest(BaseModel):
    """Telemetry trace creation request."""

    trace_id: Optional[str] = Field(default=None, description="Trace ID")
    name: str = Field(description="Trace name")
    user_id: Optional[str] = Field(default=None, description="User ID")
    session_id: Optional[str] = Field(default=None, description="Session ID")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Trace metadata"
    )
    tags: Optional[List[str]] = Field(default=None, description="Trace tags")


class TelemetrySpanRequest(BaseModel):
    """Telemetry span creation request."""

    trace_id: str = Field(description="Parent trace ID")
    span_id: Optional[str] = Field(default=None, description="Span ID")
    parent_span_id: Optional[str] = Field(default=None, description="Parent span ID")
    name: str = Field(description="Span name")
    span_type: str = Field(description="Span type")
    start_time: Optional[str] = Field(
        default=None, description="Start time (ISO format)"
    )
    end_time: Optional[str] = Field(default=None, description="End time (ISO format)")
    attributes: Optional[Dict[str, Any]] = Field(
        default=None, description="Span attributes"
    )
    events: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Span events"
    )
    status: Optional[Dict[str, Any]] = Field(default=None, description="Span status")


class TelemetryEventBatchRequest(BaseModel):
    """Batch telemetry events request."""

    events: List[Dict[str, Any]] = Field(description="Telemetry events")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Batch metadata"
    )


# Cache Service Requests
class CacheGetRequest(BaseModel):
    """Cache get request."""

    key: Optional[str] = Field(default=None, description="Cache key")
    query: Optional[str] = Field(default=None, description="Query for semantic search")
    similarity_threshold: Optional[float] = Field(
        default=0.8, description="Similarity threshold"
    )
    provider: Optional[str] = Field(default=None, description="Provider filter")
    model: Optional[str] = Field(default=None, description="Model filter")


class CacheSetRequest(BaseModel):
    """Cache set request."""

    key: str = Field(description="Cache key")
    value: Dict[str, Any] = Field(description="Cache value")
    ttl: Optional[int] = Field(default=None, description="Time to live in seconds")
    embedding: Optional[List[float]] = Field(
        default=None, description="Embedding vector"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Cache metadata"
    )


class CacheInvalidateRequest(BaseModel):
    """Cache invalidation request."""

    keys: Optional[List[str]] = Field(
        default=None, description="Specific keys to invalidate"
    )
    pattern: Optional[str] = Field(default=None, description="Pattern to match keys")
    provider: Optional[str] = Field(default=None, description="Provider filter")
    model: Optional[str] = Field(default=None, description="Model filter")


class EmbeddingGenerationRequest(BaseModel):
    """Embedding generation request."""

    text: Union[str, List[str]] = Field(description="Text to embed")
    model: Optional[str] = Field(
        default="text-embedding-ada-002", description="Embedding model"
    )
    provider: Optional[str] = Field(default=None, description="Provider preference")


class SemanticSearchRequest(BaseModel):
    """Semantic search request."""

    query: str = Field(description="Search query")
    embedding: Optional[List[float]] = Field(
        default=None, description="Query embedding"
    )
    top_k: Optional[int] = Field(default=10, description="Number of results")
    similarity_threshold: Optional[float] = Field(
        default=0.7, description="Similarity threshold"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Search filters"
    )


# Cost Tracking Service Requests
class CostCalculationRequest(BaseModel):
    """Cost calculation request."""

    provider: str = Field(description="AI provider")
    model: str = Field(description="Model name")
    input_tokens: int = Field(description="Input token count")
    output_tokens: int = Field(description="Output token count")
    request_type: str = Field(description="Request type (completion, chat, embedding)")
    additional_costs: Optional[Dict[str, float]] = Field(
        default=None, description="Additional costs"
    )


class CostTrackingRequest(BaseModel):
    """Cost tracking request."""

    request_id: str = Field(description="Request ID")
    organization_id: str = Field(description="Organization ID")
    environment: str = Field(description="Environment tag")
    provider: str = Field(description="AI provider")
    model: str = Field(description="Model name")
    calculated_cost: float = Field(description="Calculated cost")
    actual_cost: Optional[float] = Field(
        default=None, description="Actual provider cost"
    )
    input_tokens: int = Field(description="Input token count")
    output_tokens: int = Field(description="Output token count")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata"
    )


class BudgetRequest(BaseModel):
    """Budget management request."""

    organization_id: str = Field(description="Organization ID")
    environment: Optional[str] = Field(default=None, description="Environment tag")
    budget_type: str = Field(description="Budget type (monthly, weekly, daily)")
    amount: float = Field(description="Budget amount in USD")
    alert_thresholds: Optional[List[float]] = Field(
        default=None, description="Alert thresholds (0.0-1.0)"
    )
    start_date: Optional[str] = Field(
        default=None, description="Budget start date (ISO format)"
    )
    end_date: Optional[str] = Field(
        default=None, description="Budget end date (ISO format)"
    )


class CostComparisonRequest(BaseModel):
    """Cost comparison request."""

    providers: List[str] = Field(description="Providers to compare")
    model_mappings: Dict[str, str] = Field(description="Model mappings per provider")
    input_tokens: int = Field(description="Input token count")
    output_tokens: int = Field(description="Output token count")
    request_type: str = Field(description="Request type")


# ML Service Requests
class MLRoutingRequest(BaseModel):
    """ML routing recommendation request."""

    model: str = Field(description="Requested model")
    input_text: Optional[str] = Field(
        default=None, description="Input text for analysis"
    )
    routing_strategy: Optional[str] = Field(
        default="balanced", description="Routing strategy"
    )
    constraints: Optional[Dict[str, Any]] = Field(
        default=None, description="Routing constraints"
    )
    historical_data: Optional[Dict[str, Any]] = Field(
        default=None, description="Historical performance data"
    )


class MLTrainingDataRequest(BaseModel):
    """ML training data submission request."""

    request_id: str = Field(description="Request ID")
    provider: str = Field(description="Provider used")
    model: str = Field(description="Model used")
    input_features: Dict[str, Any] = Field(description="Input features")
    performance_metrics: Dict[str, float] = Field(description="Performance metrics")
    outcome: str = Field(description="Outcome (success, failure, timeout)")
    feedback_score: Optional[float] = Field(
        default=None, description="User feedback score"
    )


# Configuration Service Requests
class ConfigRequest(BaseModel):
    """Configuration request."""

    organization_id: Optional[str] = Field(default=None, description="Organization ID")
    environment: Optional[str] = Field(default=None, description="Environment")
    config_keys: Optional[List[str]] = Field(
        default=None, description="Specific config keys"
    )


class FeatureFlagRequest(BaseModel):
    """Feature flag request."""

    organization_id: str = Field(description="Organization ID")
    feature_name: Optional[str] = Field(
        default=None, description="Specific feature name"
    )
    tier: Optional[str] = Field(default=None, description="Subscription tier")


class SubscriptionLimitRequest(BaseModel):
    """Subscription limit request."""

    organization_id: str = Field(description="Organization ID")
    limit_type: Optional[str] = Field(default=None, description="Specific limit type")
    tier: Optional[str] = Field(default=None, description="Subscription tier")


# Billing Service Requests
class UsageRecordingRequest(BaseModel):
    """Usage recording request."""

    organization_id: str = Field(description="Organization ID")
    environment: str = Field(description="Environment tag")
    request_id: str = Field(description="Request ID")
    provider: str = Field(description="AI provider")
    model: str = Field(description="Model used")
    request_type: str = Field(description="Request type")
    input_tokens: int = Field(description="Input token count")
    output_tokens: int = Field(description="Output token count")
    cost_usd: float = Field(description="Cost in USD")
    latency_ms: float = Field(description="Response latency in ms")
    timestamp: Optional[str] = Field(
        default=None, description="Request timestamp (ISO format)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata"
    )


class QuotaCheckRequest(BaseModel):
    """Quota check request."""

    organization_id: str = Field(description="Organization ID")
    environment: Optional[str] = Field(default=None, description="Environment tag")
    resource_type: str = Field(description="Resource type (requests, tokens, cost)")
    requested_amount: int = Field(description="Requested amount")


# Notification Service Requests
class NotificationRequest(BaseModel):
    """Base notification request."""

    recipient: str = Field(description="Recipient (email/phone/user_id)")
    subject: Optional[str] = Field(default=None, description="Notification subject")
    message: str = Field(description="Notification message")
    template_id: Optional[str] = Field(default=None, description="Template ID")
    template_data: Optional[Dict[str, Any]] = Field(
        default=None, description="Template data"
    )
    priority: Optional[str] = Field(
        default="normal", description="Priority (low, normal, high, urgent)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata"
    )


class EmailNotificationRequest(NotificationRequest):
    """Email notification request."""

    sender: Optional[str] = Field(default=None, description="Sender email")
    cc: Optional[List[str]] = Field(default=None, description="CC recipients")
    bcc: Optional[List[str]] = Field(default=None, description="BCC recipients")
    attachments: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Email attachments"
    )


class SMSNotificationRequest(NotificationRequest):
    """SMS notification request."""

    sender: Optional[str] = Field(default=None, description="Sender phone number")


class PushNotificationRequest(NotificationRequest):
    """Push notification request."""

    device_tokens: Optional[List[str]] = Field(
        default=None, description="Device tokens"
    )
    payload: Optional[Dict[str, Any]] = Field(default=None, description="Push payload")


# Observability Service Requests
class ObservabilityTraceRequest(BaseModel):
    """Create trace request for observability service."""

    external_trace_id: str = Field(description="External trace ID")
    name: str = Field(description="Trace name")
    user_id: Optional[str] = Field(default=None, description="User ID")
    session_id: Optional[str] = Field(default=None, description="Session ID")
    parent_trace_id: Optional[str] = Field(default=None, description="Parent trace ID")
    tags: Optional[Dict[str, Any]] = Field(default=None, description="Trace tags")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Trace metadata"
    )


class ObservabilityObservationRequest(BaseModel):
    """Create observation request for observability service."""

    trace_id: str = Field(description="Trace ID")
    external_observation_id: str = Field(description="External observation ID")
    parent_observation_id: Optional[str] = Field(
        default=None, description="Parent observation ID"
    )
    type: str = Field(description="Observation type (llm, span, event, etc.)")
    name: str = Field(description="Observation name")
    start_time: str = Field(description="Start time in ISO format")
    end_time: Optional[str] = Field(default=None, description="End time in ISO format")
    level: Optional[str] = Field(default="DEFAULT", description="Observation level")
    status_message: Optional[str] = Field(default=None, description="Status message")
    version: Optional[str] = Field(default=None, description="Version")
    model: Optional[str] = Field(default=None, description="Model name")
    provider: Optional[str] = Field(default=None, description="Provider name")
    input: Optional[Dict[str, Any]] = Field(default=None, description="Input data")
    output: Optional[Dict[str, Any]] = Field(default=None, description="Output data")
    model_parameters: Optional[Dict[str, Any]] = Field(
        default=None, description="Model parameters"
    )
    prompt_tokens: Optional[int] = Field(default=None, description="Prompt tokens")
    completion_tokens: Optional[int] = Field(
        default=None, description="Completion tokens"
    )
    total_tokens: Optional[int] = Field(default=None, description="Total tokens")
    input_cost: Optional[float] = Field(default=None, description="Input cost")
    output_cost: Optional[float] = Field(default=None, description="Output cost")
    total_cost: Optional[float] = Field(default=None, description="Total cost")


class ObservabilityQualityScoreRequest(BaseModel):
    """Create quality score request for observability service."""

    trace_id: str = Field(description="Trace ID")
    observation_id: Optional[str] = Field(default=None, description="Observation ID")
    score_name: str = Field(description="Score name")
    score_value: Optional[float] = Field(
        default=None, description="Numeric score value"
    )
    string_value: Optional[str] = Field(default=None, description="String score value")
    data_type: str = Field(description="Data type (NUMERIC, CATEGORICAL, BOOLEAN)")
    source: str = Field(description="Score source (API, AUTO, HUMAN, EVAL)")
    evaluator_name: Optional[str] = Field(default=None, description="Evaluator name")
    evaluator_version: Optional[str] = Field(
        default=None, description="Evaluator version"
    )
    comment: Optional[str] = Field(default=None, description="Score comment")
    author_user_id: Optional[str] = Field(default=None, description="Author user ID")


class ObservabilityBatchRequest(BaseModel):
    """Batch request for observability service."""

    traces: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Batch traces"
    )
    observations: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Batch observations"
    )
    quality_scores: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Batch quality scores"
    )
