"""
OpenTelemetry span attributes for Brokle Platform.

This module defines constants and functions for managing OpenTelemetry span attributes
used by Brokle SDK. It follows the LangFuse pattern but adapted for Brokle's
specific features like intelligent routing, cost optimization, and evaluation.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel


class BrokleOtelSpanAttributes:
    """OpenTelemetry span attribute constants for Brokle Platform."""
    
    # Trace-level attributes
    TRACE_NAME = "brokle.trace.name"
    TRACE_USER_ID = "user.id"
    TRACE_SESSION_ID = "session.id"
    TRACE_ORGANIZATION_ID = "brokle.trace.organization_id"
    TRACE_SECRET_KEY = "brokle.trace.secret_key"
    TRACE_ENVIRONMENT = "brokle.trace.environment"
    TRACE_TAGS = "brokle.trace.tags"
    TRACE_METADATA = "brokle.trace.metadata"
    TRACE_INPUT = "brokle.trace.input"
    TRACE_OUTPUT = "brokle.trace.output"
    TRACE_PUBLIC = "brokle.trace.public"
    
    # Span-level attributes
    SPAN_TYPE = "brokle.span.type"
    SPAN_NAME = "brokle.span.name"
    SPAN_INPUT = "brokle.span.input"
    SPAN_OUTPUT = "brokle.span.output"
    SPAN_LEVEL = "brokle.span.level"
    SPAN_STATUS_MESSAGE = "brokle.span.status_message"
    SPAN_METADATA = "brokle.span.metadata"
    SPAN_VERSION = "brokle.span.version"
    SPAN_RELEASE = "brokle.span.release"
    
    # Generation-level attributes (LLM calls)
    GENERATION_TYPE = "brokle.generation.type"
    GENERATION_NAME = "brokle.generation.name"
    GENERATION_MODEL = "brokle.generation.model"
    GENERATION_PROVIDER = "brokle.generation.provider"
    GENERATION_PROVIDER_MODEL_ID = "brokle.generation.provider_model_id"
    GENERATION_REQUEST_TYPE = "brokle.generation.request_type"
    GENERATION_INPUT = "brokle.generation.input"
    GENERATION_OUTPUT = "brokle.generation.output"
    GENERATION_COMPLETION_START_TIME = "brokle.generation.completion_start_time"
    GENERATION_MODEL_PARAMETERS = "brokle.generation.model_parameters"
    GENERATION_USAGE_DETAILS = "brokle.generation.usage_details"
    GENERATION_COST_DETAILS = "brokle.generation.cost_details"
    GENERATION_PROMPT_NAME = "brokle.generation.prompt_name"
    GENERATION_PROMPT_VERSION = "brokle.generation.prompt_version"
    
    # Token usage attributes
    TOKENS_INPUT = "brokle.tokens.input"
    TOKENS_OUTPUT = "brokle.tokens.output"
    TOKENS_TOTAL = "brokle.tokens.total"
    
    # Cost attributes
    COST_USD = "brokle.cost.usd"
    COST_INPUT_USD = "brokle.cost.input_usd"
    COST_OUTPUT_USD = "brokle.cost.output_usd"
    COST_TOTAL_USD = "brokle.cost.total_usd"
    
    # Timing attributes
    LATENCY_MS = "brokle.latency.ms"
    LATENCY_PROVIDER_MS = "brokle.latency.provider_ms"
    LATENCY_GATEWAY_MS = "brokle.latency.gateway_ms"
    LATENCY_TOTAL_MS = "brokle.latency.total_ms"
    
    # Routing-specific attributes
    ROUTING_STRATEGY = "brokle.routing.strategy"
    ROUTING_REASON = "brokle.routing.reason"
    ROUTING_DECISION = "brokle.routing.decision"
    ROUTING_COST_FACTOR = "brokle.routing.cost_factor"
    ROUTING_LATENCY_FACTOR = "brokle.routing.latency_factor"
    ROUTING_QUALITY_FACTOR = "brokle.routing.quality_factor"
    ROUTING_LOAD_FACTOR = "brokle.routing.load_factor"
    ROUTING_CONFIDENCE = "brokle.routing.confidence"
    
    # Caching attributes
    CACHE_ENABLED = "brokle.cache.enabled"
    CACHE_STRATEGY = "brokle.cache.strategy"
    CACHE_HIT = "brokle.cache.hit"
    CACHE_KEY = "brokle.cache.key"
    CACHE_TTL = "brokle.cache.ttl"
    CACHE_SIMILARITY_SCORE = "brokle.cache.similarity_score"
    CACHE_SIMILARITY_THRESHOLD = "brokle.cache.similarity_threshold"
    
    # Evaluation attributes
    EVALUATION_ENABLED = "brokle.evaluation.enabled"
    EVALUATION_SCORES = "brokle.evaluation.scores"
    EVALUATION_FEEDBACK = "brokle.evaluation.feedback"
    EVALUATION_QUALITY_SCORE = "brokle.evaluation.quality_score"
    EVALUATION_RELEVANCE_SCORE = "brokle.evaluation.relevance_score"
    EVALUATION_ACCURACY_SCORE = "brokle.evaluation.accuracy_score"
    EVALUATION_SAFETY_SCORE = "brokle.evaluation.safety_score"
    EVALUATION_COHERENCE_SCORE = "brokle.evaluation.coherence_score"
    
    # A/B Testing attributes
    AB_TEST_ENABLED = "brokle.ab_test.enabled"
    AB_TEST_ID = "brokle.ab_test.id"
    AB_TEST_VARIANT = "brokle.ab_test.variant"
    AB_TEST_EXPERIMENT = "brokle.ab_test.experiment"
    AB_TEST_CONTROL_GROUP = "brokle.ab_test.control_group"
    
    # Error attributes
    ERROR_TYPE = "brokle.error.type"
    ERROR_CODE = "brokle.error.code"
    ERROR_MESSAGE = "brokle.error.message"
    ERROR_STACK_TRACE = "brokle.error.stack_trace"
    ERROR_PROVIDER = "brokle.error.provider"
    ERROR_RETRYABLE = "brokle.error.retryable"
    
    # Performance attributes
    PERFORMANCE_QUEUE_TIME = "brokle.performance.queue_time"
    PERFORMANCE_PROCESSING_TIME = "brokle.performance.processing_time"
    PERFORMANCE_TOTAL_TIME = "brokle.performance.total_time"
    
    # SDK attributes
    SDK_VERSION = "brokle.sdk.version"
    SDK_LANGUAGE = "brokle.sdk.language"
    SDK_INTEGRATION_TYPE = "brokle.sdk.integration_type"
    
    # Environment attributes
    ENVIRONMENT = "brokle.environment"
    RELEASE = "brokle.release"
    VERSION = "brokle.version"


def create_trace_attributes(
    *,
    name: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    secret_key: Optional[str] = None,
    environment: Optional[str] = None,
    version: Optional[str] = None,
    release: Optional[str] = None,
    input: Optional[Any] = None,
    output: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    public: Optional[bool] = None,
) -> Dict[str, Any]:
    """Create trace-level attributes."""
    attributes = {
        BrokleOtelSpanAttributes.TRACE_NAME: name,
        BrokleOtelSpanAttributes.TRACE_USER_ID: user_id,
        BrokleOtelSpanAttributes.TRACE_SESSION_ID: session_id,
        BrokleOtelSpanAttributes.TRACE_ORGANIZATION_ID: organization_id,
        BrokleOtelSpanAttributes.TRACE_SECRET_KEY: secret_key,
        BrokleOtelSpanAttributes.TRACE_ENVIRONMENT: environment,
        BrokleOtelSpanAttributes.VERSION: version,
        BrokleOtelSpanAttributes.RELEASE: release,
        BrokleOtelSpanAttributes.TRACE_INPUT: _serialize(input),
        BrokleOtelSpanAttributes.TRACE_OUTPUT: _serialize(output),
        BrokleOtelSpanAttributes.TRACE_TAGS: tags,
        BrokleOtelSpanAttributes.TRACE_PUBLIC: public,
    }
    
    # Add flattened metadata
    if metadata:
        metadata_attrs = _flatten_and_serialize_metadata(metadata, "trace")
        attributes.update(metadata_attrs)
    
    # Remove None values
    return {k: v for k, v in attributes.items() if v is not None}


def create_span_attributes(
    *,
    span_type: str = "span",
    name: Optional[str] = None,
    input: Optional[Any] = None,
    output: Optional[Any] = None,
    level: Optional[str] = None,
    status_message: Optional[str] = None,
    version: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create span-level attributes."""
    attributes = {
        BrokleOtelSpanAttributes.SPAN_TYPE: span_type,
        BrokleOtelSpanAttributes.SPAN_NAME: name,
        BrokleOtelSpanAttributes.SPAN_INPUT: _serialize(input),
        BrokleOtelSpanAttributes.SPAN_OUTPUT: _serialize(output),
        BrokleOtelSpanAttributes.SPAN_LEVEL: level,
        BrokleOtelSpanAttributes.SPAN_STATUS_MESSAGE: status_message,
        BrokleOtelSpanAttributes.SPAN_VERSION: version,
    }
    
    # Add flattened metadata
    if metadata:
        metadata_attrs = _flatten_and_serialize_metadata(metadata, "span")
        attributes.update(metadata_attrs)
    
    # Remove None values
    return {k: v for k, v in attributes.items() if v is not None}


def create_generation_attributes(
    *,
    name: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    provider_model_id: Optional[str] = None,
    request_type: Optional[str] = None,
    input: Optional[Any] = None,
    output: Optional[Any] = None,
    completion_start_time: Optional[datetime] = None,
    model_parameters: Optional[Dict[str, Any]] = None,
    usage_details: Optional[Dict[str, Any]] = None,
    cost_details: Optional[Dict[str, Any]] = None,
    prompt_name: Optional[str] = None,
    prompt_version: Optional[str] = None,
    level: Optional[str] = None,
    status_message: Optional[str] = None,
    version: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create generation-level attributes for LLM calls."""
    attributes = {
        BrokleOtelSpanAttributes.GENERATION_TYPE: "generation",
        BrokleOtelSpanAttributes.GENERATION_NAME: name,
        BrokleOtelSpanAttributes.GENERATION_MODEL: model,
        BrokleOtelSpanAttributes.GENERATION_PROVIDER: provider,
        BrokleOtelSpanAttributes.GENERATION_PROVIDER_MODEL_ID: provider_model_id,
        BrokleOtelSpanAttributes.GENERATION_REQUEST_TYPE: request_type,
        BrokleOtelSpanAttributes.GENERATION_INPUT: _serialize(input),
        BrokleOtelSpanAttributes.GENERATION_OUTPUT: _serialize(output),
        BrokleOtelSpanAttributes.GENERATION_COMPLETION_START_TIME: _serialize(completion_start_time),
        BrokleOtelSpanAttributes.GENERATION_MODEL_PARAMETERS: _serialize(model_parameters),
        BrokleOtelSpanAttributes.GENERATION_USAGE_DETAILS: _serialize(usage_details),
        BrokleOtelSpanAttributes.GENERATION_COST_DETAILS: _serialize(cost_details),
        BrokleOtelSpanAttributes.GENERATION_PROMPT_NAME: prompt_name,
        BrokleOtelSpanAttributes.GENERATION_PROMPT_VERSION: prompt_version,
        BrokleOtelSpanAttributes.SPAN_LEVEL: level,
        BrokleOtelSpanAttributes.SPAN_STATUS_MESSAGE: status_message,
        BrokleOtelSpanAttributes.SPAN_VERSION: version,
    }
    
    # Add flattened metadata
    if metadata:
        metadata_attrs = _flatten_and_serialize_metadata(metadata, "generation")
        attributes.update(metadata_attrs)
    
    # Remove None values
    return {k: v for k, v in attributes.items() if v is not None}


def create_routing_attributes(
    *,
    strategy: Optional[str] = None,
    reason: Optional[str] = None,
    decision: Optional[str] = None,
    cost_factor: Optional[float] = None,
    latency_factor: Optional[float] = None,
    quality_factor: Optional[float] = None,
    load_factor: Optional[float] = None,
    confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """Create routing-specific attributes."""
    attributes = {
        BrokleOtelSpanAttributes.ROUTING_STRATEGY: strategy,
        BrokleOtelSpanAttributes.ROUTING_REASON: reason,
        BrokleOtelSpanAttributes.ROUTING_DECISION: decision,
        BrokleOtelSpanAttributes.ROUTING_COST_FACTOR: cost_factor,
        BrokleOtelSpanAttributes.ROUTING_LATENCY_FACTOR: latency_factor,
        BrokleOtelSpanAttributes.ROUTING_QUALITY_FACTOR: quality_factor,
        BrokleOtelSpanAttributes.ROUTING_LOAD_FACTOR: load_factor,
        BrokleOtelSpanAttributes.ROUTING_CONFIDENCE: confidence,
    }
    
    # Remove None values
    return {k: v for k, v in attributes.items() if v is not None}


def create_cache_attributes(
    *,
    enabled: Optional[bool] = None,
    strategy: Optional[str] = None,
    hit: Optional[bool] = None,
    key: Optional[str] = None,
    ttl: Optional[int] = None,
    similarity_score: Optional[float] = None,
    similarity_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Create cache-specific attributes."""
    attributes = {
        BrokleOtelSpanAttributes.CACHE_ENABLED: enabled,
        BrokleOtelSpanAttributes.CACHE_STRATEGY: strategy,
        BrokleOtelSpanAttributes.CACHE_HIT: hit,
        BrokleOtelSpanAttributes.CACHE_KEY: key,
        BrokleOtelSpanAttributes.CACHE_TTL: ttl,
        BrokleOtelSpanAttributes.CACHE_SIMILARITY_SCORE: similarity_score,
        BrokleOtelSpanAttributes.CACHE_SIMILARITY_THRESHOLD: similarity_threshold,
    }
    
    # Remove None values
    return {k: v for k, v in attributes.items() if v is not None}


def create_evaluation_attributes(
    *,
    enabled: Optional[bool] = None,
    scores: Optional[Dict[str, float]] = None,
    feedback: Optional[str] = None,
    quality_score: Optional[float] = None,
    relevance_score: Optional[float] = None,
    accuracy_score: Optional[float] = None,
    safety_score: Optional[float] = None,
    coherence_score: Optional[float] = None,
) -> Dict[str, Any]:
    """Create evaluation-specific attributes."""
    attributes = {
        BrokleOtelSpanAttributes.EVALUATION_ENABLED: enabled,
        BrokleOtelSpanAttributes.EVALUATION_SCORES: _serialize(scores),
        BrokleOtelSpanAttributes.EVALUATION_FEEDBACK: feedback,
        BrokleOtelSpanAttributes.EVALUATION_QUALITY_SCORE: quality_score,
        BrokleOtelSpanAttributes.EVALUATION_RELEVANCE_SCORE: relevance_score,
        BrokleOtelSpanAttributes.EVALUATION_ACCURACY_SCORE: accuracy_score,
        BrokleOtelSpanAttributes.EVALUATION_SAFETY_SCORE: safety_score,
        BrokleOtelSpanAttributes.EVALUATION_COHERENCE_SCORE: coherence_score,
    }
    
    # Remove None values
    return {k: v for k, v in attributes.items() if v is not None}


def create_error_attributes(
    *,
    error_type: Optional[str] = None,
    error_code: Optional[str] = None,
    error_message: Optional[str] = None,
    stack_trace: Optional[str] = None,
    provider: Optional[str] = None,
    retryable: Optional[bool] = None,
) -> Dict[str, Any]:
    """Create error-specific attributes."""
    attributes = {
        BrokleOtelSpanAttributes.ERROR_TYPE: error_type,
        BrokleOtelSpanAttributes.ERROR_CODE: error_code,
        BrokleOtelSpanAttributes.ERROR_MESSAGE: error_message,
        BrokleOtelSpanAttributes.ERROR_STACK_TRACE: stack_trace,
        BrokleOtelSpanAttributes.ERROR_PROVIDER: provider,
        BrokleOtelSpanAttributes.ERROR_RETRYABLE: retryable,
    }
    
    # Remove None values
    return {k: v for k, v in attributes.items() if v is not None}


def _serialize(obj: Any) -> Optional[str]:
    """Serialize object to JSON string."""
    if obj is None:
        return None
    
    try:
        if isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return json.dumps(obj, default=_json_serializer)
    except Exception:
        return str(obj)


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for complex objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, BaseModel):
        return obj.model_dump()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return str(obj)


def _flatten_and_serialize_metadata(
    metadata: Dict[str, Any], 
    prefix_type: str
) -> Dict[str, Any]:
    """Flatten and serialize metadata with appropriate prefix."""
    if prefix_type == "trace":
        prefix = BrokleOtelSpanAttributes.TRACE_METADATA
    elif prefix_type == "span":
        prefix = BrokleOtelSpanAttributes.SPAN_METADATA
    elif prefix_type == "generation":
        prefix = BrokleOtelSpanAttributes.SPAN_METADATA
    else:
        prefix = f"brokle.{prefix_type}.metadata"
    
    flattened = {}
    
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            flattened[f"{prefix}.{key}"] = value
        else:
            flattened[f"{prefix}.{key}"] = _serialize(value)
    
    return flattened