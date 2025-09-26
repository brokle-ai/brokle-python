"""
Type definitions for Brokle SDK.
"""

# from .attributes import BrokleOtelSpanAttributes  # TODO: Add when attributes module is created
from .requests import (
    CompletionRequest,
    ChatCompletionRequest,
    EmbeddingRequest,
    AnalyticsRequest,
    EvaluationRequest,
)
# Phase 2: Import core models from new modular structure
from .responses import (
    CompletionResponse,
    ChatCompletionResponse,
    EmbeddingResponse,
    AnalyticsResponse,
    EvaluationResponse,
)

__all__ = [
    # Requests
    "CompletionRequest",
    "ChatCompletionRequest", 
    "EmbeddingRequest",
    "AnalyticsRequest",
    "EvaluationRequest",
    
    # Responses
    "CompletionResponse",
    "ChatCompletionResponse",
    "EmbeddingResponse",
    "AnalyticsResponse",
    "EvaluationResponse",
]