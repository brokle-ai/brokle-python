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
# Import from modular response structure
from .responses.core import (
    CompletionResponse,
    ChatCompletionResponse,
    EmbeddingResponse,
)
from .responses.observability import (
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