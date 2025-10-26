"""
Core AI response models for Brokle SDK.

This module contains the most commonly used response models:
- ChatCompletionResponse
- EmbeddingResponse
- CompletionResponse

Models follow industry standard patterns with clean namespace separation
via response.brokle.* for all platform metadata and insights.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Import BaseResponse and BrokleMetadata from base module
# Import our mixins and base classes
from .base import (
    BaseResponse,
    BrokleMetadata,
    BrokleResponseBase,
    CostTrackingMixin,
    ProviderMixin,
    TimestampMixin,
    TokenUsageMixin,
)


# Supporting model classes (unchanged for compatibility)
class CompletionChoice(BaseModel):
    """Completion choice model."""

    text: str = Field(description="Generated text")
    index: int = Field(description="Choice index")
    logprobs: Optional[Dict[str, Any]] = Field(
        default=None, description="Log probabilities"
    )
    finish_reason: Optional[str] = Field(default=None, description="Finish reason")


class ChatCompletionMessage(BaseModel):
    """Chat completion message model."""

    role: str = Field(description="Message role")
    content: Optional[str] = Field(default=None, description="Message content")
    function_call: Optional[Dict[str, Any]] = Field(
        default=None, description="Function call"
    )
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Tool calls"
    )
    name: Optional[str] = Field(default=None, description="Function name")


class ChatCompletionChoice(BaseModel):
    """Chat completion choice model."""

    index: int = Field(description="Choice index")
    message: ChatCompletionMessage = Field(description="Message")
    finish_reason: Optional[str] = Field(default=None, description="Finish reason")
    logprobs: Optional[Dict[str, Any]] = Field(
        default=None, description="Log probabilities"
    )


class EmbeddingData(BaseModel):
    """Embedding data model."""

    object: str = Field(description="Object type")
    embedding: List[float] = Field(description="Embedding vector")
    index: int = Field(description="Embedding index")


# Core Response Models - Industry Standard Pattern
class CompletionResponse(BaseResponse):
    """
    Completion response model following industry standard pattern.

    Platform metadata accessible via response.brokle.* namespace.
    AI response data accessible directly via response.choices, response.model, etc.
    """

    id: str = Field(description="Response ID")
    object: str = Field(description="Object type")
    created: int = Field(description="Creation timestamp")
    model: str = Field(description="Model used")
    choices: List[CompletionChoice] = Field(description="Generated choices")
    usage: Optional[Dict[str, int]] = Field(default=None, description="Token usage")


class ChatCompletionResponse(BaseResponse):
    """
    Chat completion response model following industry standard pattern.

    Most commonly used response model in the SDK. Platform metadata accessible
    via response.brokle.* namespace like AWS ResponseMetadata pattern.
    """

    id: str = Field(description="Response ID")
    object: str = Field(description="Object type")
    created: int = Field(description="Creation timestamp")
    model: str = Field(description="Model used")
    choices: List[ChatCompletionChoice] = Field(description="Generated choices")
    usage: Optional[Dict[str, int]] = Field(default=None, description="Token usage")
    system_fingerprint: Optional[str] = Field(
        default=None, description="System fingerprint"
    )


class EmbeddingResponse(BaseResponse):
    """
    Embedding response model.

    Maintains complete backward compatibility with original while internally
    leveraging mixins for common field patterns.
    """

    object: str = Field(description="Object type")
    data: List[EmbeddingData] = Field(description="Embedding data")
    model: str = Field(description="Model used")
    usage: Optional[Dict[str, int]] = Field(default=None, description="Token usage")


# Re-export for backward compatibility
__all__ = [
    # Core Response Models (backward compatible)
    "ChatCompletionResponse",
    "EmbeddingResponse",
    "CompletionResponse",
    # Supporting Models
    "ChatCompletionMessage",
    "ChatCompletionChoice",
    "EmbeddingData",
    "CompletionChoice",
]
