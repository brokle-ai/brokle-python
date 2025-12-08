"""
Brokle OpenTelemetry type definitions and attribute constants.
"""

from .attributes import Attrs  # Convenience alias
from .attributes import (
    BrokleOtelSpanAttributes,
    LLMProvider,
    OperationType,
    SchemaURLs,
    ScoreDataType,
    SpanLevel,
    SpanType,
)

__all__ = [
    "BrokleOtelSpanAttributes",
    "Attrs",
    "SpanType",
    "SpanLevel",
    "LLMProvider",
    "OperationType",
    "ScoreDataType",
    "SchemaURLs",
]
