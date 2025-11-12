"""
Brokle OpenTelemetry type definitions and attribute constants.
"""

from .attributes import (
    BrokleOtelSpanAttributes,
    Attrs,  # Convenience alias
    SpanType,
    SpanLevel,
    LLMProvider,
    OperationType,
    ScoreDataType,
)

__all__ = [
    "BrokleOtelSpanAttributes",
    "Attrs",
    "SpanType",
    "SpanLevel",
    "LLMProvider",
    "OperationType",
    "ScoreDataType",
]