"""
Brokle OpenTelemetry type definitions and attribute constants.
"""

from .attributes import (
    BrokleOtelSpanAttributes,
    Attrs,  # Convenience alias
    ObservationType,
    ObservationLevel,
    LLMProvider,
    OperationType,
    ScoreDataType,
)

__all__ = [
    "BrokleOtelSpanAttributes",
    "Attrs",
    "ObservationType",
    "ObservationLevel",
    "LLMProvider",
    "OperationType",
    "ScoreDataType",
]