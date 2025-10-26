"""
Type definitions for Brokle SDK.
"""

# from .attributes import BrokleOtelSpanAttributes  # TODO: Add when attributes module is created
from .observability import (
    Observation,
    ObservationLevel,
    ObservationType,
    Score,
    ScoreDataType,
    ScoreSource,
    Session,
    Trace,
)
from .requests import (
    AnalyticsRequest,
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
    EvaluationRequest,
)

# Import from modular response structure
from .responses.core import (
    ChatCompletionResponse,
    CompletionResponse,
    EmbeddingResponse,
)
from .responses.observability import (
    AnalyticsResponse,
    EvaluationResponse,
)
from .telemetry import (
    BatchEventError,
    TelemetryBatchRequest,
    TelemetryBatchResponse,
    TelemetryEvent,
    TelemetryEventType,
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
    # Observability entities
    "Trace",
    "Observation",
    "Score",
    "Session",
    "ObservationType",
    "ObservationLevel",
    "ScoreDataType",
    "ScoreSource",
    # Telemetry batch API
    "TelemetryEvent",
    "TelemetryEventType",
    "TelemetryBatchRequest",
    "TelemetryBatchResponse",
    "BatchEventError",
]
