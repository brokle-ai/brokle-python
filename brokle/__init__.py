"""
Brokle SDK - OpenTelemetry-native observability for AI applications.

This SDK leverages OpenTelemetry as the underlying telemetry framework,
providing industry-standard OTLP export with Brokle-specific enhancements
for LLM observability.

Basic Usage:
    >>> from brokle import Brokle
    >>> client = Brokle(api_key="bk_your_secret")
    >>> with client.start_as_current_span("my-operation") as span:
    ...     span.set_attribute("output", "Hello, world!")
    >>> client.flush()

Singleton Pattern:
    >>> from brokle import get_client
    >>> client = get_client()  # Reads from BROKLE_* env vars

LLM Generation Tracking:
    >>> with client.start_as_current_generation(
    ...     name="chat",
    ...     model="gpt-4",
    ...     provider="openai"
    ... ) as gen:
    ...     # Your LLM call
    ...     gen.set_attribute("gen_ai.output.messages", [...])
"""

from .version import __version__, __version_info__
from .config import BrokleConfig
from .client import Brokle, get_client, reset_client
from .decorators import observe
from .types import (
    BrokleOtelSpanAttributes,
    Attrs,
    ObservationType,
    ObservationLevel,
    LLMProvider,
    OperationType,
    ScoreDataType,
)

# Wrappers are imported separately to avoid requiring provider SDKs
# from .wrappers import wrap_openai, wrap_anthropic

__all__ = [
    # Version
    "__version__",
    "__version_info__",

    # Core classes
    "Brokle",
    "BrokleConfig",

    # Client functions
    "get_client",
    "reset_client",

    # Decorators
    "observe",

    # Type constants
    "BrokleOtelSpanAttributes",
    "Attrs",
    "ObservationType",
    "ObservationLevel",
    "LLMProvider",
    "OperationType",
    "ScoreDataType",
]
