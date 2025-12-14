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

from .client import Brokle, get_client, reset_client
from .config import BrokleConfig
from .decorators import observe
from .utils.masking import MaskingHelper
from .metrics import (
    DURATION_BOUNDARIES,
    TOKEN_BOUNDARIES,
    TTFT_BOUNDARIES,
    GenAIMetrics,
    MetricNames,
    create_genai_metrics,
)
from .observations import (
    BrokleAgent,
    BrokleEvent,
    BrokleGeneration,
    BrokleObservation,
    BrokleRetrieval,
    BrokleTool,
    ObservationType,
)
from .streaming import (
    StreamingAccumulator,
    StreamingMetrics,
    StreamingResult,
)
from .transport import (
    TransportType,
    create_metric_exporter,
    create_trace_exporter,
)

# Prompt Management
from .prompt import (
    Prompt,
    PromptClient,
    PromptCache,
    CacheOptions,
    # Exceptions
    PromptError,
    PromptNotFoundError,
    PromptCompileError,
    PromptFetchError,
    # Compiler utilities
    extract_variables,
    compile_template,
    compile_text_template,
    compile_chat_template,
    validate_variables,
    is_text_template,
    is_chat_template,
    get_compiled_content,
    get_compiled_messages,
    # Types
    PromptType,
    MessageRole,
    ChatMessage,
    TextTemplate,
    ChatTemplate,
    Template,
    ModelConfig,
    PromptConfig,
    PromptVersion,
    PromptData,
    GetPromptOptions,
    ListPromptsOptions,
    Pagination,
    PaginatedResponse,
    UpsertPromptRequest,
    CacheEntry,
    OpenAIMessage,
    AnthropicMessage,
    AnthropicRequest,
    Variables,
    FallbackConfig,
)

from .types import (
    Attrs,
    BrokleOtelSpanAttributes,
    LLMProvider,
    OperationType,
    SchemaURLs,
    ScoreDataType,
    SpanLevel,
    SpanType,
)
from .version import __version__, __version_info__

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
    # Masking utilities
    "MaskingHelper",
    # Type constants
    "BrokleOtelSpanAttributes",
    "Attrs",
    "SpanType",
    "SpanLevel",
    "LLMProvider",
    "OperationType",
    "ScoreDataType",
    "SchemaURLs",
    # Metrics
    "GenAIMetrics",
    "create_genai_metrics",
    "MetricNames",
    "TOKEN_BOUNDARIES",
    "DURATION_BOUNDARIES",
    "TTFT_BOUNDARIES",
    # Streaming
    "StreamingAccumulator",
    "StreamingResult",
    "StreamingMetrics",
    # Observations
    "ObservationType",
    "BrokleObservation",
    "BrokleGeneration",
    "BrokleEvent",
    "BrokleAgent",
    "BrokleTool",
    "BrokleRetrieval",
    # Transport
    "TransportType",
    "create_trace_exporter",
    "create_metric_exporter",
    # Prompt Management
    "Prompt",
    "PromptClient",
    "PromptCache",
    "CacheOptions",
    # Prompt Exceptions
    "PromptError",
    "PromptNotFoundError",
    "PromptCompileError",
    "PromptFetchError",
    # Prompt Compiler
    "extract_variables",
    "compile_template",
    "compile_text_template",
    "compile_chat_template",
    "validate_variables",
    "is_text_template",
    "is_chat_template",
    "get_compiled_content",
    "get_compiled_messages",
    # Prompt Types
    "PromptType",
    "MessageRole",
    "ChatMessage",
    "TextTemplate",
    "ChatTemplate",
    "Template",
    "ModelConfig",
    "PromptConfig",
    "PromptVersion",
    "PromptData",
    "GetPromptOptions",
    "ListPromptsOptions",
    "Pagination",
    "PaginatedResponse",
    "UpsertPromptRequest",
    "CacheEntry",
    "OpenAIMessage",
    "AnthropicMessage",
    "AnthropicRequest",
    "Variables",
    "FallbackConfig",
]
