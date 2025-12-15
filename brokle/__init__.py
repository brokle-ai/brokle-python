"""
Brokle SDK - OpenTelemetry-native observability for AI applications.

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
    ...     name="chat", model="gpt-4", provider="openai"
    ... ) as gen:
    ...     response = openai_client.chat.completions.create(...)
    ...     gen.set_attribute("gen_ai.output.messages", [...])
"""

from ._client import (
    AsyncBrokle,
    Brokle,
    get_async_client,
    get_client,
    reset_async_client,
    reset_client,
)
from .config import BrokleConfig
from .decorators import observe
from .metrics import (
    DURATION_BOUNDARIES,
    TOKEN_BOUNDARIES,
    TTFT_BOUNDARIES,
    GenAIMetrics,
    MetricNames,
    create_genai_metrics,
)
from .transport import (
    TransportType,
    create_metric_exporter,
    create_trace_exporter,
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
from .utils.masking import MaskingHelper
from .evaluations import (
    AsyncEvaluationsManager,
    EvaluationsManager,
)
from .prompts import (
    # Manager classes
    AsyncPromptManager,
    PromptManager,
    # Core classes
    Prompt,
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
    Fallback,
    TextFallback,
    ChatFallback,
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
    # Client
    "Brokle",
    "AsyncBrokle",
    "BrokleConfig",
    "get_client",
    "reset_client",
    "get_async_client",
    "reset_async_client",
    # Decorators
    "observe",
    # Types
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
    # Transport
    "TransportType",
    "create_trace_exporter",
    "create_metric_exporter",
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
    # Utilities
    "MaskingHelper",
    # Prompts
    "PromptManager",
    "AsyncPromptManager",
    "Prompt",
    "PromptCache",
    "CacheOptions",
    "PromptError",
    "PromptNotFoundError",
    "PromptCompileError",
    "PromptFetchError",
    "extract_variables",
    "compile_template",
    "compile_text_template",
    "compile_chat_template",
    "validate_variables",
    "is_text_template",
    "is_chat_template",
    "get_compiled_content",
    "get_compiled_messages",
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
    "Fallback",
    "TextFallback",
    "ChatFallback",
    # Evaluations
    "EvaluationsManager",
    "AsyncEvaluationsManager",
]
