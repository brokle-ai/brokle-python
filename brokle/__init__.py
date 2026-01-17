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
    set_async_client,
    set_client,
)
from ._utils.sync import run_sync, run_sync_safely
from .config import BrokleConfig

# New namespace modules (recommended)
from .datasets import (
    AsyncDataset,
    AsyncDatasetsManager,
    Dataset,
    DatasetData,
    DatasetError,
    DatasetItem,
    DatasetItemInput,
    DatasetsManager,
)
from .decorators import observe
from .experiments import (
    AsyncExperimentsManager,
    EvaluationError,
    EvaluationItem,
    EvaluationResults,
    Experiment,
    ExperimentsManager,
    ScorerExecutionError,
    SummaryStats,
    TaskError,
)
from .experiments.types import (
    SpanExtractExpected,
    SpanExtractInput,
    SpanExtractOutput,
)
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
from .prompts import (  # Manager classes; Core classes; Exceptions; Compiler utilities; Types
    AnthropicMessage,
    AnthropicRequest,
    AsyncPromptManager,
    CacheEntry,
    CacheOptions,
    ChatFallback,
    ChatMessage,
    ChatTemplate,
    Fallback,
    GetPromptOptions,
    ListPromptsOptions,
    MessageRole,
    ModelConfig,
    OpenAIMessage,
    PaginatedResponse,
    Pagination,
    Prompt,
    PromptCache,
    PromptCompileError,
    PromptConfig,
    PromptData,
    PromptError,
    PromptFetchError,
    PromptManager,
    PromptNotFoundError,
    PromptType,
    PromptVersion,
    Template,
    TextFallback,
    TextTemplate,
    UpsertPromptRequest,
    Variables,
    compile_chat_template,
    compile_template,
    compile_text_template,
    extract_variables,
    get_compiled_content,
    get_compiled_messages,
    is_chat_template,
    is_text_template,
    validate_variables,
)
from .query import (
    AsyncQueryManager,
    InvalidFilterError,
    QueriedSpan,
    QueryAPIError,
    QueryError,
    QueryManager,
    QueryResult,
    SpanEvent,
    TokenUsage,
    ValidationResult,
)
from .scorers import (  # Built-in scorers; LLM-as-Judge scorers; Decorators
    Contains,
    ExactMatch,
    JSONValid,
    LengthCheck,
    LLMScorer,
    RegexMatch,
    multi_scorer,
    scorer,
)
from .scores import (
    AsyncScoresManager,
    ScoreError,
    Scorer,
    ScorerArgs,
    ScorerError,
    ScoreResult,
    ScorerProtocol,
    ScoresManager,
    ScoreSource,
    ScoreType,
    ScoreValue,
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
from .utils.masking import MaskingHelper
from .version import __version__, __version_info__

# Wrappers are imported separately to avoid requiring provider SDKs
# Usage: from brokle.wrappers import wrap_openai, wrap_anthropic, wrap_google, etc.
# Available wrappers:
#   - wrap_openai, wrap_openai_async (OpenAI)
#   - wrap_anthropic, wrap_anthropic_async (Anthropic)
#   - wrap_azure_openai, wrap_azure_openai_async (Azure OpenAI)
#   - wrap_google (Google GenAI)
#   - wrap_mistral (Mistral AI)
#   - wrap_cohere (Cohere)
#   - wrap_bedrock (AWS Bedrock)

__all__ = [
    # Version
    "__version__",
    "__version_info__",
    # Client
    "Brokle",
    "AsyncBrokle",
    "BrokleConfig",
    "get_client",
    "set_client",
    "reset_client",
    "get_async_client",
    "set_async_client",
    "reset_async_client",
    # Sync Utilities
    "run_sync",
    "run_sync_safely",
    # Decorators
    "observe",
    # Types
    "BrokleOtelSpanAttributes",
    "Attrs",
    "SpanType",
    "SpanLevel",
    "LLMProvider",
    "OperationType",
    "ScoreType",
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
    # Datasets
    "DatasetsManager",
    "AsyncDatasetsManager",
    "Dataset",
    "AsyncDataset",
    "DatasetItem",
    "DatasetItemInput",
    "DatasetData",
    "DatasetError",
    # Scores
    "ScoresManager",
    "AsyncScoresManager",
    "ScoreType",
    "ScoreSource",
    "ScoreResult",
    "ScoreValue",
    "ScorerProtocol",
    "Scorer",
    "ScorerArgs",
    "ScoreError",
    "ScorerError",
    # Scorers
    "ExactMatch",
    "Contains",
    "RegexMatch",
    "JSONValid",
    "LengthCheck",
    "LLMScorer",
    "scorer",
    "multi_scorer",
    # Experiments
    "ExperimentsManager",
    "AsyncExperimentsManager",
    "EvaluationResults",
    "EvaluationItem",
    "SummaryStats",
    "Experiment",
    "EvaluationError",
    "TaskError",
    "ScorerExecutionError",
    # Query (THE WEDGE)
    "QueryManager",
    "AsyncQueryManager",
    "QueriedSpan",
    "QueryResult",
    "ValidationResult",
    "TokenUsage",
    "SpanEvent",
    "QueryError",
    "InvalidFilterError",
    "QueryAPIError",
    # Span Extract Types (for span-based evaluation)
    "SpanExtractInput",
    "SpanExtractOutput",
    "SpanExtractExpected",
]
