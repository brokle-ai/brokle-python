"""
OpenTelemetry GenAI and Brokle custom attribute constants.

This module provides constants for all OTLP span attributes used across the SDK.
Follows OpenTelemetry GenAI 1.28+ semantic conventions.
See ATTRIBUTE_MAPPING.md for complete cross-platform attribute specification.
"""


class BrokleOtelSpanAttributes:
    """
    OpenTelemetry GenAI 1.28+ and Brokle custom attribute constants.

    Use these constants instead of magic strings to avoid typos and ensure
    OTEL compliance. See ATTRIBUTE_MAPPING.md for detailed specifications.
    """

    # ========== GenAI Provider & Operation (OTEL 1.28+) ==========
    GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"  # e.g. "openai", "anthropic"
    GEN_AI_OPERATION_NAME = "gen_ai.operation.name"  # e.g. "chat", "embeddings"

    # ========== GenAI Request Parameters (OTEL Standard) ==========
    GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
    GEN_AI_REQUEST_TOP_K = "gen_ai.request.top_k"
    GEN_AI_REQUEST_FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
    GEN_AI_REQUEST_PRESENCE_PENALTY = "gen_ai.request.presence_penalty"
    GEN_AI_REQUEST_STOP_SEQUENCES = "gen_ai.request.stop_sequences"
    GEN_AI_REQUEST_USER = "gen_ai.request.user"  # User ID for rate limiting

    # ========== GenAI Response Metadata (OTEL Standard) ==========
    GEN_AI_RESPONSE_ID = "gen_ai.response.id"
    GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"  # Actual model used
    GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
    GEN_AI_RESPONSE_FINISH_REASON = "gen_ai.response.finish_reason"  # Single finish reason

    # ========== GenAI Streaming Metrics (OTEL GenAI Extensions) ==========
    # Time to first token in milliseconds (streaming responses)
    GEN_AI_RESPONSE_TTFT = "gen_ai.response.time_to_first_token"
    # Inter-token latency in milliseconds (streaming responses)
    GEN_AI_RESPONSE_ITL = "gen_ai.response.inter_token_latency"
    # Total response duration in milliseconds
    GEN_AI_RESPONSE_DURATION = "gen_ai.response.duration"

    # ========== GenAI Token Usage (Aliases for metrics) ==========
    GEN_AI_TOKEN_USAGE_INPUT = "gen_ai.token.usage.input"
    GEN_AI_TOKEN_USAGE_OUTPUT = "gen_ai.token.usage.output"
    GEN_AI_TOKEN_USAGE_TOTAL = "gen_ai.token.usage.total"

    # ========== GenAI Messages (OTEL 1.28+ JSON format) ==========
    GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"  # JSON array of messages
    GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"  # JSON array of messages
    GEN_AI_SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"  # System prompts

    # ========== GenAI Usage (OTEL Standard - Optional) ==========
    GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    # Note: total_tokens is NOT in OTEL spec, use brokle.usage.total_tokens

    # ========== GenAI Extended Usage (Cache, Audio, Multi-modal) ==========
    # Cache tokens (Anthropic/OpenAI prompt caching)
    GEN_AI_USAGE_INPUT_TOKENS_CACHE_READ = "gen_ai.usage.input_tokens.cache_read"
    GEN_AI_USAGE_INPUT_TOKENS_CACHE_CREATION = "gen_ai.usage.input_tokens.cache_creation"

    # Audio tokens (OpenAI Whisper, TTS, Realtime API)
    GEN_AI_USAGE_INPUT_AUDIO_TOKENS = "gen_ai.usage.input_audio_tokens"
    GEN_AI_USAGE_OUTPUT_AUDIO_TOKENS = "gen_ai.usage.output_audio_tokens"

    # Reasoning tokens (OpenAI o1 models - internal chain-of-thought)
    GEN_AI_USAGE_REASONING_TOKENS = "gen_ai.usage.reasoning_tokens"

    # Image tokens (GPT-4V, Claude 3.5 Sonnet Vision)
    GEN_AI_USAGE_IMAGE_TOKENS = "gen_ai.usage.image_tokens"

    # Video tokens (Future multi-modal support)
    GEN_AI_USAGE_VIDEO_TOKENS = "gen_ai.usage.video_tokens"

    # ========== OpenAI Specific Attributes ==========
    OPENAI_REQUEST_N = "openai.request.n"
    OPENAI_REQUEST_SERVICE_TIER = "openai.request.service_tier"
    OPENAI_REQUEST_LOGIT_BIAS = "openai.request.logit_bias"
    OPENAI_REQUEST_LOGPROBS = "openai.request.logprobs"
    OPENAI_REQUEST_TOP_LOGPROBS = "openai.request.top_logprobs"
    OPENAI_REQUEST_SEED = "openai.request.seed"
    OPENAI_REQUEST_RESPONSE_FORMAT = "openai.request.response_format"
    OPENAI_REQUEST_TOOLS = "openai.request.tools"
    OPENAI_REQUEST_TOOL_CHOICE = "openai.request.tool_choice"
    OPENAI_REQUEST_PARALLEL_TOOL_CALLS = "openai.request.parallel_tool_calls"
    OPENAI_RESPONSE_SYSTEM_FINGERPRINT = "openai.response.system_fingerprint"

    # ========== Anthropic Specific Attributes ==========
    ANTHROPIC_REQUEST_TOP_K = "anthropic.request.top_k"
    ANTHROPIC_REQUEST_METADATA = "anthropic.request.metadata"
    ANTHROPIC_REQUEST_STOP_SEQUENCES = "anthropic.request.stop_sequences"
    ANTHROPIC_REQUEST_STREAM = "anthropic.request.stream"
    ANTHROPIC_REQUEST_SYSTEM = "anthropic.request.system"
    ANTHROPIC_RESPONSE_STOP_REASON = "anthropic.response.stop_reason"
    ANTHROPIC_RESPONSE_STOP_SEQUENCE = "anthropic.response.stop_sequence"

    # ========== Google Specific Attributes ==========
    GOOGLE_REQUEST_SAFETY_SETTINGS = "google.request.safety_settings"
    GOOGLE_REQUEST_GENERATION_CONFIG = "google.request.generation_config"
    GOOGLE_REQUEST_CANDIDATE_COUNT = "google.request.candidate_count"
    GOOGLE_RESPONSE_SAFETY_RATINGS = "google.response.safety_ratings"

    # ========== Session Tracking (No OTEL GenAI equivalent) ==========
    SESSION_ID = "session.id"  # Session grouping identifier

    # ========== OpenInference Generic Input/Output (Industry Standard) ==========
    # https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md
    INPUT_VALUE = "input.value"  # Generic input data (any format)
    INPUT_MIME_TYPE = "input.mime_type"  # MIME type: "application/json" or "text/plain"
    OUTPUT_VALUE = "output.value"  # Generic output data (any format)
    OUTPUT_MIME_TYPE = "output.mime_type"  # MIME type: "application/json" or "text/plain"

    # ========== Brokle Trace Management ==========
    BROKLE_TRACE_ID = "brokle.trace_id"  # Internal trace ID
    BROKLE_TRACE_NAME = "brokle.trace.name"  # Human-readable trace name
    BROKLE_TRACE_TAGS = "brokle.trace.tags"  # Filterable tags
    BROKLE_TRACE_METADATA = "brokle.trace.metadata"  # Custom metadata
    BROKLE_TRACE_PUBLIC = "brokle.trace.public"  # Public visibility flag

    # ========== Brokle Span Management ==========
    BROKLE_SPAN_ID = "brokle.span_id"
    BROKLE_SPAN_TYPE = "brokle.span.type"  # generation/span/event
    BROKLE_SPAN_NAME = "brokle.span_name"
    BROKLE_PARENT_SPAN_ID = "brokle.parent_span_id"
    BROKLE_SPAN_LEVEL = "brokle.span.level"  # DEBUG/DEFAULT/WARNING/ERROR
    BROKLE_SPAN_VERSION = "brokle.span.version"  # Span-level version for A/B testing

    # ========== Brokle Extended Usage Metrics ==========
    BROKLE_USAGE_TOTAL_TOKENS = "brokle.usage.total_tokens"  # Convenience metric
    BROKLE_USAGE_LATENCY_MS = "brokle.usage.latency_ms"  # Response latency

    # Note: brokle.cost.* attributes are set by BACKEND only (calculated from usage + model pricing)
    # SDKs should NOT set cost attributes - backend calculates costs server-side
    # Usage tracking is flexible - send any combination of token types (standard, cache, audio, reasoning, etc.)
    # Backend supports 10+ token types via flexible usage_details Maps - no schema changes needed for new types

    # ========== Brokle Prompt Management ==========
    BROKLE_PROMPT_ID = "brokle.prompt.id"
    BROKLE_PROMPT_NAME = "brokle.prompt.name"
    BROKLE_PROMPT_VERSION = "brokle.prompt.version"

    # ========== Brokle Quality Scores ==========
    BROKLE_SCORE_NAME = "brokle.score.name"
    BROKLE_SCORE_VALUE = "brokle.score.value"
    BROKLE_SCORE_DATA_TYPE = "brokle.score.data_type"  # numeric/boolean/categorical
    BROKLE_SCORE_COMMENT = "brokle.score.comment"

    # ========== Brokle Internal Flags ==========
    BROKLE_STREAMING = "brokle.streaming"  # Streaming response flag
    BROKLE_PROJECT_ID = "brokle.project_id"  # Project identifier
    BROKLE_ENVIRONMENT = "brokle.environment"  # Environment tag
    BROKLE_VERSION = "brokle.version"  # SDK/app version
    BROKLE_RELEASE = "brokle.release"  # Release/deployment identifier

    # ========== Filterable Metadata (Root Level for Querying) ==========
    # These are promoted to root level in backend for efficient filtering
    USER_ID = "user.id"  # OTEL standard
    TRACE_NAME = "trace_name"  # Maps from brokle.trace.name or span name
    TAGS = "tags"  # Maps from brokle.trace.tags
    METADATA = "metadata"  # Maps from brokle.trace.metadata
    VERSION = "version"  # Maps from brokle.version
    ENVIRONMENT = "environment"  # Maps from brokle.environment

    # ========== Framework Component Attributes (GenAI Extension) ==========
    # For LangChain, LlamaIndex, CrewAI, and other AI framework instrumentation
    # These extend the gen_ai.* namespace following industry patterns (OpenLIT, OpenLLMetry)

    # Framework identification
    GEN_AI_FRAMEWORK_NAME = "gen_ai.framework.name"  # e.g., "langchain", "llamaindex", "crewai"
    GEN_AI_FRAMEWORK_VERSION = "gen_ai.framework.version"
    GEN_AI_COMPONENT_TYPE = "gen_ai.component.type"  # e.g., "agent", "chain", "retriever"

    # Agent-specific attributes
    GEN_AI_AGENT_NAME = "gen_ai.agent.name"
    GEN_AI_AGENT_STRATEGY = "gen_ai.agent.strategy"  # e.g., "react", "cot", "plan_and_execute"
    GEN_AI_AGENT_ITERATION_COUNT = "gen_ai.agent.iteration_count"
    GEN_AI_AGENT_MAX_ITERATIONS = "gen_ai.agent.max_iterations"

    # Tool/function calling attributes
    GEN_AI_TOOL_NAME = "gen_ai.tool.name"
    GEN_AI_TOOL_DESCRIPTION = "gen_ai.tool.description"
    GEN_AI_TOOL_PARAMETERS = "gen_ai.tool.parameters"  # JSON schema of tool params

    # Retrieval/RAG attributes
    GEN_AI_RETRIEVER_TYPE = "gen_ai.retriever.type"  # e.g., "vector", "bm25", "hybrid"
    GEN_AI_RETRIEVAL_TOP_K = "gen_ai.retrieval.top_k"
    GEN_AI_RETRIEVAL_SCORE = "gen_ai.retrieval.score"
    GEN_AI_RETRIEVAL_SOURCE = "gen_ai.retrieval.source"

    # Memory attributes
    GEN_AI_MEMORY_TYPE = "gen_ai.memory.type"  # e.g., "buffer", "summary", "conversation"

    # Execution context attributes
    GEN_AI_EXECUTION_PARALLEL_COUNT = "gen_ai.execution.parallel_count"
    GEN_AI_EXECUTION_SEQUENTIAL_ORDER = "gen_ai.execution.sequential_order"


class SpanType:
    """
    Span type constants for brokle.span_type attribute.

    These align with ObservationType enum in the observations module
    for semantic differentiation in the Brokle backend.
    """

    # Core types
    GENERATION = "generation"  # LLM generation (chat, completion)
    SPAN = "span"  # Generic span
    EVENT = "event"  # Point-in-time event

    # AI Agent types
    TOOL = "tool"  # Tool/function call
    AGENT = "agent"  # AI agent operation
    CHAIN = "chain"  # Chain of operations

    # RAG types
    RETRIEVAL = "retrieval"  # RAG retrieval operation
    EMBEDDING = "embedding"  # Embedding generation

    # Quality & evaluation types
    EVALUATOR = "evaluator"  # Quality evaluation
    GUARDRAIL = "guardrail"  # Safety guardrail check

    # Utility types
    RERANK = "rerank"  # Reranking operation
    WORKFLOW = "workflow"  # High-level workflow orchestration


class SpanLevel:
    """Span level constants for brokle.span.level attribute."""
    DEBUG = "DEBUG"
    DEFAULT = "DEFAULT"
    WARNING = "WARNING"
    ERROR = "ERROR"


class LLMProvider:
    """LLM provider constants for gen_ai.provider.name attribute."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    AZURE_OPENAI = "azure_openai"
    BEDROCK = "bedrock"
    VERTEX_AI = "vertex_ai"
    REPLICATE = "replicate"
    TOGETHER = "together"
    ANYSCALE = "anyscale"
    PERPLEXITY = "perplexity"
    CUSTOM = "custom"


class OperationType:
    """Operation type constants for gen_ai.operation.name attribute."""
    CHAT = "chat"  # Chat completions
    TEXT_COMPLETION = "text_completion"  # Legacy completions
    EMBEDDINGS = "embeddings"  # Text embeddings
    IMAGE_GENERATION = "image_generation"  # Image generation
    AUDIO_TRANSCRIPTION = "audio_transcription"  # Speech to text
    AUDIO_GENERATION = "audio_generation"  # Text to speech
    MODERATION = "moderation"  # Content moderation
    FINE_TUNING = "fine_tuning"  # Model fine-tuning


class ScoreDataType:
    """Score data type constants for brokle.score.data_type attribute."""
    NUMERIC = "numeric"  # Float/int score
    BOOLEAN = "boolean"  # True/false
    CATEGORICAL = "categorical"  # Category/enum value


class ComponentType:
    """
    Standard component types for framework instrumentation.

    Use with gen_ai.component.type attribute.
    """
    AGENT = "agent"  # AI agent operation
    CHAIN = "chain"  # Chain/pipeline of operations
    RETRIEVER = "retriever"  # RAG retrieval component
    MEMORY = "memory"  # Memory/context management
    TOOL = "tool"  # Tool/function call
    WORKFLOW = "workflow"  # High-level workflow orchestration
    PLANNER = "planner"  # Planning component


class AgentStrategy:
    """
    Standard agent strategy types.

    Use with gen_ai.agent.strategy attribute.
    """
    REACT = "react"  # ReAct: Reasoning + Acting
    COT = "cot"  # Chain of Thought
    PLAN_AND_EXECUTE = "plan_and_execute"  # Plan then execute
    TREE_OF_THOUGHT = "tree_of_thought"  # Tree of Thought
    REFLEXION = "reflexion"  # Reflexion pattern
    SELF_ASK = "self_ask"  # Self-Ask pattern
    ZERO_SHOT = "zero_shot"  # Zero-shot prompting


class RetrieverType:
    """
    Standard retriever types for RAG pipelines.

    Use with gen_ai.retriever.type attribute.
    """
    VECTOR = "vector"  # Vector/embedding similarity
    BM25 = "bm25"  # BM25 keyword search
    HYBRID = "hybrid"  # Hybrid (vector + keyword)
    KEYWORD = "keyword"  # Keyword/TF-IDF search
    SEMANTIC = "semantic"  # Semantic search
    MULTI_QUERY = "multi_query"  # Multi-query retrieval
    PARENT_DOCUMENT = "parent_document"  # Parent document retrieval


class MemoryType:
    """
    Standard memory types for AI frameworks.

    Use with gen_ai.memory.type attribute.
    """
    BUFFER = "buffer"  # Simple buffer memory
    SUMMARY = "summary"  # Summary-based memory
    CONVERSATION = "conversation"  # Conversation history
    ENTITY = "entity"  # Entity-based memory
    KNOWLEDGE_GRAPH = "knowledge_graph"  # Knowledge graph memory
    VECTOR = "vector"  # Vector store memory


# Convenience aliases for common usage
Attrs = BrokleOtelSpanAttributes


class SchemaURLs:
    """
    OpenTelemetry semantic convention schema URLs.

    Schema URLs provide versioning for semantic conventions, allowing backends
    to understand which version of the conventions a span was recorded with.
    This enables forward compatibility as conventions evolve.

    See: https://opentelemetry.io/docs/specs/otel/schemas/
    """

    # OpenTelemetry GenAI 1.28+ semantic conventions
    # This is the primary schema for LLM observability attributes
    OTEL_GENAI_1_28 = "https://opentelemetry.io/schemas/1.28.0"

    # OpenTelemetry GenAI 1.29+ (includes additional token types)
    OTEL_GENAI_1_29 = "https://opentelemetry.io/schemas/1.29.0"

    # OpenInference semantic conventions (Arize Phoenix)
    # https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md
    OPENINFERENCE_1_0 = "https://arize.com/openinference/1.0.0"

    # Current default schema URL for Brokle SDK
    # Uses OTEL 1.28+ which includes all GenAI attributes we support
    DEFAULT = OTEL_GENAI_1_28