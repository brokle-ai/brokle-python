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

    # ========== GenAI Messages (OTEL 1.28+ JSON format) ==========
    GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"  # JSON array of messages
    GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"  # JSON array of messages
    GEN_AI_SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"  # System prompts

    # ========== GenAI Usage (OTEL Standard - Optional) ==========
    GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    # Note: total_tokens is NOT in OTEL spec, use brokle.usage.total_tokens

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

    # ========== Brokle Trace Management ==========
    BROKLE_TRACE_ID = "brokle.trace_id"  # Internal trace ID
    BROKLE_TRACE_NAME = "brokle.trace.name"  # Human-readable trace name
    BROKLE_TRACE_TAGS = "brokle.trace.tags"  # Filterable tags
    BROKLE_TRACE_METADATA = "brokle.trace.metadata"  # Custom metadata
    BROKLE_TRACE_INPUT = "brokle.trace.input"  # Trace-level input
    BROKLE_TRACE_OUTPUT = "brokle.trace.output"  # Trace-level output
    BROKLE_TRACE_PUBLIC = "brokle.trace.public"  # Public visibility flag

    # ========== Brokle Span Management ==========
    BROKLE_SPAN_ID = "brokle.span_id"
    BROKLE_SPAN_TYPE = "brokle.span.type"  # generation/span/event
    BROKLE_SPAN_NAME = "brokle.span_name"
    BROKLE_PARENT_SPAN_ID = "brokle.parent_span_id"
    BROKLE_SPAN_LEVEL = "brokle.span.level"  # DEBUG/DEFAULT/WARNING/ERROR

    # ========== Brokle Extended Usage Metrics ==========
    BROKLE_USAGE_TOTAL_TOKENS = "brokle.usage.total_tokens"  # Convenience metric
    BROKLE_USAGE_LATENCY_MS = "brokle.usage.latency_ms"  # Response latency

    # Note: brokle.cost.* attributes are set by BACKEND only (calculated from usage + model pricing)
    # SDKs should NOT set cost attributes - backend calculates costs server-side

    # ========== Brokle Prompt Management ==========
    BROKLE_PROMPT_ID = "brokle.prompt.id"
    BROKLE_PROMPT_NAME = "brokle.prompt.name"
    BROKLE_PROMPT_VERSION = "brokle.prompt.version"

    # ========== Brokle Quality Scores ==========
    BROKLE_SCORE_NAME = "brokle.score.name"
    BROKLE_SCORE_VALUE = "brokle.score.value"
    BROKLE_SCORE_DATA_TYPE = "brokle.score.data_type"  # numeric/boolean/categorical
    BROKLE_SCORE_COMMENT = "brokle.score.comment"

    # ========== Brokle Intelligent Routing ==========
    BROKLE_ROUTING_STRATEGY = "brokle.routing.strategy"
    BROKLE_ROUTING_PROVIDER_SELECTED = "brokle.routing.provider_selected"
    BROKLE_ROUTING_MODEL_SELECTED = "brokle.routing.model_selected"
    BROKLE_ROUTING_FALLBACK_COUNT = "brokle.routing.fallback_count"
    BROKLE_ROUTING_CACHE_HIT = "brokle.routing.cache_hit"

    # ========== Brokle Internal Flags ==========
    BROKLE_STREAMING = "brokle.streaming"  # Streaming response flag
    BROKLE_CACHED = "brokle.cached"  # Response from cache
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


class SpanType:
    """Span type constants for brokle.span_type attribute."""
    GENERATION = "generation"  # LLM generation (chat, completion)
    SPAN = "span"  # Generic span
    EVENT = "event"  # Point-in-time event
    TOOL = "tool"  # Tool/function call
    RETRIEVAL = "retrieval"  # RAG retrieval operation
    EMBEDDING = "embedding"  # Embedding generation


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


# Convenience aliases for common usage
Attrs = BrokleOtelSpanAttributes