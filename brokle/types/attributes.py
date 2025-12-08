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
    GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
    GEN_AI_OPERATION_NAME = "gen_ai.operation.name"

    # ========== GenAI Request Parameters (OTEL Standard) ==========
    GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
    GEN_AI_REQUEST_TOP_K = "gen_ai.request.top_k"
    GEN_AI_REQUEST_FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
    GEN_AI_REQUEST_PRESENCE_PENALTY = "gen_ai.request.presence_penalty"
    GEN_AI_REQUEST_STOP_SEQUENCES = "gen_ai.request.stop_sequences"
    GEN_AI_REQUEST_USER = "gen_ai.request.user"

    # ========== GenAI Response Metadata (OTEL Standard) ==========
    GEN_AI_RESPONSE_ID = "gen_ai.response.id"
    GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
    GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
    GEN_AI_RESPONSE_FINISH_REASON = "gen_ai.response.finish_reason"

    # ========== GenAI Streaming Metrics (OTEL GenAI Extensions) ==========
    GEN_AI_RESPONSE_TTFT = "gen_ai.response.time_to_first_token"
    GEN_AI_RESPONSE_ITL = "gen_ai.response.inter_token_latency"
    GEN_AI_RESPONSE_DURATION = "gen_ai.response.duration"

    # ========== GenAI Messages (OTEL 1.28+ JSON format) ==========
    GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
    GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"
    GEN_AI_SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"

    # ========== GenAI Usage (OTEL Standard - Optional) ==========
    GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

    # ========== GenAI Extended Usage (Cache, Audio, Multi-modal) ==========
    GEN_AI_USAGE_INPUT_TOKENS_CACHE_READ = "gen_ai.usage.input_tokens.cache_read"
    GEN_AI_USAGE_INPUT_TOKENS_CACHE_CREATION = (
        "gen_ai.usage.input_tokens.cache_creation"
    )
    GEN_AI_USAGE_INPUT_AUDIO_TOKENS = "gen_ai.usage.input_audio_tokens"
    GEN_AI_USAGE_OUTPUT_AUDIO_TOKENS = "gen_ai.usage.output_audio_tokens"
    GEN_AI_USAGE_REASONING_TOKENS = "gen_ai.usage.reasoning_tokens"
    GEN_AI_USAGE_IMAGE_TOKENS = "gen_ai.usage.image_tokens"
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
    SESSION_ID = "session.id"

    # ========== OpenInference Generic Input/Output (Industry Standard) ==========
    INPUT_VALUE = "input.value"
    INPUT_MIME_TYPE = "input.mime_type"
    OUTPUT_VALUE = "output.value"
    OUTPUT_MIME_TYPE = "output.mime_type"

    # ========== Brokle Trace Management ==========
    BROKLE_TRACE_ID = "brokle.trace_id"
    BROKLE_TRACE_NAME = "brokle.trace.name"
    BROKLE_TRACE_TAGS = "brokle.trace.tags"
    BROKLE_TRACE_METADATA = "brokle.trace.metadata"
    BROKLE_TRACE_PUBLIC = "brokle.trace.public"

    # ========== Brokle Span Management ==========
    BROKLE_SPAN_ID = "brokle.span_id"
    BROKLE_SPAN_TYPE = "brokle.span.type"
    BROKLE_SPAN_NAME = "brokle.span_name"
    BROKLE_PARENT_SPAN_ID = "brokle.parent_span_id"
    BROKLE_SPAN_LEVEL = "brokle.span.level"
    BROKLE_SPAN_VERSION = "brokle.span.version"

    # ========== Brokle Extended Usage Metrics ==========
    BROKLE_USAGE_TOTAL_TOKENS = "brokle.usage.total_tokens"
    BROKLE_USAGE_LATENCY_MS = "brokle.usage.latency_ms"

    # ========== Brokle Prompt Management ==========
    BROKLE_PROMPT_ID = "brokle.prompt.id"
    BROKLE_PROMPT_NAME = "brokle.prompt.name"
    BROKLE_PROMPT_VERSION = "brokle.prompt.version"

    # ========== Brokle Quality Scores ==========
    BROKLE_SCORE_NAME = "brokle.score.name"
    BROKLE_SCORE_VALUE = "brokle.score.value"
    BROKLE_SCORE_DATA_TYPE = "brokle.score.data_type"
    BROKLE_SCORE_COMMENT = "brokle.score.comment"

    # ========== Brokle Internal Flags ==========
    BROKLE_STREAMING = "brokle.streaming"
    BROKLE_PROJECT_ID = "brokle.project_id"
    BROKLE_ENVIRONMENT = "brokle.environment"
    BROKLE_VERSION = "brokle.version"
    BROKLE_RELEASE = "brokle.release"

    # ========== Filterable Metadata (Root Level for Querying) ==========
    USER_ID = "user.id"
    TRACE_NAME = "trace_name"
    TAGS = "tags"
    METADATA = "metadata"
    VERSION = "version"
    ENVIRONMENT = "environment"

    # ========== Framework Component Attributes (GenAI Extension) ==========
    GEN_AI_FRAMEWORK_NAME = "gen_ai.framework.name"
    GEN_AI_FRAMEWORK_VERSION = "gen_ai.framework.version"
    GEN_AI_COMPONENT_TYPE = "gen_ai.component.type"
    GEN_AI_AGENT_NAME = "gen_ai.agent.name"
    GEN_AI_AGENT_STRATEGY = "gen_ai.agent.strategy"
    GEN_AI_AGENT_ITERATION_COUNT = "gen_ai.agent.iteration_count"
    GEN_AI_AGENT_MAX_ITERATIONS = "gen_ai.agent.max_iterations"
    GEN_AI_TOOL_NAME = "gen_ai.tool.name"
    GEN_AI_TOOL_DESCRIPTION = "gen_ai.tool.description"
    GEN_AI_TOOL_PARAMETERS = "gen_ai.tool.parameters"
    GEN_AI_RETRIEVER_TYPE = "gen_ai.retriever.type"
    GEN_AI_RETRIEVAL_TOP_K = "gen_ai.retrieval.top_k"
    GEN_AI_RETRIEVAL_SCORE = "gen_ai.retrieval.score"
    GEN_AI_RETRIEVAL_SOURCE = "gen_ai.retrieval.source"
    GEN_AI_MEMORY_TYPE = "gen_ai.memory.type"
    GEN_AI_EXECUTION_PARALLEL_COUNT = "gen_ai.execution.parallel_count"
    GEN_AI_EXECUTION_SEQUENTIAL_ORDER = "gen_ai.execution.sequential_order"


class SpanType:
    """Span type constants for brokle.span_type attribute."""

    GENERATION = "generation"
    SPAN = "span"
    EVENT = "event"
    TOOL = "tool"
    AGENT = "agent"
    CHAIN = "chain"
    RETRIEVAL = "retrieval"
    EMBEDDING = "embedding"
    EVALUATOR = "evaluator"
    GUARDRAIL = "guardrail"
    RERANK = "rerank"
    WORKFLOW = "workflow"


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

    NUMERIC = "numeric"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"


class ComponentType:
    """Standard component types for framework instrumentation."""

    AGENT = "agent"
    CHAIN = "chain"
    RETRIEVER = "retriever"
    MEMORY = "memory"
    TOOL = "tool"
    WORKFLOW = "workflow"
    PLANNER = "planner"


class AgentStrategy:
    """Standard agent strategy types."""

    REACT = "react"
    COT = "cot"
    PLAN_AND_EXECUTE = "plan_and_execute"
    TREE_OF_THOUGHT = "tree_of_thought"
    REFLEXION = "reflexion"
    SELF_ASK = "self_ask"
    ZERO_SHOT = "zero_shot"


class RetrieverType:
    """Standard retriever types for RAG pipelines."""

    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    MULTI_QUERY = "multi_query"
    PARENT_DOCUMENT = "parent_document"


class MemoryType:
    """Standard memory types for AI frameworks."""

    BUFFER = "buffer"
    SUMMARY = "summary"
    CONVERSATION = "conversation"
    ENTITY = "entity"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    VECTOR = "vector"


Attrs = BrokleOtelSpanAttributes


class SchemaURLs:
    """OpenTelemetry semantic convention schema URLs."""

    OTEL_GENAI_1_28 = "https://opentelemetry.io/schemas/1.28.0"
    OTEL_GENAI_1_29 = "https://opentelemetry.io/schemas/1.29.0"
    OPENINFERENCE_1_0 = "https://arize.com/openinference/1.0.0"
    DEFAULT = OTEL_GENAI_1_28
