"""
OpenTelemetry span attributes for Brokle observability.

Provides BrokleOtelSpanAttributes for compatibility with existing code.
"""


class BrokleOtelSpanAttributes:
    """
    Brokle OpenTelemetry span attributes.

    Maintains exact compatibility with existing _client.attributes module.
    """

    # Core attributes
    REQUEST_ID = "brokle.request_id"
    USER_ID = "brokle.user_id"
    SESSION_ID = "brokle.session_id"
    TRACE_ID = "brokle.trace_id"

    # LLM attributes
    LLM_MODEL = "llm.model"
    LLM_PROVIDER = "llm.provider"
    LLM_REQUEST_TYPE = "llm.request_type"
    LLM_RESPONSE_TYPE = "llm.response_type"

    # Model name attributes (for provider compatibility)
    MODEL_NAME = "llm.model"  # Alias for LLM_MODEL
    MODEL_NAME_NORMALIZED = "llm.model.normalized"
    MODEL_NAME_RESPONSE = "llm.model.response"

    # Message and prompt attributes
    MESSAGE_COUNT = "llm.message.count"
    MESSAGE_ROLES = "llm.message.roles"
    SYSTEM_MESSAGE = "llm.message.system"
    PROMPT_LENGTH = "llm.prompt.length"
    PROMPT_COUNT = "llm.prompt.count"

    # Request parameters
    MAX_TOKENS = "llm.max_tokens"  # Alias for LLM_MAX_TOKENS
    TEMPERATURE = "llm.temperature"  # Alias for LLM_TEMPERATURE
    TOP_P = "llm.top_p"  # Alias for LLM_TOP_P
    FREQUENCY_PENALTY = "llm.frequency_penalty"  # Alias for LLM_FREQUENCY_PENALTY
    PRESENCE_PENALTY = "llm.presence_penalty"  # Alias for LLM_PRESENCE_PENALTY
    N_COMPLETIONS = "llm.n_completions"
    STREAM_ENABLED = "llm.stream"

    # Function and tool attributes
    FUNCTION_COUNT = "llm.function.count"
    FUNCTION_NAMES = "llm.function.names"
    FUNCTION_NAME = "brokle.function.name"  # For @observe decorator
    FUNCTION_SIGNATURE = "brokle.function.signature"
    FUNCTION_EXECUTED = "brokle.function.executed"
    FUNCTION_MODULE = "brokle.function.module"

    # Operation and timing attributes
    OPERATION_TYPE = "brokle.operation.type"
    REQUEST_START_TIME = "brokle.request.start_time"
    REQUEST_DURATION = "brokle.request.duration"
    RESPONSE_END_TIME = "brokle.request.end_time"

    # Error attributes
    ERROR_TYPE = "brokle.error.type"
    ERROR_MESSAGE = "brokle.error.message"

    # Output attributes
    OUTPUT = "brokle.output"
    OUTPUT_TYPE = "brokle.output.type"

    # AI Provider attributes
    PROVIDER = "brokle.ai.provider"

    # Workflow attributes
    WORKFLOW_NAME = "brokle.workflow.name"

    # Tags attribute
    TAGS = "brokle.tags"
    FUNCTION_CALL_NAME = "llm.function_call.name"
    TOOL_COUNT = "llm.tool.count"
    TOOL_TYPES = "llm.tool.types"
    TOOL_CALL_NAMES = "llm.tool_call.names"

    # Token usage (aliases for compatibility)
    INPUT_TOKENS = "llm.usage.prompt_tokens"  # Alias for LLM_USAGE_PROMPT_TOKENS
    OUTPUT_TOKENS = (
        "llm.usage.completion_tokens"  # Alias for LLM_USAGE_COMPLETION_TOKENS
    )
    TOTAL_TOKENS = "llm.usage.total_tokens"  # Alias for LLM_USAGE_TOTAL_TOKENS

    # Cost (alias for compatibility)
    COST_USD = "llm.cost.usd"  # Alias for LLM_COST_USD

    # Response attributes
    FINISH_REASON = "llm.finish_reason"
    STOP_REASON = "llm.stop_reason"  # For Anthropic compatibility
    RESPONSE_CONTENT_LENGTH = "llm.response.content_length"

    # Embedding attributes
    EMBEDDING_COUNT = "llm.embedding.count"
    EMBEDDING_DIMENSIONS = "llm.embedding.dimensions"

    # Image attributes
    IMAGE_COUNT = "llm.image.count"

    # Input/Output
    LLM_PROMPTS = "llm.prompts"
    LLM_COMPLETIONS = "llm.completions"
    LLM_INPUT_MESSAGES = "llm.input.messages"
    LLM_OUTPUT_MESSAGES = "llm.output.messages"

    # Token usage
    LLM_USAGE_PROMPT_TOKENS = "llm.usage.prompt_tokens"
    LLM_USAGE_COMPLETION_TOKENS = "llm.usage.completion_tokens"
    LLM_USAGE_TOTAL_TOKENS = "llm.usage.total_tokens"

    # Cost tracking
    LLM_COST_USD = "llm.cost.usd"
    LLM_COST_PER_TOKEN = "llm.cost.per_token"

    # Model parameters
    LLM_TEMPERATURE = "llm.temperature"
    LLM_MAX_TOKENS = "llm.max_tokens"
    LLM_TOP_P = "llm.top_p"
    LLM_FREQUENCY_PENALTY = "llm.frequency_penalty"
    LLM_PRESENCE_PENALTY = "llm.presence_penalty"
    LLM_STOP_SEQUENCES = "llm.stop_sequences"

    # Brokle-specific attributes
    BROKLE_ROUTING_STRATEGY = "brokle.routing.strategy"
    BROKLE_CACHE_STRATEGY = "brokle.cache.strategy"
    BROKLE_CACHE_HIT = "brokle.cache.hit"
    BROKLE_CACHE_KEY = "brokle.cache.key"
    BROKLE_ENVIRONMENT = "brokle.environment"

    # Generation-specific attributes
    BROKLE_GENERATION_MODEL = "brokle.generation.model"
    BROKLE_GENERATION_MODEL_NORMALIZED = "brokle.generation.model_normalized"

    # Workflow and operation attributes
    OPERATION_TYPE = "brokle.operation.type"
    WORKFLOW_NAME = "brokle.workflow.name"
    REQUEST_START_TIME = "brokle.request.start_time"
    REQUEST_DURATION = "brokle.request.duration"
    RESPONSE_END_TIME = "brokle.response.end_time"

    # Quality metrics
    BROKLE_QUALITY_SCORE = "brokle.quality.score"
    BROKLE_QUALITY_METRICS = "brokle.quality.metrics"

    # Performance
    BROKLE_LATENCY_MS = "brokle.latency_ms"
    BROKLE_ROUTING_TIME_MS = "brokle.routing.time_ms"

    # Tags and metadata
    BROKLE_TAGS = "brokle.tags"
    BROKLE_METADATA = "brokle.metadata"

    # Error attributes
    ERROR_TYPE = "error.type"
    ERROR_MESSAGE = "error.message"
    ERROR_STACK = "error.stack"

    # HTTP attributes (for provider calls)
    HTTP_METHOD = "http.method"
    HTTP_URL = "http.url"
    HTTP_STATUS_CODE = "http.status_code"
    HTTP_RESPONSE_SIZE = "http.response.size"

    @classmethod
    def get_all_attributes(cls) -> dict:
        """
        Get all defined attributes as a dictionary.

        Returns:
            Dictionary of attribute names and values
        """
        attributes = {}
        for name in dir(cls):
            if not name.startswith("_") and not callable(getattr(cls, name)):
                value = getattr(cls, name)
                if isinstance(value, str) and not name.startswith("get_"):
                    attributes[name] = value
        return attributes

    @classmethod
    def is_brokle_attribute(cls, attr_name: str) -> bool:
        """
        Check if attribute name is a Brokle attribute.

        Args:
            attr_name: Attribute name to check

        Returns:
            True if it's a Brokle attribute
        """
        return attr_name.startswith("brokle.") or attr_name.startswith("llm.")
