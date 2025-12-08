"""
Observation wrapper classes for enhanced span management.

Provides Langfuse-style observation wrappers that wrap OpenTelemetry spans
with type-specific methods and attributes.

Based on Langfuse SDK patterns:
- completion_start_time tracking for TTFT
- usage_details dict for flexible token tracking
- retroactive trace updates
"""

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from opentelemetry.trace import Span, Status, StatusCode

from ..types import Attrs, SpanType
from .types import ObservationType

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan


@dataclass
class UsageDetails:
    """
    Flexible usage details for token tracking.

    Supports multiple token types including cache, audio, reasoning, etc.
    Following Langfuse's cost_details pattern for maximum flexibility.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Extended token types
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    audio_input_tokens: int = 0
    audio_output_tokens: int = 0
    reasoning_tokens: int = 0
    image_tokens: int = 0

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary, excluding zero values."""
        result = {}
        if self.input_tokens > 0:
            result["input_tokens"] = self.input_tokens
        if self.output_tokens > 0:
            result["output_tokens"] = self.output_tokens
        if self.total_tokens > 0:
            result["total_tokens"] = self.total_tokens
        if self.cache_read_tokens > 0:
            result["cache_read_tokens"] = self.cache_read_tokens
        if self.cache_creation_tokens > 0:
            result["cache_creation_tokens"] = self.cache_creation_tokens
        if self.audio_input_tokens > 0:
            result["audio_input_tokens"] = self.audio_input_tokens
        if self.audio_output_tokens > 0:
            result["audio_output_tokens"] = self.audio_output_tokens
        if self.reasoning_tokens > 0:
            result["reasoning_tokens"] = self.reasoning_tokens
        if self.image_tokens > 0:
            result["image_tokens"] = self.image_tokens
        return result


class BrokleObservation:
    """
    Base observation wrapper for OpenTelemetry spans.

    Provides enhanced methods for updating span attributes and
    managing observation lifecycle.

    Example:
        >>> with client.start_observation("my-operation") as obs:
        ...     obs.update(metadata={"key": "value"})
        ...     obs.set_output(result)
    """

    def __init__(
        self,
        span: Span,
        observation_type: ObservationType = ObservationType.SPAN,
    ):
        """
        Initialize observation wrapper.

        Args:
            span: Underlying OpenTelemetry span
            observation_type: Type of observation
        """
        self._span = span
        self._observation_type = observation_type
        self._start_time = time.perf_counter()
        self._completion_start_time: Optional[float] = None
        self._ended = False

    @property
    def span(self) -> Span:
        """Access the underlying OpenTelemetry span."""
        return self._span

    @property
    def observation_type(self) -> ObservationType:
        """Get the observation type."""
        return self._observation_type

    @property
    def trace_id(self) -> str:
        """Get the trace ID as hex string."""
        ctx = self._span.get_span_context()
        return format(ctx.trace_id, "032x") if ctx else ""

    @property
    def span_id(self) -> str:
        """Get the span ID as hex string."""
        ctx = self._span.get_span_context()
        return format(ctx.span_id, "016x") if ctx else ""

    def update(
        self,
        name: Optional[str] = None,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        level: Optional[str] = None,
        version: Optional[str] = None,
        **extra_attributes,
    ) -> "BrokleObservation":
        """
        Update observation attributes.

        Args:
            name: Update span name
            input: Set input value
            output: Set output value
            metadata: Update metadata (merged with existing)
            level: Set span level (DEBUG, DEFAULT, WARNING, ERROR)
            version: Set version
            **extra_attributes: Additional span attributes

        Returns:
            Self for chaining
        """
        if self._ended:
            return self

        if name:
            # Note: OpenTelemetry spans don't support name updates after creation
            # Store as attribute for backend to use
            self._span.set_attribute(Attrs.BROKLE_SPAN_NAME, name)

        if input is not None:
            input_str = self._serialize(input)
            self._span.set_attribute(Attrs.INPUT_VALUE, input_str)
            self._span.set_attribute(Attrs.INPUT_MIME_TYPE, "application/json")

        if output is not None:
            output_str = self._serialize(output)
            self._span.set_attribute(Attrs.OUTPUT_VALUE, output_str)
            self._span.set_attribute(Attrs.OUTPUT_MIME_TYPE, "application/json")

        if metadata:
            self._span.set_attribute(Attrs.BROKLE_TRACE_METADATA, json.dumps(metadata))

        if level:
            self._span.set_attribute(Attrs.BROKLE_SPAN_LEVEL, level)

        if version:
            self._span.set_attribute(Attrs.BROKLE_VERSION, version)

        for key, value in extra_attributes.items():
            self._span.set_attribute(key, value)

        return self

    def set_input(self, input: Any) -> "BrokleObservation":
        """Set input value."""
        return self.update(input=input)

    def set_output(self, output: Any) -> "BrokleObservation":
        """Set output value."""
        return self.update(output=output)

    def set_error(self, error: Exception) -> "BrokleObservation":
        """
        Record an error on this observation.

        Args:
            error: Exception to record

        Returns:
            Self for chaining
        """
        if not self._ended:
            self._span.set_status(Status(StatusCode.ERROR, str(error)))
            self._span.record_exception(error)
            self._span.set_attribute(Attrs.BROKLE_SPAN_LEVEL, "ERROR")
        return self

    def update_trace(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "BrokleObservation":
        """
        Update trace-level attributes (propagated to all spans in trace).

        Following Langfuse pattern of retroactive trace updates.

        Args:
            session_id: Session grouping identifier
            user_id: User identifier
            tags: Trace tags (additive - merged with existing)
            metadata: Trace metadata (merged with existing)

        Returns:
            Self for chaining
        """
        if self._ended:
            return self

        if session_id:
            self._span.set_attribute(Attrs.SESSION_ID, session_id)
        if user_id:
            self._span.set_attribute(Attrs.USER_ID, user_id)
        if tags:
            self._span.set_attribute(Attrs.BROKLE_TRACE_TAGS, json.dumps(tags))
        if metadata:
            self._span.set_attribute(Attrs.BROKLE_TRACE_METADATA, json.dumps(metadata))

        return self

    def end(self) -> None:
        """End the observation (span)."""
        if not self._ended:
            self._span.end()
            self._ended = True

    def __enter__(self) -> "BrokleObservation":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_val:
            self.set_error(exc_val)
        self.end()
        return False

    @staticmethod
    def _serialize(value: Any) -> str:
        """Serialize value to JSON string."""
        try:
            if value is None:
                return "null"
            if isinstance(value, str):
                return json.dumps(value)
            if hasattr(value, "model_dump"):
                return json.dumps(value.model_dump(exclude_none=True))
            if hasattr(value, "__dataclass_fields__"):
                import dataclasses

                return json.dumps(dataclasses.asdict(value))
            return json.dumps(value, default=str)
        except Exception:
            return json.dumps(str(value))


class BrokleGeneration(BrokleObservation):
    """
    Generation observation for LLM calls.

    Provides additional methods for tracking:
    - Model and parameters
    - Token usage
    - Completion start time (for TTFT)
    - Cost calculation hints

    Example:
        >>> with client.start_generation(
        ...     name="chat",
        ...     model="gpt-4",
        ... ) as gen:
        ...     response = openai.chat.completions.create(...)
        ...     gen.mark_completion_start()  # Mark first token
        ...     gen.set_usage(input_tokens=100, output_tokens=50)
        ...     gen.set_output(response.choices[0].message.content)
    """

    def __init__(
        self,
        span: Span,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        super().__init__(span, ObservationType.GENERATION)
        self._model = model
        self._provider = provider

        # Set generation-specific attributes
        if model:
            span.set_attribute(Attrs.GEN_AI_REQUEST_MODEL, model)
        if provider:
            span.set_attribute(Attrs.GEN_AI_PROVIDER_NAME, provider)

    def mark_completion_start(self) -> "BrokleGeneration":
        """
        Mark the completion start time (first token received).

        Call this when the first token of a streaming response arrives
        to accurately track time-to-first-token (TTFT).

        Returns:
            Self for chaining
        """
        if self._completion_start_time is None:
            self._completion_start_time = time.perf_counter()

            # Calculate TTFT
            ttft_ms = (self._completion_start_time - self._start_time) * 1000
            self._span.set_attribute(Attrs.GEN_AI_RESPONSE_TTFT, ttft_ms)

        return self

    def set_usage(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: Optional[int] = None,
        **extended_usage,
    ) -> "BrokleGeneration":
        """
        Set token usage for this generation.

        Args:
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens
            total_tokens: Total tokens (calculated if not provided)
            **extended_usage: Extended token types (cache_read_tokens, audio_tokens, etc.)

        Returns:
            Self for chaining
        """
        if self._ended:
            return self

        # Calculate total if not provided
        if total_tokens is None:
            total_tokens = input_tokens + output_tokens

        # Set standard usage attributes
        if input_tokens > 0:
            self._span.set_attribute(Attrs.GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
        if output_tokens > 0:
            self._span.set_attribute(Attrs.GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)
        if total_tokens > 0:
            self._span.set_attribute(Attrs.BROKLE_USAGE_TOTAL_TOKENS, total_tokens)

        # Set extended usage attributes
        if "cache_read_tokens" in extended_usage:
            self._span.set_attribute(
                Attrs.GEN_AI_USAGE_INPUT_TOKENS_CACHE_READ,
                extended_usage["cache_read_tokens"],
            )
        if "cache_creation_tokens" in extended_usage:
            self._span.set_attribute(
                Attrs.GEN_AI_USAGE_INPUT_TOKENS_CACHE_CREATION,
                extended_usage["cache_creation_tokens"],
            )
        if "audio_input_tokens" in extended_usage:
            self._span.set_attribute(
                Attrs.GEN_AI_USAGE_INPUT_AUDIO_TOKENS,
                extended_usage["audio_input_tokens"],
            )
        if "audio_output_tokens" in extended_usage:
            self._span.set_attribute(
                Attrs.GEN_AI_USAGE_OUTPUT_AUDIO_TOKENS,
                extended_usage["audio_output_tokens"],
            )
        if "reasoning_tokens" in extended_usage:
            self._span.set_attribute(
                Attrs.GEN_AI_USAGE_REASONING_TOKENS, extended_usage["reasoning_tokens"]
            )

        return self

    def set_model_parameters(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        **extra_params,
    ) -> "BrokleGeneration":
        """
        Set model parameters for this generation.

        Args:
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            **extra_params: Additional model-specific parameters

        Returns:
            Self for chaining
        """
        if self._ended:
            return self

        if temperature is not None:
            self._span.set_attribute(Attrs.GEN_AI_REQUEST_TEMPERATURE, temperature)
        if max_tokens is not None:
            self._span.set_attribute(Attrs.GEN_AI_REQUEST_MAX_TOKENS, max_tokens)
        if top_p is not None:
            self._span.set_attribute(Attrs.GEN_AI_REQUEST_TOP_P, top_p)
        if frequency_penalty is not None:
            self._span.set_attribute(
                Attrs.GEN_AI_REQUEST_FREQUENCY_PENALTY, frequency_penalty
            )
        if presence_penalty is not None:
            self._span.set_attribute(
                Attrs.GEN_AI_REQUEST_PRESENCE_PENALTY, presence_penalty
            )

        return self

    def set_finish_reason(self, reason: str) -> "BrokleGeneration":
        """Set the finish/stop reason."""
        if not self._ended:
            self._span.set_attribute(Attrs.GEN_AI_RESPONSE_FINISH_REASON, reason)
        return self


class BrokleEvent(BrokleObservation):
    """
    Event observation for point-in-time occurrences.

    Events have no meaningful duration - they mark specific moments.

    Example:
        >>> with client.start_event("user-action") as event:
        ...     event.update(metadata={"action": "click", "target": "submit"})
    """

    def __init__(self, span: Span):
        super().__init__(span, ObservationType.EVENT)


class BrokleAgent(BrokleObservation):
    """
    Agent observation for autonomous AI agent operations.

    Tracks agent reasoning, tool usage, and multi-step operations.

    Example:
        >>> with client.start_agent("research-agent") as agent:
        ...     agent.update(metadata={"objective": "Find information about X"})
        ...     # Agent loop
        ...     agent.set_output(final_result)
    """

    def __init__(self, span: Span):
        super().__init__(span, ObservationType.AGENT)

    def set_tools(self, tools: List[str]) -> "BrokleAgent":
        """Set available tools for this agent."""
        if not self._ended:
            self._span.set_attribute("brokle.agent.tools", json.dumps(tools))
        return self

    def set_iterations(self, count: int) -> "BrokleAgent":
        """Set the number of agent iterations/steps."""
        if not self._ended:
            self._span.set_attribute("brokle.agent.iterations", count)
        return self


class BrokleTool(BrokleObservation):
    """
    Tool observation for function/tool calls.

    Tracks tool name, input parameters, and results.

    Example:
        >>> with client.start_tool("calculator") as tool:
        ...     tool.set_input({"operation": "add", "a": 1, "b": 2})
        ...     result = calculator.add(1, 2)
        ...     tool.set_output(result)
    """

    def __init__(self, span: Span, tool_name: Optional[str] = None):
        super().__init__(span, ObservationType.TOOL)
        if tool_name:
            span.set_attribute("brokle.tool.name", tool_name)


class BrokleRetrieval(BrokleObservation):
    """
    Retrieval observation for RAG operations.

    Tracks retrieval queries, document counts, and relevance scores.

    Example:
        >>> with client.start_retrieval("vector-search") as retrieval:
        ...     retrieval.set_query(query_text)
        ...     docs = vector_db.search(query_text, k=5)
        ...     retrieval.set_documents(docs)
    """

    def __init__(self, span: Span):
        super().__init__(span, ObservationType.RETRIEVAL)

    def set_query(self, query: str) -> "BrokleRetrieval":
        """Set the retrieval query."""
        if not self._ended:
            self._span.set_attribute("brokle.retrieval.query", query)
        return self

    def set_documents(
        self,
        documents: List[Dict[str, Any]],
        count: Optional[int] = None,
    ) -> "BrokleRetrieval":
        """
        Set retrieved documents.

        Args:
            documents: List of retrieved documents
            count: Document count (calculated from list if not provided)

        Returns:
            Self for chaining
        """
        if not self._ended:
            doc_count = count if count is not None else len(documents)
            self._span.set_attribute("brokle.retrieval.document_count", doc_count)
            # Store first few documents for debugging (limit to avoid huge spans)
            preview = documents[:5] if len(documents) > 5 else documents
            self._span.set_attribute(
                "brokle.retrieval.documents", json.dumps(preview, default=str)
            )
        return self

    def set_top_k(self, k: int) -> "BrokleRetrieval":
        """Set the top-k parameter used for retrieval."""
        if not self._ended:
            self._span.set_attribute("brokle.retrieval.top_k", k)
        return self
