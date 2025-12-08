"""
LlamaIndex integration for automatic Brokle tracing.

Provides global handler registration for automatic OpenTelemetry span
creation for LlamaIndex operations (queries, retrievals, LLM calls).

Example:
    from llama_index import VectorStoreIndex, SimpleDirectoryReader
    from brokle.integrations import set_global_handler

    # Set Brokle as global handler
    set_global_handler("brokle", user_id="user-123")

    # Use LlamaIndex normally
    documents = SimpleDirectoryReader("data/").load_data()
    index = VectorStoreIndex.from_documents(documents)
    response = index.as_query_engine().query("Question")  # Automatically traced
"""

import json
import time
from typing import Any, Dict, List, Optional
from uuid import UUID

try:
    from llama_index.core.callbacks import CallbackManager
    from llama_index.core.callbacks.base_handler import BaseCallbackHandler
    from llama_index.core.callbacks.schema import CBEventType, EventPayload
except ImportError:
    # Try legacy imports
    try:
        from llama_index.callbacks import CallbackManager
        from llama_index.callbacks.base_handler import BaseCallbackHandler
        from llama_index.callbacks.schema import CBEventType, EventPayload
    except ImportError:
        raise ImportError(
            "LlamaIndex integration requires the 'llama-index' package. "
            "Install with: pip install llama-index"
        )

from opentelemetry.trace import Status, StatusCode

from ..client import get_client
from ..types import Attrs, LLMProvider, OperationType, SpanType


class BrokleLlamaIndexHandler(BaseCallbackHandler):
    """
    LlamaIndex callback handler for automatic Brokle tracing.

    This handler automatically creates OpenTelemetry spans for LlamaIndex
    operations, following GenAI 1.28+ semantic conventions.

    Attributes:
        user_id: Optional user identifier for tracking
        session_id: Optional session identifier for grouping
        metadata: Optional custom metadata to attach to all spans
        tags: Optional tags for categorization
    """

    def __init__(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        event_starts_to_ignore: Optional[List[CBEventType]] = None,
        event_ends_to_ignore: Optional[List[CBEventType]] = None,
    ):
        """
        Initialize LlamaIndex callback handler.

        Args:
            user_id: User identifier
            session_id: Session identifier
            metadata: Custom metadata
            tags: Categorization tags
            event_starts_to_ignore: Event types to ignore on start
            event_ends_to_ignore: Event types to ignore on end
        """
        super().__init__(
            event_starts_to_ignore=event_starts_to_ignore or [],
            event_ends_to_ignore=event_ends_to_ignore or [],
        )

        self.user_id = user_id
        self.session_id = session_id
        self.metadata = metadata or {}
        self.tags = tags or []

        # Get Brokle client
        self._client = get_client()

        # Track active spans by event_id
        self._spans: Dict[str, Any] = {}
        self._span_start_times: Dict[str, float] = {}

    def _get_common_attributes(self) -> Dict[str, Any]:
        """Get common attributes to attach to all spans."""
        attrs = {}

        if self.user_id:
            attrs[Attrs.USER_ID] = self.user_id
            attrs[Attrs.GEN_AI_REQUEST_USER] = self.user_id

        if self.session_id:
            attrs[Attrs.SESSION_ID] = self.session_id

        if self.tags:
            attrs[Attrs.TAGS] = json.dumps(self.tags)

        if self.metadata:
            attrs[Attrs.METADATA] = json.dumps(self.metadata)

        return attrs

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """
        Called when an event starts.

        Creates appropriate span based on event type.

        Args:
            event_type: Type of event (LLM, QUERY, RETRIEVE, etc.)
            payload: Event payload data
            event_id: Unique event identifier
            parent_id: Parent event identifier
            **kwargs: Additional arguments

        Returns:
            Event ID
        """
        payload = payload or {}

        # Build base attributes
        attrs = self._get_common_attributes()
        attrs["llamaindex.event_type"] = str(event_type)
        attrs["llamaindex.event_id"] = event_id

        # Determine span name and attributes based on event type
        if event_type == CBEventType.LLM:
            span_name, span_attrs = self._handle_llm_start(payload)
            attrs.update(span_attrs)

        elif event_type == CBEventType.QUERY:
            span_name = "query"
            attrs[Attrs.BROKLE_SPAN_TYPE] = SpanType.SPAN
            if EventPayload.QUERY_STR in payload:
                attrs["llamaindex.query"] = payload[EventPayload.QUERY_STR]

        elif event_type == CBEventType.RETRIEVE:
            span_name = "retrieve"
            attrs[Attrs.BROKLE_SPAN_TYPE] = SpanType.RETRIEVAL
            if EventPayload.QUERY_STR in payload:
                attrs["llamaindex.query"] = payload[EventPayload.QUERY_STR]

        elif event_type == CBEventType.EMBEDDING:
            span_name = "embedding"
            attrs[Attrs.BROKLE_SPAN_TYPE] = SpanType.EMBEDDING
            attrs[Attrs.GEN_AI_OPERATION_NAME] = OperationType.EMBEDDINGS

        elif event_type == CBEventType.NODE_PARSING:
            span_name = "node_parsing"
            attrs[Attrs.BROKLE_SPAN_TYPE] = SpanType.SPAN

        elif event_type == CBEventType.CHUNKING:
            span_name = "chunking"
            attrs[Attrs.BROKLE_SPAN_TYPE] = SpanType.SPAN

        else:
            # Generic span for other event types
            span_name = str(event_type).lower()
            attrs[Attrs.BROKLE_SPAN_TYPE] = SpanType.SPAN

        # Create span
        span = self._client._tracer.start_span(name=span_name, attributes=attrs)
        self._spans[event_id] = span
        self._span_start_times[event_id] = time.time()

        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """
        Called when an event ends.

        Updates the span with results and ends it.

        Args:
            event_type: Type of event
            payload: Event payload data
            event_id: Unique event identifier
            **kwargs: Additional arguments
        """
        span = self._spans.get(event_id)
        if not span:
            return

        payload = payload or {}

        try:
            # Handle event type-specific outputs
            if event_type == CBEventType.LLM:
                self._handle_llm_end(span, payload)

            elif event_type == CBEventType.QUERY:
                if EventPayload.RESPONSE in payload:
                    response = payload[EventPayload.RESPONSE]
                    span.set_attribute("llamaindex.response", str(response))

            elif event_type == CBEventType.RETRIEVE:
                if EventPayload.NODES in payload:
                    nodes = payload[EventPayload.NODES]
                    span.set_attribute("llamaindex.retrieved_nodes", len(nodes))
                    # Store node scores if available
                    if nodes and hasattr(nodes[0], "score"):
                        scores = [
                            node.score for node in nodes if hasattr(node, "score")
                        ]
                        if scores:
                            span.set_attribute("llamaindex.node_scores", scores)

            elif event_type == CBEventType.EMBEDDING:
                if EventPayload.CHUNKS in payload:
                    chunks = payload[EventPayload.CHUNKS]
                    span.set_attribute("llamaindex.chunk_count", len(chunks))

            # Calculate latency
            if event_id in self._span_start_times:
                latency_ms = (time.time() - self._span_start_times[event_id]) * 1000
                span.set_attribute(Attrs.BROKLE_USAGE_LATENCY_MS, latency_ms)

            # Mark as successful
            span.set_status(Status(StatusCode.OK))

        except Exception as e:
            # Record any errors during processing
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)

        finally:
            # End span
            span.end()
            self._spans.pop(event_id, None)
            self._span_start_times.pop(event_id, None)

    def _handle_llm_start(self, payload: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """
        Handle LLM event start.

        Returns:
            Tuple of (span_name, attributes)
        """
        attrs = {
            Attrs.BROKLE_SPAN_TYPE: SpanType.GENERATION,
            Attrs.GEN_AI_OPERATION_NAME: OperationType.CHAT,
        }

        # Extract model information
        if EventPayload.SERIALIZED in payload:
            serialized = payload[EventPayload.SERIALIZED]

            # Model name
            if "model" in serialized:
                model = serialized["model"]
                attrs[Attrs.GEN_AI_REQUEST_MODEL] = model
            elif "model_name" in serialized:
                model = serialized["model_name"]
                attrs[Attrs.GEN_AI_REQUEST_MODEL] = model
            else:
                model = "unknown"

            # Provider (infer from class name)
            if "class_name" in serialized:
                class_name = serialized["class_name"].lower()
                if "openai" in class_name:
                    attrs[Attrs.GEN_AI_PROVIDER_NAME] = LLMProvider.OPENAI
                elif "anthropic" in class_name:
                    attrs[Attrs.GEN_AI_PROVIDER_NAME] = LLMProvider.ANTHROPIC
                elif "google" in class_name or "gemini" in class_name:
                    attrs[Attrs.GEN_AI_PROVIDER_NAME] = LLMProvider.GOOGLE

            # Model parameters
            if "temperature" in serialized:
                attrs[Attrs.GEN_AI_REQUEST_TEMPERATURE] = serialized["temperature"]
            if "max_tokens" in serialized:
                attrs[Attrs.GEN_AI_REQUEST_MAX_TOKENS] = serialized["max_tokens"]

        # Extract prompts/messages
        if EventPayload.PROMPT in payload:
            prompt = payload[EventPayload.PROMPT]
            input_messages = [{"role": "user", "content": prompt}]
            attrs[Attrs.GEN_AI_INPUT_MESSAGES] = json.dumps(input_messages)
        elif EventPayload.MESSAGES in payload:
            messages = payload[EventPayload.MESSAGES]
            attrs[Attrs.GEN_AI_INPUT_MESSAGES] = json.dumps(messages)

        # Span name
        model = attrs.get(Attrs.GEN_AI_REQUEST_MODEL, "llm")
        operation = attrs.get(Attrs.GEN_AI_OPERATION_NAME, "chat")
        span_name = f"{operation} {model}"

        return span_name, attrs

    def _handle_llm_end(self, span: Any, payload: Dict[str, Any]) -> None:
        """Handle LLM event end."""
        # Extract response
        if EventPayload.RESPONSE in payload:
            response = payload[EventPayload.RESPONSE]

            # Convert response to output messages
            if hasattr(response, "message"):
                # Chat response
                output_messages = [
                    {
                        "role": "assistant",
                        "content": str(response.message.content),
                    }
                ]
                span.set_attribute(
                    Attrs.GEN_AI_OUTPUT_MESSAGES, json.dumps(output_messages)
                )
            elif hasattr(response, "text"):
                # Text completion response
                output_messages = [
                    {
                        "role": "assistant",
                        "content": response.text,
                    }
                ]
                span.set_attribute(
                    Attrs.GEN_AI_OUTPUT_MESSAGES, json.dumps(output_messages)
                )

            # Extract additional metadata
            if hasattr(response, "raw"):
                raw = response.raw
                if hasattr(raw, "model"):
                    span.set_attribute(Attrs.GEN_AI_RESPONSE_MODEL, raw.model)
                if hasattr(raw, "id"):
                    span.set_attribute(Attrs.GEN_AI_RESPONSE_ID, raw.id)

                # Usage information
                if hasattr(raw, "usage"):
                    usage = raw.usage
                    if hasattr(usage, "prompt_tokens"):
                        span.set_attribute(
                            Attrs.GEN_AI_USAGE_INPUT_TOKENS, usage.prompt_tokens
                        )
                    if hasattr(usage, "completion_tokens"):
                        span.set_attribute(
                            Attrs.GEN_AI_USAGE_OUTPUT_TOKENS, usage.completion_tokens
                        )
                    if hasattr(usage, "total_tokens"):
                        span.set_attribute(
                            Attrs.BROKLE_USAGE_TOTAL_TOKENS, usage.total_tokens
                        )

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Start a new trace (no-op for compatibility)."""
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """End a trace (no-op for compatibility)."""
        pass


def set_global_handler(
    handler_name: str = "brokle",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> BrokleLlamaIndexHandler:
    """
    Set Brokle as the global LlamaIndex callback handler.

    This function creates a BrokleLlamaIndexHandler and registers it
    as the global handler for all LlamaIndex operations.

    Args:
        handler_name: Handler name (must be "brokle")
        user_id: User identifier for tracking
        session_id: Session identifier for grouping
        metadata: Custom metadata
        tags: Categorization tags

    Returns:
        The created BrokleLlamaIndexHandler instance

    Example:
        >>> from brokle.integrations import set_global_handler
        >>> set_global_handler("brokle", user_id="user-123")
        >>> # Now all LlamaIndex operations are automatically traced

    Raises:
        ValueError: If handler_name is not "brokle"
    """
    if handler_name != "brokle":
        raise ValueError(
            f"Invalid handler name: {handler_name}. " "Use 'brokle' for Brokle handler."
        )

    # Create handler
    handler = BrokleLlamaIndexHandler(
        user_id=user_id,
        session_id=session_id,
        metadata=metadata,
        tags=tags,
    )

    # Set as global handler
    try:
        # Try new API
        from llama_index.core import Settings

        Settings.callback_manager = CallbackManager([handler])
    except (ImportError, AttributeError):
        # Try legacy API
        try:
            import llama_index

            llama_index.global_handler = handler
        except (ImportError, AttributeError):
            # Manual registration as fallback
            callback_manager = CallbackManager([handler])
            # Store for user to use manually if needed
            handler._callback_manager = callback_manager

    return handler
