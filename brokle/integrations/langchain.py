"""
LangChain integration for automatic Brokle tracing.

Provides callback handlers that automatically create OpenTelemetry spans
for LangChain operations (LLM calls, chains, tools, retrievers).

Example:
    from langchain.chains import LLMChain
    from langchain.chat_models import ChatOpenAI
    from brokle.integrations import BrokleLangChainCallback

    # Create Brokle callback
    callback = BrokleLangChainCallback(
        user_id="user-123",
        session_id="session-456"
    )

    # Use with LangChain
    llm = ChatOpenAI(callbacks=[callback])
    chain = LLMChain(llm=llm, callbacks=[callback])
    result = chain.run("Hello")  # Automatically traced
"""

import json
import time
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

try:
    from langchain.callbacks.base import BaseCallbackHandler
    from langchain.schema import LLMResult
except ImportError:
    raise ImportError(
        "LangChain integration requires the 'langchain' package. "
        "Install with: pip install langchain langchain-core"
    )

from opentelemetry.trace import Status, StatusCode

from ..client import get_client
from ..types import Attrs, LLMProvider, OperationType, SpanType


class BrokleLangChainCallback(BaseCallbackHandler):
    """
    LangChain callback handler for automatic Brokle tracing.

    This callback handler automatically creates OpenTelemetry spans for
    LangChain operations, following GenAI 1.28+ semantic conventions.

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
        **kwargs,
    ):
        """
        Initialize LangChain callback handler.

        Args:
            user_id: User identifier
            session_id: Session identifier
            metadata: Custom metadata
            tags: Categorization tags
            **kwargs: Additional arguments passed to BaseCallbackHandler
        """
        super().__init__(**kwargs)

        self.user_id = user_id
        self.session_id = session_id
        self.metadata = metadata or {}
        self.tags = tags or []

        # Get Brokle client
        self._client = get_client()

        # Track active spans by run_id
        self._spans: Dict[UUID, Any] = {}
        self._span_start_times: Dict[UUID, float] = {}

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

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Called when LLM starts running.

        Creates a generation span with GenAI attributes.
        """
        # Extract model and provider information
        model = self._extract_model(serialized, kwargs)
        provider = self._extract_provider(serialized, kwargs)
        operation = "chat" if "chat" in str(serialized).lower() else "text_completion"

        # Build span attributes
        attrs = self._get_common_attributes()
        attrs.update(
            {
                Attrs.BROKLE_SPAN_TYPE: SpanType.GENERATION,
                Attrs.GEN_AI_OPERATION_NAME: operation,
            }
        )

        if model:
            attrs[Attrs.GEN_AI_REQUEST_MODEL] = model

        if provider:
            attrs[Attrs.GEN_AI_PROVIDER_NAME] = provider

        # Add prompts as input messages
        if prompts:
            input_messages = [{"role": "user", "content": prompt} for prompt in prompts]
            attrs[Attrs.GEN_AI_INPUT_MESSAGES] = json.dumps(input_messages)

        # Extract model parameters from invocation_params
        invocation_params = kwargs.get("invocation_params", {})
        if invocation_params:
            if "temperature" in invocation_params:
                attrs[Attrs.GEN_AI_REQUEST_TEMPERATURE] = invocation_params[
                    "temperature"
                ]
            if "max_tokens" in invocation_params:
                attrs[Attrs.GEN_AI_REQUEST_MAX_TOKENS] = invocation_params["max_tokens"]
            if "top_p" in invocation_params:
                attrs[Attrs.GEN_AI_REQUEST_TOP_P] = invocation_params["top_p"]

        # Create span name following OTEL pattern
        span_name = f"{operation} {model}" if model else operation

        # Create span
        span = self._client._tracer.start_span(name=span_name, attributes=attrs)
        self._spans[run_id] = span
        self._span_start_times[run_id] = time.time()

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """
        Called when LLM ends running.

        Updates the span with outputs and usage information.
        """
        span = self._spans.get(run_id)
        if not span:
            return

        try:
            # Extract outputs
            if response.generations:
                output_messages = []
                finish_reasons = []

                for generation_list in response.generations:
                    for generation in generation_list:
                        output_messages.append(
                            {
                                "role": "assistant",
                                "content": generation.text,
                            }
                        )

                        # Extract finish reason if available
                        if (
                            hasattr(generation, "generation_info")
                            and generation.generation_info
                        ):
                            finish_reason = generation.generation_info.get(
                                "finish_reason"
                            )
                            if finish_reason:
                                finish_reasons.append(finish_reason)

                # Set output messages
                if output_messages:
                    span.set_attribute(
                        Attrs.GEN_AI_OUTPUT_MESSAGES, json.dumps(output_messages)
                    )

                # Set finish reasons
                if finish_reasons:
                    span.set_attribute(
                        Attrs.GEN_AI_RESPONSE_FINISH_REASONS, finish_reasons
                    )

            # Extract usage information
            if hasattr(response, "llm_output") and response.llm_output:
                llm_output = response.llm_output

                # Token usage
                if "token_usage" in llm_output:
                    token_usage = llm_output["token_usage"]
                    if "prompt_tokens" in token_usage:
                        span.set_attribute(
                            Attrs.GEN_AI_USAGE_INPUT_TOKENS,
                            token_usage["prompt_tokens"],
                        )
                    if "completion_tokens" in token_usage:
                        span.set_attribute(
                            Attrs.GEN_AI_USAGE_OUTPUT_TOKENS,
                            token_usage["completion_tokens"],
                        )
                    if "total_tokens" in token_usage:
                        span.set_attribute(
                            Attrs.BROKLE_USAGE_TOTAL_TOKENS, token_usage["total_tokens"]
                        )

                # Model name from response
                if "model_name" in llm_output:
                    span.set_attribute(
                        Attrs.GEN_AI_RESPONSE_MODEL, llm_output["model_name"]
                    )

            # Calculate latency
            if run_id in self._span_start_times:
                latency_ms = (time.time() - self._span_start_times[run_id]) * 1000
                span.set_attribute(Attrs.BROKLE_USAGE_LATENCY_MS, latency_ms)

            # Mark as successful
            span.set_status(Status(StatusCode.OK))

        finally:
            # End span
            span.end()
            self._spans.pop(run_id, None)
            self._span_start_times.pop(run_id, None)

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """
        Called when LLM errors.

        Records the error in the span.
        """
        span = self._spans.get(run_id)
        if not span:
            return

        try:
            # Record error
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.record_exception(error)

        finally:
            # End span
            span.end()
            self._spans.pop(run_id, None)
            self._span_start_times.pop(run_id, None)

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Called when chain starts running.

        Creates a parent span for the chain.
        """
        # Extract chain name
        chain_name = serialized.get("name", "chain")

        # Build span attributes
        attrs = self._get_common_attributes()
        attrs.update(
            {
                Attrs.BROKLE_SPAN_TYPE: SpanType.SPAN,
                "langchain.chain_type": chain_name,
            }
        )

        # Add inputs
        if inputs:
            attrs["langchain.chain_input"] = json.dumps(inputs, default=str)

        # Create span
        span = self._client._tracer.start_span(
            name=f"chain:{chain_name}", attributes=attrs
        )
        self._spans[run_id] = span
        self._span_start_times[run_id] = time.time()

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """
        Called when chain ends running.

        Updates the span with outputs.
        """
        span = self._spans.get(run_id)
        if not span:
            return

        try:
            # Add outputs
            if outputs:
                span.set_attribute(
                    "langchain.chain_output", json.dumps(outputs, default=str)
                )

            # Calculate latency
            if run_id in self._span_start_times:
                latency_ms = (time.time() - self._span_start_times[run_id]) * 1000
                span.set_attribute(Attrs.BROKLE_USAGE_LATENCY_MS, latency_ms)

            # Mark as successful
            span.set_status(Status(StatusCode.OK))

        finally:
            # End span
            span.end()
            self._spans.pop(run_id, None)
            self._span_start_times.pop(run_id, None)

    def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """
        Called when chain errors.

        Records the error in the span.
        """
        span = self._spans.get(run_id)
        if not span:
            return

        try:
            # Record error
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.record_exception(error)

        finally:
            # End span
            span.end()
            self._spans.pop(run_id, None)
            self._span_start_times.pop(run_id, None)

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Called when tool starts running.

        Creates a tool span.
        """
        # Extract tool name
        tool_name = serialized.get("name", "tool")

        # Build span attributes
        attrs = self._get_common_attributes()
        attrs.update(
            {
                Attrs.BROKLE_SPAN_TYPE: SpanType.TOOL,
                "langchain.tool_name": tool_name,
                "langchain.tool_input": input_str,
            }
        )

        # Create span
        span = self._client._tracer.start_span(
            name=f"tool:{tool_name}", attributes=attrs
        )
        self._spans[run_id] = span
        self._span_start_times[run_id] = time.time()

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """
        Called when tool ends running.

        Updates the span with outputs.
        """
        span = self._spans.get(run_id)
        if not span:
            return

        try:
            # Add output
            span.set_attribute("langchain.tool_output", output)

            # Calculate latency
            if run_id in self._span_start_times:
                latency_ms = (time.time() - self._span_start_times[run_id]) * 1000
                span.set_attribute(Attrs.BROKLE_USAGE_LATENCY_MS, latency_ms)

            # Mark as successful
            span.set_status(Status(StatusCode.OK))

        finally:
            # End span
            span.end()
            self._spans.pop(run_id, None)
            self._span_start_times.pop(run_id, None)

    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """
        Called when tool errors.

        Records the error in the span.
        """
        span = self._spans.get(run_id)
        if not span:
            return

        try:
            # Record error
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.record_exception(error)

        finally:
            # End span
            span.end()
            self._spans.pop(run_id, None)
            self._span_start_times.pop(run_id, None)

    def _extract_model(
        self, serialized: Dict[str, Any], kwargs: Dict[str, Any]
    ) -> Optional[str]:
        """Extract model name from serialized LLM or kwargs."""
        # Try to get from serialized
        if "model_name" in serialized:
            return serialized["model_name"]
        if "model" in serialized:
            return serialized["model"]

        # Try to get from kwargs
        if "invocation_params" in kwargs:
            params = kwargs["invocation_params"]
            if "model_name" in params:
                return params["model_name"]
            if "model" in params:
                return params["model"]

        return None

    def _extract_provider(
        self, serialized: Dict[str, Any], kwargs: Dict[str, Any]
    ) -> Optional[str]:
        """Extract provider name from serialized LLM."""
        # Try to infer from class name
        class_name = serialized.get("id", [""])[0] if "id" in serialized else ""

        if "openai" in class_name.lower():
            return LLMProvider.OPENAI
        elif "anthropic" in class_name.lower():
            return LLMProvider.ANTHROPIC
        elif "google" in class_name.lower():
            return LLMProvider.GOOGLE
        elif "cohere" in class_name.lower():
            return LLMProvider.COHERE

        return None
