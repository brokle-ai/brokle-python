"""
LangChain Integration - Brokle SDK

This module provides comprehensive LangChain integration through callback handlers,
following the industry-standard pattern for framework integration. Supports both
LangChain and LangChain Community packages with version compatibility.

Key Features:
- Comprehensive callback handler for all LangChain operations
- Multi-version compatibility (LangChain 0.1+ and Community)
- Automatic chain and agent tracing
- Token usage and cost tracking
- Error handling and debugging support
- Async/await support throughout

Usage:
    from brokle.langchain import BrokleCallbackHandler

    # Add to any LangChain operation
    handler = BrokleCallbackHandler()

    # With LLM chains
    llm = OpenAI(callbacks=[handler])
    response = llm("What is AI?")

    # With agents
    agent = initialize_agent(
        tools, llm, callbacks=[handler]
    )

    # With async operations
    handler = BrokleCallbackHandler(session_id="session123")
    async_chain.arun(input_text, callbacks=[handler])
"""

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Union

try:
    from langchain.callbacks.base import BaseCallbackHandler
    from langchain.schema import AgentAction, AgentFinish, LLMResult

    HAS_LANGCHAIN = True
except ImportError:
    try:
        from langchain_core.callbacks.base import BaseCallbackHandler
        from langchain_core.schema import AgentAction, AgentFinish, LLMResult

        HAS_LANGCHAIN = True
    except ImportError:
        # Stub implementation if LangChain not available
        BaseCallbackHandler = object
        AgentAction = None
        AgentFinish = None
        LLMResult = None
        HAS_LANGCHAIN = False

from .._utils.telemetry import add_span_attributes, create_span, record_span_exception
from ..client import get_client
from ..exceptions import BrokleError
from ..types.attributes import BrokleOtelSpanAttributes

logger = logging.getLogger(__name__)


class BrokleCallbackHandler(BaseCallbackHandler):
    """
    Comprehensive LangChain callback handler for Brokle observability.

    Provides automatic tracing and observability for all LangChain operations
    including chains, agents, tools, and LLM calls. Follows LangChain's
    callback handler pattern while adding Brokle-specific features.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        capture_inputs: bool = True,
        capture_outputs: bool = True,
        capture_errors: bool = True,
        max_content_length: int = 10000,
    ):
        """
        Initialize Brokle callback handler.

        Args:
            session_id: Session identifier for grouping related operations
            user_id: User identifier for user-scoped analytics
            tags: List of tags for categorization
            metadata: Custom metadata dictionary
            capture_inputs: Whether to capture operation inputs
            capture_outputs: Whether to capture operation outputs
            capture_errors: Whether to capture exceptions
            max_content_length: Maximum length for content capture
        """
        super().__init__()

        self.session_id = session_id or str(uuid.uuid4())
        self.user_id = user_id
        self.tags = tags or []
        self.metadata = metadata or {}
        self.capture_inputs = capture_inputs
        self.capture_outputs = capture_outputs
        self.capture_errors = capture_errors
        self.max_content_length = max_content_length

        # Track active spans for proper nesting
        self._active_spans: Dict[str, Any] = {}
        self._span_stack: List[str] = []

        # Performance tracking
        self._operation_times: Dict[str, float] = {}

        logger.debug(f"Initialized BrokleCallbackHandler for session {self.session_id}")

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> Any:
        """Called when LLM starts running."""
        run_id = str(kwargs.get("run_id", uuid.uuid4()))

        try:
            span = create_span(
                name="langchain.llm",
                attributes={
                    BrokleOtelSpanAttributes.OPERATION_TYPE: "llm_call",
                    BrokleOtelSpanAttributes.SESSION_ID: self.session_id,
                    BrokleOtelSpanAttributes.RUN_ID: run_id,
                    BrokleOtelSpanAttributes.REQUEST_START_TIME: time.time(),
                },
            )

            # Add LLM metadata
            if serialized:
                add_span_attributes(
                    span,
                    {
                        BrokleOtelSpanAttributes.LLM_CLASS: serialized.get(
                            "_type", "unknown"
                        ),
                        BrokleOtelSpanAttributes.LLM_KWARGS: json.dumps(
                            {k: v for k, v in serialized.items() if k != "kwargs"},
                            default=str,
                        )[: self.max_content_length],
                    },
                )

            # Add session and user context
            self._add_context_attributes(span)

            # Capture prompts if enabled
            if self.capture_inputs and prompts:
                for i, prompt in enumerate(prompts[:5]):  # Limit to first 5 prompts
                    add_span_attributes(
                        span, {f"input.prompt.{i}": self._truncate_content(prompt)}
                    )

            self._active_spans[run_id] = span
            self._span_stack.append(run_id)
            self._operation_times[run_id] = time.time()

        except Exception as e:
            logger.warning(f"Failed to handle llm_start: {e}")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Called when LLM ends running."""
        run_id = str(kwargs.get("run_id", ""))

        try:
            span = self._active_spans.get(run_id)
            if not span:
                return

            # Record timing
            start_time = self._operation_times.get(run_id, time.time())
            duration = time.time() - start_time

            add_span_attributes(
                span,
                {
                    BrokleOtelSpanAttributes.REQUEST_DURATION: duration,
                    BrokleOtelSpanAttributes.RESPONSE_END_TIME: time.time(),
                    BrokleOtelSpanAttributes.LLM_SUCCESS: True,
                },
            )

            # Capture response if enabled
            if self.capture_outputs and response:
                self._record_llm_response(span, response)

            span.end()
            self._cleanup_span(run_id)

        except Exception as e:
            logger.warning(f"Failed to handle llm_end: {e}")

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Called when LLM errors."""
        run_id = str(kwargs.get("run_id", ""))

        try:
            span = self._active_spans.get(run_id)
            if not span:
                return

            if self.capture_errors:
                record_span_exception(span, error)
                add_span_attributes(
                    span,
                    {
                        BrokleOtelSpanAttributes.ERROR_TYPE: type(error).__name__,
                        BrokleOtelSpanAttributes.ERROR_MESSAGE: str(error),
                        BrokleOtelSpanAttributes.LLM_SUCCESS: False,
                    },
                )

            span.end()
            self._cleanup_span(run_id)

        except Exception as e:
            logger.warning(f"Failed to handle llm_error: {e}")

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Called when chain starts running."""
        run_id = str(kwargs.get("run_id", uuid.uuid4()))

        try:
            span = create_span(
                name="langchain.chain",
                attributes={
                    BrokleOtelSpanAttributes.OPERATION_TYPE: "chain_execution",
                    BrokleOtelSpanAttributes.SESSION_ID: self.session_id,
                    BrokleOtelSpanAttributes.RUN_ID: run_id,
                    BrokleOtelSpanAttributes.REQUEST_START_TIME: time.time(),
                },
            )

            # Add chain metadata
            if serialized:
                add_span_attributes(
                    span,
                    {
                        BrokleOtelSpanAttributes.CHAIN_TYPE: serialized.get(
                            "_type", "unknown"
                        ),
                        BrokleOtelSpanAttributes.CHAIN_NAME: serialized.get(
                            "name", "unnamed"
                        ),
                    },
                )

            # Add session and user context
            self._add_context_attributes(span)

            # Capture inputs if enabled
            if self.capture_inputs and inputs:
                add_span_attributes(
                    span,
                    {BrokleOtelSpanAttributes.INPUT: self._serialize_inputs(inputs)},
                )

            self._active_spans[run_id] = span
            self._span_stack.append(run_id)
            self._operation_times[run_id] = time.time()

        except Exception as e:
            logger.warning(f"Failed to handle chain_start: {e}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Called when chain ends running."""
        run_id = str(kwargs.get("run_id", ""))

        try:
            span = self._active_spans.get(run_id)
            if not span:
                return

            # Record timing
            start_time = self._operation_times.get(run_id, time.time())
            duration = time.time() - start_time

            add_span_attributes(
                span,
                {
                    BrokleOtelSpanAttributes.REQUEST_DURATION: duration,
                    BrokleOtelSpanAttributes.RESPONSE_END_TIME: time.time(),
                    BrokleOtelSpanAttributes.CHAIN_SUCCESS: True,
                },
            )

            # Capture outputs if enabled
            if self.capture_outputs and outputs:
                add_span_attributes(
                    span,
                    {BrokleOtelSpanAttributes.OUTPUT: self._serialize_outputs(outputs)},
                )

            span.end()
            self._cleanup_span(run_id)

        except Exception as e:
            logger.warning(f"Failed to handle chain_end: {e}")

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Called when chain errors."""
        run_id = str(kwargs.get("run_id", ""))

        try:
            span = self._active_spans.get(run_id)
            if not span:
                return

            if self.capture_errors:
                record_span_exception(span, error)
                add_span_attributes(
                    span,
                    {
                        BrokleOtelSpanAttributes.ERROR_TYPE: type(error).__name__,
                        BrokleOtelSpanAttributes.ERROR_MESSAGE: str(error),
                        BrokleOtelSpanAttributes.CHAIN_SUCCESS: False,
                    },
                )

            span.end()
            self._cleanup_span(run_id)

        except Exception as e:
            logger.warning(f"Failed to handle chain_error: {e}")

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> Any:
        """Called when tool starts running."""
        run_id = str(kwargs.get("run_id", uuid.uuid4()))

        try:
            span = create_span(
                name="langchain.tool",
                attributes={
                    BrokleOtelSpanAttributes.OPERATION_TYPE: "tool_execution",
                    BrokleOtelSpanAttributes.SESSION_ID: self.session_id,
                    BrokleOtelSpanAttributes.RUN_ID: run_id,
                    BrokleOtelSpanAttributes.REQUEST_START_TIME: time.time(),
                },
            )

            # Add tool metadata
            if serialized:
                add_span_attributes(
                    span,
                    {
                        BrokleOtelSpanAttributes.TOOL_NAME: serialized.get(
                            "name", "unknown"
                        ),
                        BrokleOtelSpanAttributes.TOOL_DESCRIPTION: serialized.get(
                            "description", ""
                        )[:500],
                    },
                )

            # Add session and user context
            self._add_context_attributes(span)

            # Capture input if enabled
            if self.capture_inputs and input_str:
                add_span_attributes(
                    span,
                    {
                        BrokleOtelSpanAttributes.TOOL_INPUT: self._truncate_content(
                            input_str
                        )
                    },
                )

            self._active_spans[run_id] = span
            self._span_stack.append(run_id)
            self._operation_times[run_id] = time.time()

        except Exception as e:
            logger.warning(f"Failed to handle tool_start: {e}")

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Called when tool ends running."""
        run_id = str(kwargs.get("run_id", ""))

        try:
            span = self._active_spans.get(run_id)
            if not span:
                return

            # Record timing
            start_time = self._operation_times.get(run_id, time.time())
            duration = time.time() - start_time

            add_span_attributes(
                span,
                {
                    BrokleOtelSpanAttributes.REQUEST_DURATION: duration,
                    BrokleOtelSpanAttributes.RESPONSE_END_TIME: time.time(),
                    BrokleOtelSpanAttributes.TOOL_SUCCESS: True,
                },
            )

            # Capture output if enabled
            if self.capture_outputs and output:
                add_span_attributes(
                    span,
                    {
                        BrokleOtelSpanAttributes.TOOL_OUTPUT: self._truncate_content(
                            output
                        )
                    },
                )

            span.end()
            self._cleanup_span(run_id)

        except Exception as e:
            logger.warning(f"Failed to handle tool_end: {e}")

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Called when tool errors."""
        run_id = str(kwargs.get("run_id", ""))

        try:
            span = self._active_spans.get(run_id)
            if not span:
                return

            if self.capture_errors:
                record_span_exception(span, error)
                add_span_attributes(
                    span,
                    {
                        BrokleOtelSpanAttributes.ERROR_TYPE: type(error).__name__,
                        BrokleOtelSpanAttributes.ERROR_MESSAGE: str(error),
                        BrokleOtelSpanAttributes.TOOL_SUCCESS: False,
                    },
                )

            span.end()
            self._cleanup_span(run_id)

        except Exception as e:
            logger.warning(f"Failed to handle tool_error: {e}")

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Called when agent takes an action."""
        run_id = str(kwargs.get("run_id", uuid.uuid4()))

        try:
            span = create_span(
                name="langchain.agent_action",
                attributes={
                    BrokleOtelSpanAttributes.OPERATION_TYPE: "agent_action",
                    BrokleOtelSpanAttributes.SESSION_ID: self.session_id,
                    BrokleOtelSpanAttributes.RUN_ID: run_id,
                    BrokleOtelSpanAttributes.REQUEST_START_TIME: time.time(),
                },
            )

            # Add action metadata
            if action:
                add_span_attributes(
                    span,
                    {
                        BrokleOtelSpanAttributes.AGENT_TOOL: action.tool,
                        BrokleOtelSpanAttributes.AGENT_TOOL_INPUT: self._truncate_content(
                            str(action.tool_input)
                        ),
                        BrokleOtelSpanAttributes.AGENT_LOG: self._truncate_content(
                            action.log
                        ),
                    },
                )

            # Add session and user context
            self._add_context_attributes(span)

            # For agent actions, we end the span immediately
            span.end()

        except Exception as e:
            logger.warning(f"Failed to handle agent_action: {e}")

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Called when agent finishes."""
        run_id = str(kwargs.get("run_id", uuid.uuid4()))

        try:
            span = create_span(
                name="langchain.agent_finish",
                attributes={
                    BrokleOtelSpanAttributes.OPERATION_TYPE: "agent_finish",
                    BrokleOtelSpanAttributes.SESSION_ID: self.session_id,
                    BrokleOtelSpanAttributes.RUN_ID: run_id,
                    BrokleOtelSpanAttributes.REQUEST_START_TIME: time.time(),
                },
            )

            # Add finish metadata
            if finish and hasattr(finish, "return_values"):
                add_span_attributes(
                    span,
                    {
                        BrokleOtelSpanAttributes.AGENT_RETURN_VALUES: self._serialize_outputs(
                            finish.return_values
                        )
                    },
                )

            if finish and hasattr(finish, "log"):
                add_span_attributes(
                    span,
                    {
                        BrokleOtelSpanAttributes.AGENT_LOG: self._truncate_content(
                            finish.log
                        )
                    },
                )

            # Add session and user context
            self._add_context_attributes(span)

            # For agent finish, we end the span immediately
            span.end()

        except Exception as e:
            logger.warning(f"Failed to handle agent_finish: {e}")

    def _add_context_attributes(self, span: Any):
        """Add session and user context to span"""
        try:
            if self.session_id:
                add_span_attributes(
                    span, {BrokleOtelSpanAttributes.SESSION_ID: self.session_id}
                )

            if self.user_id:
                add_span_attributes(
                    span, {BrokleOtelSpanAttributes.USER_ID: self.user_id}
                )

            if self.tags:
                add_span_attributes(
                    span, {BrokleOtelSpanAttributes.TAGS: ",".join(self.tags)}
                )

            # Add custom metadata
            for key, value in self.metadata.items():
                add_span_attributes(span, {f"metadata.{key}": str(value)})

        except Exception as e:
            logger.warning(f"Failed to add context attributes: {e}")

    def _record_llm_response(self, span: Any, response: LLMResult):
        """Record LLM response attributes"""
        try:
            if hasattr(response, "llm_output") and response.llm_output:
                llm_output = response.llm_output
                if isinstance(llm_output, dict):
                    # Token usage
                    if "token_usage" in llm_output:
                        usage = llm_output["token_usage"]
                        add_span_attributes(
                            span,
                            {
                                BrokleOtelSpanAttributes.INPUT_TOKENS: usage.get(
                                    "prompt_tokens", 0
                                ),
                                BrokleOtelSpanAttributes.OUTPUT_TOKENS: usage.get(
                                    "completion_tokens", 0
                                ),
                                BrokleOtelSpanAttributes.TOTAL_TOKENS: usage.get(
                                    "total_tokens", 0
                                ),
                            },
                        )

                    # Model name
                    if "model_name" in llm_output:
                        add_span_attributes(
                            span,
                            {
                                BrokleOtelSpanAttributes.MODEL_NAME: llm_output[
                                    "model_name"
                                ]
                            },
                        )

            # Capture generations
            if hasattr(response, "generations") and response.generations:
                for i, generation_list in enumerate(
                    response.generations[:3]
                ):  # Limit to first 3
                    for j, generation in enumerate(
                        generation_list[:2]
                    ):  # Limit to first 2 per list
                        if hasattr(generation, "text"):
                            add_span_attributes(
                                span,
                                {
                                    f"output.generation.{i}.{j}": self._truncate_content(
                                        generation.text
                                    )
                                },
                            )

        except Exception as e:
            logger.warning(f"Failed to record LLM response: {e}")

    def _serialize_inputs(self, inputs: Dict[str, Any]) -> str:
        """Safely serialize inputs to string"""
        try:
            # Remove or truncate large values
            safe_inputs = {}
            for key, value in inputs.items():
                if isinstance(value, str):
                    safe_inputs[key] = self._truncate_content(value)
                elif isinstance(value, (list, dict)):
                    safe_inputs[key] = str(value)[: self.max_content_length]
                else:
                    safe_inputs[key] = str(value)

            return json.dumps(safe_inputs, default=str)[: self.max_content_length]

        except Exception:
            return str(inputs)[: self.max_content_length]

    def _serialize_outputs(self, outputs: Dict[str, Any]) -> str:
        """Safely serialize outputs to string"""
        try:
            # Remove or truncate large values
            safe_outputs = {}
            for key, value in outputs.items():
                if isinstance(value, str):
                    safe_outputs[key] = self._truncate_content(value)
                elif isinstance(value, (list, dict)):
                    safe_outputs[key] = str(value)[: self.max_content_length]
                else:
                    safe_outputs[key] = str(value)

            return json.dumps(safe_outputs, default=str)[: self.max_content_length]

        except Exception:
            return str(outputs)[: self.max_content_length]

    def _truncate_content(self, content: str) -> str:
        """Truncate content to maximum length"""
        if len(content) <= self.max_content_length:
            return content
        return content[: self.max_content_length] + "...[TRUNCATED]"

    def _cleanup_span(self, run_id: str):
        """Clean up span tracking data"""
        try:
            if run_id in self._active_spans:
                del self._active_spans[run_id]
            if run_id in self._operation_times:
                del self._operation_times[run_id]
            if run_id in self._span_stack:
                self._span_stack.remove(run_id)
        except Exception as e:
            logger.warning(f"Failed to cleanup span {run_id}: {e}")


# Convenience function for quick setup
def create_callback_handler(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    **kwargs,
) -> BrokleCallbackHandler:
    """
    Create a Brokle callback handler with common settings.

    Args:
        session_id: Session identifier
        user_id: User identifier
        tags: List of tags for categorization
        **kwargs: Additional handler configuration

    Returns:
        Configured BrokleCallbackHandler instance

    Example:
        handler = create_callback_handler(
            session_id="user-session-123",
            tags=["production", "langchain"]
        )

        chain.run(input_text, callbacks=[handler])
    """
    if not HAS_LANGCHAIN:
        raise BrokleError(
            "LangChain not installed. Install with: pip install langchain>=0.1.0"
        )

    return BrokleCallbackHandler(
        session_id=session_id, user_id=user_id, tags=tags, **kwargs
    )


# Export public API
__all__ = [
    "BrokleCallbackHandler",
    "create_callback_handler",
]

if not HAS_LANGCHAIN:
    __all__.append("HAS_LANGCHAIN")
