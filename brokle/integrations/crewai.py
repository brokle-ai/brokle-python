"""
CrewAI integration for automatic Brokle tracing.

Provides callback handlers that automatically create OpenTelemetry spans
for CrewAI operations (agents, tasks, tools, crews).

Example:
    from crewai import Agent, Task, Crew
    from brokle.integrations import BrokleCrewAICallback

    # Create Brokle callback
    callback = BrokleCrewAICallback(
        user_id="user-123",
        session_id="session-456"
    )

    # Create agents with callback
    researcher = Agent(
        role="Senior Researcher",
        goal="Research the topic",
        backstory="An experienced researcher",
        callbacks=[callback],
    )

    # Create crew with callback
    crew = Crew(
        agents=[researcher],
        tasks=[...],
        callbacks=[callback],
    )

    # Run crew - all operations automatically traced
    result = crew.kickoff()
"""

import json
import time
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

try:
    from crewai.utilities.events.base_event_listener import BaseEventListener
except ImportError:
    # Fallback for older CrewAI versions
    BaseEventListener = object

from opentelemetry.trace import Status, StatusCode

from .._client import get_client
from ..types import Attrs, LLMProvider, OperationType, SpanType


class BrokleCrewAICallback(BaseEventListener if BaseEventListener else object):
    """
    CrewAI callback handler for automatic Brokle tracing.

    This callback handler automatically creates OpenTelemetry spans for
    CrewAI operations, following GenAI 1.28+ semantic conventions.

    Supports tracing for:
    - Crew execution (workflow spans)
    - Agent execution (agent spans)
    - Task execution (chain spans)
    - Tool usage (tool spans)
    - LLM calls (generation spans)

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
    ):
        """
        Initialize CrewAI callback handler.

        Args:
            user_id: User identifier
            session_id: Session identifier
            metadata: Custom metadata
            tags: Categorization tags
        """
        if BaseEventListener and BaseEventListener != object:
            super().__init__()

        self.user_id = user_id
        self.session_id = session_id
        self.metadata = metadata or {}
        self.tags = tags or []

        self._client = get_client()
        self._spans: Dict[str, Any] = {}
        self._span_start_times: Dict[str, float] = {}
        self._agent_iterations: Dict[str, int] = {}

    def _get_common_attributes(self) -> Dict[str, Any]:
        """Get common attributes to attach to all spans."""
        attrs = {
            Attrs.GEN_AI_FRAMEWORK_NAME: "crewai",
        }

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

    def _generate_id(self) -> str:
        """Generate a unique ID for span tracking."""
        return str(uuid4())

    # ========== Crew Events ==========

    def on_crew_start(
        self,
        crew_name: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """
        Called when a crew starts execution.

        Creates a workflow span for the crew.

        Returns:
            Crew run ID for tracking
        """
        run_id = self._generate_id()

        attrs = self._get_common_attributes()
        attrs.update({
            Attrs.BROKLE_SPAN_TYPE: SpanType.WORKFLOW,
            Attrs.GEN_AI_COMPONENT_TYPE: "crew",
            "crewai.crew_name": crew_name or "crew",
        })

        if inputs:
            attrs["crewai.crew_inputs"] = json.dumps(inputs, default=str)

        span_name = f"crew:{crew_name}" if crew_name else "crew"
        span = self._client._tracer.start_span(name=span_name, attributes=attrs)

        self._spans[run_id] = span
        self._span_start_times[run_id] = time.time()

        return run_id

    def on_crew_end(
        self,
        run_id: str,
        output: Optional[Any] = None,
        **kwargs,
    ) -> None:
        """
        Called when a crew finishes execution.

        Updates the span with outputs.
        """
        span = self._spans.get(run_id)
        if not span:
            return

        try:
            if output is not None:
                output_str = str(output) if not isinstance(output, str) else output
                # Truncate if too long
                if len(output_str) > 10000:
                    output_str = output_str[:10000] + "..."
                span.set_attribute("crewai.crew_output", output_str)

            if run_id in self._span_start_times:
                latency_ms = (time.time() - self._span_start_times[run_id]) * 1000
                span.set_attribute(Attrs.BROKLE_USAGE_LATENCY_MS, latency_ms)

            span.set_status(Status(StatusCode.OK))

        finally:
            span.end()
            self._spans.pop(run_id, None)
            self._span_start_times.pop(run_id, None)

    def on_crew_error(
        self,
        run_id: str,
        error: Exception,
        **kwargs,
    ) -> None:
        """
        Called when a crew encounters an error.

        Records the error in the span.
        """
        span = self._spans.get(run_id)
        if not span:
            return

        try:
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.record_exception(error)
        finally:
            span.end()
            self._spans.pop(run_id, None)
            self._span_start_times.pop(run_id, None)

    # ========== Agent Events ==========

    def on_agent_start(
        self,
        agent_name: Optional[str] = None,
        role: Optional[str] = None,
        goal: Optional[str] = None,
        backstory: Optional[str] = None,
        tools: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """
        Called when an agent starts execution.

        Creates an agent span.

        Returns:
            Agent run ID for tracking
        """
        run_id = self._generate_id()

        attrs = self._get_common_attributes()
        attrs.update({
            Attrs.BROKLE_SPAN_TYPE: SpanType.AGENT,
            Attrs.GEN_AI_COMPONENT_TYPE: "agent",
            Attrs.GEN_AI_AGENT_STRATEGY: "react",  # CrewAI uses ReAct pattern
        })

        if agent_name:
            attrs[Attrs.GEN_AI_AGENT_NAME] = agent_name
            attrs["crewai.agent_name"] = agent_name
        if role:
            attrs["crewai.agent_role"] = role
        if goal:
            attrs["crewai.agent_goal"] = goal
        if backstory:
            attrs["crewai.agent_backstory"] = backstory
        if tools:
            attrs["crewai.agent_tools"] = json.dumps(tools)

        # Initialize iteration counter
        self._agent_iterations[run_id] = 0

        span_name = f"agent:{role or agent_name or 'agent'}"
        span = self._client._tracer.start_span(name=span_name, attributes=attrs)

        self._spans[run_id] = span
        self._span_start_times[run_id] = time.time()

        return run_id

    def on_agent_iteration(
        self,
        run_id: str,
        thought: Optional[str] = None,
        action: Optional[str] = None,
        action_input: Optional[Any] = None,
        **kwargs,
    ) -> None:
        """
        Called on each agent reasoning iteration.

        Updates iteration count and logs reasoning steps.
        """
        span = self._spans.get(run_id)
        if not span:
            return

        # Increment iteration counter
        if run_id in self._agent_iterations:
            self._agent_iterations[run_id] += 1
            span.set_attribute(
                Attrs.GEN_AI_AGENT_ITERATION_COUNT,
                self._agent_iterations[run_id]
            )

        # Log iteration details as events
        if thought or action:
            iteration_data = {}
            if thought:
                iteration_data["thought"] = thought
            if action:
                iteration_data["action"] = action
            if action_input:
                iteration_data["action_input"] = str(action_input)

            span.add_event(
                "agent_iteration",
                attributes={
                    "iteration": self._agent_iterations.get(run_id, 0),
                    "data": json.dumps(iteration_data, default=str),
                }
            )

    def on_agent_end(
        self,
        run_id: str,
        output: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Called when an agent finishes execution.

        Updates the span with outputs.
        """
        span = self._spans.get(run_id)
        if not span:
            return

        try:
            if output is not None:
                output_messages = [{
                    "role": "assistant",
                    "content": str(output),
                }]
                span.set_attribute(
                    Attrs.GEN_AI_OUTPUT_MESSAGES, json.dumps(output_messages)
                )

            if run_id in self._span_start_times:
                latency_ms = (time.time() - self._span_start_times[run_id]) * 1000
                span.set_attribute(Attrs.BROKLE_USAGE_LATENCY_MS, latency_ms)

            span.set_status(Status(StatusCode.OK))

        finally:
            span.end()
            self._spans.pop(run_id, None)
            self._span_start_times.pop(run_id, None)
            self._agent_iterations.pop(run_id, None)

    def on_agent_error(
        self,
        run_id: str,
        error: Exception,
        **kwargs,
    ) -> None:
        """
        Called when an agent encounters an error.

        Records the error in the span.
        """
        span = self._spans.get(run_id)
        if not span:
            return

        try:
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.record_exception(error)
        finally:
            span.end()
            self._spans.pop(run_id, None)
            self._span_start_times.pop(run_id, None)
            self._agent_iterations.pop(run_id, None)

    # ========== Task Events ==========

    def on_task_start(
        self,
        task_description: Optional[str] = None,
        expected_output: Optional[str] = None,
        agent_name: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Called when a task starts execution.

        Creates a task span.

        Returns:
            Task run ID for tracking
        """
        run_id = self._generate_id()

        attrs = self._get_common_attributes()
        attrs.update({
            Attrs.BROKLE_SPAN_TYPE: SpanType.CHAIN,
            Attrs.GEN_AI_COMPONENT_TYPE: "task",
        })

        if task_description:
            attrs["crewai.task_description"] = task_description
            # Use description as input
            attrs[Attrs.GEN_AI_INPUT_MESSAGES] = json.dumps([
                {"role": "user", "content": task_description}
            ])
        if expected_output:
            attrs["crewai.task_expected_output"] = expected_output
        if agent_name:
            attrs["crewai.task_agent"] = agent_name

        span_name = f"task:{task_description[:50]}..." if task_description and len(task_description) > 50 else f"task:{task_description or 'task'}"
        span = self._client._tracer.start_span(name=span_name, attributes=attrs)

        self._spans[run_id] = span
        self._span_start_times[run_id] = time.time()

        return run_id

    def on_task_end(
        self,
        run_id: str,
        output: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Called when a task finishes execution.

        Updates the span with outputs.
        """
        span = self._spans.get(run_id)
        if not span:
            return

        try:
            if output is not None:
                output_messages = [{
                    "role": "assistant",
                    "content": str(output),
                }]
                span.set_attribute(
                    Attrs.GEN_AI_OUTPUT_MESSAGES, json.dumps(output_messages)
                )

            if run_id in self._span_start_times:
                latency_ms = (time.time() - self._span_start_times[run_id]) * 1000
                span.set_attribute(Attrs.BROKLE_USAGE_LATENCY_MS, latency_ms)

            span.set_status(Status(StatusCode.OK))

        finally:
            span.end()
            self._spans.pop(run_id, None)
            self._span_start_times.pop(run_id, None)

    def on_task_error(
        self,
        run_id: str,
        error: Exception,
        **kwargs,
    ) -> None:
        """
        Called when a task encounters an error.

        Records the error in the span.
        """
        span = self._spans.get(run_id)
        if not span:
            return

        try:
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.record_exception(error)
        finally:
            span.end()
            self._spans.pop(run_id, None)
            self._span_start_times.pop(run_id, None)

    # ========== Tool Events ==========

    def on_tool_start(
        self,
        tool_name: Optional[str] = None,
        tool_input: Optional[Any] = None,
        tool_description: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Called when a tool starts execution.

        Creates a tool span.

        Returns:
            Tool run ID for tracking
        """
        run_id = self._generate_id()

        attrs = self._get_common_attributes()
        attrs.update({
            Attrs.BROKLE_SPAN_TYPE: SpanType.TOOL,
            Attrs.GEN_AI_COMPONENT_TYPE: "tool",
        })

        if tool_name:
            attrs[Attrs.GEN_AI_TOOL_NAME] = tool_name
        if tool_description:
            attrs[Attrs.GEN_AI_TOOL_DESCRIPTION] = tool_description
        if tool_input is not None:
            input_str = json.dumps(tool_input, default=str) if not isinstance(tool_input, str) else tool_input
            attrs[Attrs.GEN_AI_TOOL_PARAMETERS] = input_str

        span_name = f"tool:{tool_name or 'tool'}"
        span = self._client._tracer.start_span(name=span_name, attributes=attrs)

        self._spans[run_id] = span
        self._span_start_times[run_id] = time.time()

        return run_id

    def on_tool_end(
        self,
        run_id: str,
        output: Optional[Any] = None,
        **kwargs,
    ) -> None:
        """
        Called when a tool finishes execution.

        Updates the span with outputs.
        """
        span = self._spans.get(run_id)
        if not span:
            return

        try:
            if output is not None:
                output_str = json.dumps(output, default=str) if not isinstance(output, str) else output
                span.set_attribute("crewai.tool_output", output_str)

            if run_id in self._span_start_times:
                latency_ms = (time.time() - self._span_start_times[run_id]) * 1000
                span.set_attribute(Attrs.BROKLE_USAGE_LATENCY_MS, latency_ms)

            span.set_status(Status(StatusCode.OK))

        finally:
            span.end()
            self._spans.pop(run_id, None)
            self._span_start_times.pop(run_id, None)

    def on_tool_error(
        self,
        run_id: str,
        error: Exception,
        **kwargs,
    ) -> None:
        """
        Called when a tool encounters an error.

        Records the error in the span.
        """
        span = self._spans.get(run_id)
        if not span:
            return

        try:
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.record_exception(error)
        finally:
            span.end()
            self._spans.pop(run_id, None)
            self._span_start_times.pop(run_id, None)

    # ========== LLM Events ==========

    def on_llm_start(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Called when an LLM call starts.

        Creates a generation span.

        Returns:
            LLM call run ID for tracking
        """
        run_id = self._generate_id()

        attrs = self._get_common_attributes()
        attrs.update({
            Attrs.BROKLE_SPAN_TYPE: SpanType.GENERATION,
            Attrs.GEN_AI_OPERATION_NAME: OperationType.CHAT,
        })

        if model:
            attrs[Attrs.GEN_AI_REQUEST_MODEL] = model
        if provider:
            attrs[Attrs.GEN_AI_PROVIDER_NAME] = provider
        if messages:
            attrs[Attrs.GEN_AI_INPUT_MESSAGES] = json.dumps(messages)
        if temperature is not None:
            attrs[Attrs.GEN_AI_REQUEST_TEMPERATURE] = temperature
        if max_tokens is not None:
            attrs[Attrs.GEN_AI_REQUEST_MAX_TOKENS] = max_tokens

        span_name = f"chat {model}" if model else "chat"
        span = self._client._tracer.start_span(name=span_name, attributes=attrs)

        self._spans[run_id] = span
        self._span_start_times[run_id] = time.time()

        return run_id

    def on_llm_end(
        self,
        run_id: str,
        response: Optional[str] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        finish_reason: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Called when an LLM call finishes.

        Updates the span with outputs and usage.
        """
        span = self._spans.get(run_id)
        if not span:
            return

        try:
            if response is not None:
                output_messages = [{
                    "role": "assistant",
                    "content": response,
                }]
                span.set_attribute(
                    Attrs.GEN_AI_OUTPUT_MESSAGES, json.dumps(output_messages)
                )

            if finish_reason:
                span.set_attribute(
                    Attrs.GEN_AI_RESPONSE_FINISH_REASONS, [finish_reason]
                )

            if input_tokens is not None:
                span.set_attribute(Attrs.GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
            if output_tokens is not None:
                span.set_attribute(Attrs.GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)
            if input_tokens is not None and output_tokens is not None:
                span.set_attribute(
                    Attrs.BROKLE_USAGE_TOTAL_TOKENS,
                    input_tokens + output_tokens
                )

            if run_id in self._span_start_times:
                latency_ms = (time.time() - self._span_start_times[run_id]) * 1000
                span.set_attribute(Attrs.BROKLE_USAGE_LATENCY_MS, latency_ms)

            span.set_status(Status(StatusCode.OK))

        finally:
            span.end()
            self._spans.pop(run_id, None)
            self._span_start_times.pop(run_id, None)

    def on_llm_error(
        self,
        run_id: str,
        error: Exception,
        **kwargs,
    ) -> None:
        """
        Called when an LLM call encounters an error.

        Records the error in the span.
        """
        span = self._spans.get(run_id)
        if not span:
            return

        try:
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.record_exception(error)
        finally:
            span.end()
            self._spans.pop(run_id, None)
            self._span_start_times.pop(run_id, None)


# Convenience function for simpler integration
def instrument_crewai(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> BrokleCrewAICallback:
    """
    Create and return a BrokleCrewAICallback instance.

    This is a convenience function for creating the callback handler.

    Args:
        user_id: User identifier
        session_id: Session identifier
        metadata: Custom metadata
        tags: Categorization tags

    Returns:
        BrokleCrewAICallback instance

    Example:
        from crewai import Crew
        from brokle.integrations import instrument_crewai

        callback = instrument_crewai(user_id="user-123")

        crew = Crew(
            agents=[...],
            tasks=[...],
            callbacks=[callback],
        )
    """
    return BrokleCrewAICallback(
        user_id=user_id,
        session_id=session_id,
        metadata=metadata,
        tags=tags,
    )
