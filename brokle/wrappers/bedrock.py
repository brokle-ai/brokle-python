"""
AWS Bedrock SDK wrapper for automatic observability.

Wraps AWS Bedrock Runtime client to automatically create OTEL spans with GenAI 1.28+ attributes.
Supports the Converse API for cross-model compatibility.
"""

import json
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from opentelemetry.trace import Status, StatusCode

from .._client import get_client
from ..streaming import StreamingAccumulator
from ..types import Attrs, LLMProvider, OperationType, SpanType
from ..utils.attributes import calculate_total_tokens
from ._common import add_prompt_attributes, extract_brokle_options

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient


def wrap_bedrock(client: "BedrockRuntimeClient") -> "BedrockRuntimeClient":
    """
    Wrap AWS Bedrock Runtime client for automatic observability.

    This function wraps the Bedrock Runtime client's converse method
    to automatically create OTEL spans with GenAI semantic attributes.

    Args:
        client: BedrockRuntimeClient instance

    Returns:
        Wrapped BedrockRuntimeClient (same instance with instrumented methods)

    Example:
        >>> import boto3
        >>> from brokle import get_client, wrap_bedrock
        >>>
        >>> # Initialize Brokle
        >>> brokle = get_client()
        >>>
        >>> # Wrap Bedrock client
        >>> bedrock = wrap_bedrock(boto3.client("bedrock-runtime"))
        >>>
        >>> # All calls automatically tracked
        >>> response = bedrock.converse(
        ...     modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        ...     messages=[{"role": "user", "content": [{"text": "Hello!"}]}]
        ... )
        >>> brokle.flush()
    """
    # Return unwrapped if SDK disabled
    brokle_client = get_client()
    if not brokle_client.config.enabled:
        return client

    original_converse = client.converse

    def wrapped_converse(*args, **kwargs):
        """Wrapped converse with automatic tracing."""
        # Extract brokle_options before processing kwargs
        kwargs, brokle_opts = extract_brokle_options(kwargs)

        brokle_client = get_client()

        model_id = kwargs.get("modelId", "unknown")
        messages = kwargs.get("messages", [])
        system = kwargs.get("system", [])
        inference_config = kwargs.get("inferenceConfig", {})
        guardrail_config = kwargs.get("guardrailConfig", {})

        # Build input messages
        input_messages = _build_input_messages(messages)
        system_messages = _build_system_messages(system)

        attrs = {
            Attrs.BROKLE_SPAN_TYPE: SpanType.GENERATION,
            Attrs.GEN_AI_PROVIDER_NAME: LLMProvider.BEDROCK,
            Attrs.GEN_AI_OPERATION_NAME: OperationType.CHAT,
            Attrs.GEN_AI_REQUEST_MODEL: model_id,
            Attrs.BEDROCK_REQUEST_MODEL_ID: model_id,
            Attrs.BROKLE_STREAMING: False,
        }

        if input_messages:
            attrs[Attrs.GEN_AI_INPUT_MESSAGES] = json.dumps(input_messages)
        if system_messages:
            attrs[Attrs.GEN_AI_SYSTEM_INSTRUCTIONS] = json.dumps(system_messages)

        # Extract inference config parameters
        if inference_config:
            if "maxTokens" in inference_config:
                attrs[Attrs.GEN_AI_REQUEST_MAX_TOKENS] = inference_config["maxTokens"]
            if "temperature" in inference_config:
                attrs[Attrs.GEN_AI_REQUEST_TEMPERATURE] = inference_config["temperature"]
            if "topP" in inference_config:
                attrs[Attrs.GEN_AI_REQUEST_TOP_P] = inference_config["topP"]
            if "stopSequences" in inference_config:
                attrs[Attrs.GEN_AI_REQUEST_STOP_SEQUENCES] = inference_config["stopSequences"]

        # Guardrail config
        if guardrail_config:
            if "guardrailIdentifier" in guardrail_config:
                attrs[Attrs.BEDROCK_REQUEST_GUARDRAIL_ID] = guardrail_config["guardrailIdentifier"]
            if "guardrailVersion" in guardrail_config:
                attrs[Attrs.BEDROCK_REQUEST_GUARDRAIL_VERSION] = guardrail_config["guardrailVersion"]

        add_prompt_attributes(attrs, brokle_opts)

        # Extract short model name for span
        model_name = model_id.split("/")[-1] if "/" in model_id else model_id.split(".")[-1]
        span_name = f"{OperationType.CHAT} {model_name}"

        return _handle_sync_response(
            brokle_client, original_converse, args, kwargs, span_name, attrs
        )

    def _handle_sync_response(
        brokle_client, original_method, args, kwargs, span_name, attrs
    ):
        """Handle non-streaming response with standard span lifecycle."""
        with brokle_client.start_as_current_span(span_name, attributes=attrs) as span:
            try:
                start_time = time.time()
                response = original_method(*args, **kwargs)
                latency_ms = (time.time() - start_time) * 1000

                # Extract response metadata
                if "output" in response and "message" in response["output"]:
                    message = response["output"]["message"]
                    output_messages = []

                    if "content" in message:
                        content_parts = []
                        for content in message["content"]:
                            if "text" in content:
                                content_parts.append(content["text"])
                            elif "toolUse" in content:
                                # Handle tool use
                                tool = content["toolUse"]
                                output_messages.append({
                                    "role": "assistant",
                                    "tool_calls": [{
                                        "id": tool.get("toolUseId", ""),
                                        "type": "function",
                                        "function": {
                                            "name": tool.get("name", ""),
                                            "arguments": json.dumps(tool.get("input", {})),
                                        },
                                    }],
                                })

                        if content_parts:
                            output_messages.insert(0, {
                                "role": "assistant",
                                "content": "".join(content_parts),
                            })

                    if output_messages:
                        span.set_attribute(
                            Attrs.GEN_AI_OUTPUT_MESSAGES, json.dumps(output_messages)
                        )

                # Stop reason
                if "stopReason" in response:
                    span.set_attribute(
                        Attrs.GEN_AI_RESPONSE_FINISH_REASONS, [response["stopReason"]]
                    )
                    span.set_attribute(
                        Attrs.BEDROCK_RESPONSE_STOP_REASON, response["stopReason"]
                    )

                # Token usage
                if "usage" in response:
                    usage = response["usage"]
                    if "inputTokens" in usage:
                        span.set_attribute(
                            Attrs.GEN_AI_USAGE_INPUT_TOKENS, usage["inputTokens"]
                        )
                    if "outputTokens" in usage:
                        span.set_attribute(
                            Attrs.GEN_AI_USAGE_OUTPUT_TOKENS, usage["outputTokens"]
                        )
                    if "totalTokens" in usage:
                        span.set_attribute(
                            Attrs.BROKLE_USAGE_TOTAL_TOKENS, usage["totalTokens"]
                        )
                    else:
                        total = calculate_total_tokens(
                            usage.get("inputTokens"),
                            usage.get("outputTokens"),
                        )
                        if total:
                            span.set_attribute(Attrs.BROKLE_USAGE_TOTAL_TOKENS, total)

                # Metrics
                if "metrics" in response:
                    span.set_attribute(
                        Attrs.BEDROCK_RESPONSE_METRICS, json.dumps(response["metrics"])
                    )

                span.set_attribute(Attrs.BROKLE_USAGE_LATENCY_MS, latency_ms)
                span.set_status(Status(StatusCode.OK))

                return response

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    client.converse = wrapped_converse

    # Also wrap converse_stream if available
    if hasattr(client, "converse_stream"):
        original_converse_stream = client.converse_stream

        def wrapped_converse_stream(*args, **kwargs):
            """Wrapped converse_stream with automatic tracing."""
            kwargs, brokle_opts = extract_brokle_options(kwargs)

            brokle_client = get_client()

            model_id = kwargs.get("modelId", "unknown")
            messages = kwargs.get("messages", [])
            system = kwargs.get("system", [])
            inference_config = kwargs.get("inferenceConfig", {})

            input_messages = _build_input_messages(messages)
            system_messages = _build_system_messages(system)

            attrs = {
                Attrs.BROKLE_SPAN_TYPE: SpanType.GENERATION,
                Attrs.GEN_AI_PROVIDER_NAME: LLMProvider.BEDROCK,
                Attrs.GEN_AI_OPERATION_NAME: OperationType.CHAT,
                Attrs.GEN_AI_REQUEST_MODEL: model_id,
                Attrs.BEDROCK_REQUEST_MODEL_ID: model_id,
                Attrs.BROKLE_STREAMING: True,
            }

            if input_messages:
                attrs[Attrs.GEN_AI_INPUT_MESSAGES] = json.dumps(input_messages)
            if system_messages:
                attrs[Attrs.GEN_AI_SYSTEM_INSTRUCTIONS] = json.dumps(system_messages)

            if inference_config:
                if "maxTokens" in inference_config:
                    attrs[Attrs.GEN_AI_REQUEST_MAX_TOKENS] = inference_config["maxTokens"]
                if "temperature" in inference_config:
                    attrs[Attrs.GEN_AI_REQUEST_TEMPERATURE] = inference_config["temperature"]
                if "topP" in inference_config:
                    attrs[Attrs.GEN_AI_REQUEST_TOP_P] = inference_config["topP"]

            add_prompt_attributes(attrs, brokle_opts)

            model_name = model_id.split("/")[-1] if "/" in model_id else model_id.split(".")[-1]
            span_name = f"{OperationType.CHAT} {model_name}"

            tracer = brokle_client._tracer
            span = tracer.start_span(span_name, attributes=attrs)

            try:
                start_time = time.perf_counter()
                response = original_converse_stream(*args, **kwargs)
                accumulator = StreamingAccumulator(start_time)
                return _BedrockStreamWrapper(response, span, accumulator)
            except BaseException as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                span.end()
                raise

        client.converse_stream = wrapped_converse_stream

    return client


class _BedrockStreamWrapper:
    """Wrapper for Bedrock streaming responses."""

    def __init__(self, response, span, accumulator):
        self._response = response
        self._span = span
        self._accumulator = accumulator
        self._content_parts = []
        self._stop_reason = None
        self._usage = {}
        self._stream = response.get("stream", iter([]))

    def __iter__(self):
        return self

    def __next__(self):
        try:
            event = next(self._stream)
            self._accumulator.on_chunk_received()

            # Handle different event types
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"].get("delta", {})
                if "text" in delta:
                    self._content_parts.append(delta["text"])
            elif "messageStop" in event:
                self._stop_reason = event["messageStop"].get("stopReason")
            elif "metadata" in event:
                metadata = event["metadata"]
                if "usage" in metadata:
                    self._usage = metadata["usage"]

            return event

        except StopIteration:
            self._finalize()
            raise

    def _finalize(self):
        """Finalize span with accumulated data."""
        if self._content_parts:
            output_messages = [{
                "role": "assistant",
                "content": "".join(self._content_parts),
            }]
            self._span.set_attribute(
                Attrs.GEN_AI_OUTPUT_MESSAGES, json.dumps(output_messages)
            )

        if self._stop_reason:
            self._span.set_attribute(
                Attrs.GEN_AI_RESPONSE_FINISH_REASONS, [self._stop_reason]
            )
            self._span.set_attribute(
                Attrs.BEDROCK_RESPONSE_STOP_REASON, self._stop_reason
            )

        if self._usage:
            if "inputTokens" in self._usage:
                self._span.set_attribute(
                    Attrs.GEN_AI_USAGE_INPUT_TOKENS, self._usage["inputTokens"]
                )
            if "outputTokens" in self._usage:
                self._span.set_attribute(
                    Attrs.GEN_AI_USAGE_OUTPUT_TOKENS, self._usage["outputTokens"]
                )

        # Set streaming metrics
        if self._accumulator.ttft_ms is not None:
            self._span.set_attribute(Attrs.GEN_AI_RESPONSE_TTFT, self._accumulator.ttft_ms)
        if self._accumulator.avg_itl_ms is not None:
            self._span.set_attribute(Attrs.GEN_AI_RESPONSE_ITL, self._accumulator.avg_itl_ms)
        if self._accumulator.duration_ms is not None:
            self._span.set_attribute(Attrs.BROKLE_USAGE_LATENCY_MS, self._accumulator.duration_ms)

        self._span.set_status(Status(StatusCode.OK))
        self._span.end()


def _build_input_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build input messages from Bedrock message format."""
    result = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", [])

        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    text_parts.append(item["text"])
            elif isinstance(item, str):
                text_parts.append(item)

        if text_parts:
            result.append({
                "role": role,
                "content": "".join(text_parts),
            })

    return result


def _build_system_messages(system: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build system messages from Bedrock system format."""
    result = []
    for item in system:
        if isinstance(item, dict) and "text" in item:
            result.append({
                "role": "system",
                "content": item["text"],
            })
        elif isinstance(item, str):
            result.append({
                "role": "system",
                "content": item,
            })
    return result
