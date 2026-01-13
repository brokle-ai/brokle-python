"""
Shared helper functions for LLM SDK wrappers.
"""

import json
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from ..types import Attrs
from ..utils.attributes import calculate_total_tokens

if TYPE_CHECKING:
    from ..prompts import Prompt


def extract_brokle_options(
    kwargs: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Extract brokle_options from kwargs and return clean kwargs.

    Args:
        kwargs: Original keyword arguments

    Returns:
        Tuple of (clean_kwargs without brokle_options, brokle_opts dict)
    """
    brokle_options = kwargs.pop("brokle_options", None)
    return kwargs, brokle_options or {}


def add_prompt_attributes(attrs: Dict[str, Any], brokle_opts: Dict[str, Any]) -> None:
    """
    Add prompt attributes to span attributes if prompt is provided and not a fallback.

    Args:
        attrs: Span attributes dict to modify
        brokle_opts: Brokle options containing optional prompt
    """
    prompt: Optional["Prompt"] = brokle_opts.get("prompt")
    if prompt and not prompt.is_fallback:
        attrs[Attrs.BROKLE_PROMPT_NAME] = prompt.name
        attrs[Attrs.BROKLE_PROMPT_VERSION] = prompt.version
        if prompt.id and prompt.id != "fallback":
            attrs[Attrs.BROKLE_PROMPT_ID] = prompt.id


def record_openai_response(span: Any, response: Any, latency_ms: float) -> None:
    """
    Record OpenAI/Azure OpenAI chat completion response attributes on span.

    This helper ensures parity between sync and async paths by extracting
    response data consistently.

    Args:
        span: The span to record attributes on
        response: OpenAI ChatCompletion response object
        latency_ms: Response latency in milliseconds
    """
    # 1. Response ID
    if hasattr(response, "id") and response.id:
        span.set_attribute(Attrs.GEN_AI_RESPONSE_ID, response.id)

    # 2. Response model
    if hasattr(response, "model") and response.model:
        span.set_attribute(Attrs.GEN_AI_RESPONSE_MODEL, response.model)

    # 3. System fingerprint (OpenAI determinism tracking)
    if hasattr(response, "system_fingerprint") and response.system_fingerprint:
        span.set_attribute(
            Attrs.OPENAI_RESPONSE_SYSTEM_FINGERPRINT,
            response.system_fingerprint,
        )

    # 4. Output messages and finish reasons from choices
    if hasattr(response, "choices") and response.choices:
        output_messages = []
        finish_reasons = []

        for choice in response.choices:
            if hasattr(choice, "message") and choice.message:
                msg_dict = {
                    "role": getattr(choice.message, "role", None),
                    "content": getattr(choice.message, "content", None),
                }

                # Tool calls (function calling)
                if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                    tool_calls = []
                    for tc in choice.message.tool_calls:
                        tool_calls.append(
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                        )
                    msg_dict["tool_calls"] = tool_calls

                # Refusal (content moderation)
                if hasattr(choice.message, "refusal") and choice.message.refusal:
                    msg_dict["refusal"] = choice.message.refusal

                output_messages.append(msg_dict)

            if hasattr(choice, "finish_reason") and choice.finish_reason:
                finish_reasons.append(choice.finish_reason)

        if output_messages:
            span.set_attribute(Attrs.GEN_AI_OUTPUT_MESSAGES, json.dumps(output_messages))
        if finish_reasons:
            span.set_attribute(Attrs.GEN_AI_RESPONSE_FINISH_REASONS, finish_reasons)

    # 5. Token usage
    if hasattr(response, "usage") and response.usage:
        usage = response.usage
        if hasattr(usage, "prompt_tokens") and usage.prompt_tokens:
            span.set_attribute(Attrs.GEN_AI_USAGE_INPUT_TOKENS, usage.prompt_tokens)
        if hasattr(usage, "completion_tokens") and usage.completion_tokens:
            span.set_attribute(Attrs.GEN_AI_USAGE_OUTPUT_TOKENS, usage.completion_tokens)

        total_tokens = calculate_total_tokens(
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
        )
        if total_tokens:
            span.set_attribute(Attrs.BROKLE_USAGE_TOTAL_TOKENS, total_tokens)

    # 6. Latency
    span.set_attribute(Attrs.BROKLE_USAGE_LATENCY_MS, latency_ms)
