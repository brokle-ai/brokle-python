"""
OpenAI Provider Implementation - Observability Only

Specific telemetry and instrumentation logic for OpenAI SDK.
Focused on request/response attribute extraction for observability.

Business logic (cost calculation, routing, caching) is handled by backend.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from ..observability.attributes import (
    BrokleOtelSpanAttributes as BrokleInstrumentationAttributes,
)
from .base import BaseProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """OpenAI-specific provider implementation for observability."""

    def get_provider_name(self) -> str:
        """Return OpenAI provider identifier."""
        return "openai"

    def get_methods_to_instrument(self) -> List[Dict[str, Any]]:
        """Define OpenAI SDK methods to instrument."""
        return [
            # Chat Completions (primary API)
            {
                "path": "chat.completions.create",
                "operation": "chat_completion",
                "async": False,
                "stream_support": True,
                "cost_tracked": True,
            },
            {
                "path": "chat.completions.acreate",
                "operation": "chat_completion",
                "async": True,
                "stream_support": True,
                "cost_tracked": True,
            },
            # Legacy Completions
            {
                "path": "completions.create",
                "operation": "completion",
                "async": False,
                "stream_support": True,
                "cost_tracked": True,
            },
            # Embeddings
            {
                "path": "embeddings.create",
                "operation": "embedding",
                "async": False,
                "stream_support": False,
                "cost_tracked": True,
            },
            # Fine-tuning (tracking only)
            {
                "path": "fine_tuning.jobs.create",
                "operation": "fine_tune_create",
                "async": False,
                "stream_support": False,
                "cost_tracked": False,
            },
            {
                "path": "fine_tuning.jobs.retrieve",
                "operation": "fine_tune_retrieve",
                "async": False,
                "stream_support": False,
                "cost_tracked": False,
            },
            # Images (DALL-E)
            {
                "path": "images.generate",
                "operation": "image_generation",
                "async": False,
                "stream_support": False,
                "cost_tracked": True,
            },
            # Audio (Whisper, TTS)
            {
                "path": "audio.transcriptions.create",
                "operation": "audio_transcription",
                "async": False,
                "stream_support": False,
                "cost_tracked": True,
            },
            {
                "path": "audio.speech.create",
                "operation": "audio_speech",
                "async": False,
                "stream_support": False,
                "cost_tracked": True,
            },
        ]

    def extract_request_attributes(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract OpenAI request attributes for telemetry."""
        attributes = {}

        # Model information
        if "model" in kwargs:
            model = kwargs["model"]
            attributes[BrokleInstrumentationAttributes.MODEL_NAME] = model
            attributes[BrokleInstrumentationAttributes.MODEL_NAME_NORMALIZED] = (
                self.normalize_model_name(model)
            )

        # Chat completion specific
        if "messages" in kwargs:
            messages = kwargs["messages"]
            if isinstance(messages, list):
                attributes[BrokleInstrumentationAttributes.MESSAGE_COUNT] = len(
                    messages
                )

                # Extract message roles and types
                roles = [
                    msg.get("role", "unknown")
                    for msg in messages
                    if isinstance(msg, dict)
                ]
                attributes[BrokleInstrumentationAttributes.MESSAGE_ROLES] = ",".join(
                    roles
                )

                # Check for system message
                has_system = any(
                    msg.get("role") == "system"
                    for msg in messages
                    if isinstance(msg, dict)
                )
                attributes[BrokleInstrumentationAttributes.SYSTEM_MESSAGE] = has_system

        # Legacy completion specific
        if "prompt" in kwargs:
            prompt = kwargs["prompt"]
            if isinstance(prompt, str):
                attributes[BrokleInstrumentationAttributes.PROMPT_LENGTH] = len(prompt)
            elif isinstance(prompt, list):
                attributes[BrokleInstrumentationAttributes.PROMPT_COUNT] = len(prompt)

        # Common parameters
        common_params = {
            "max_tokens": BrokleInstrumentationAttributes.MAX_TOKENS,
            "temperature": BrokleInstrumentationAttributes.TEMPERATURE,
            "top_p": BrokleInstrumentationAttributes.TOP_P,
            "frequency_penalty": BrokleInstrumentationAttributes.FREQUENCY_PENALTY,
            "presence_penalty": BrokleInstrumentationAttributes.PRESENCE_PENALTY,
            "n": BrokleInstrumentationAttributes.N_COMPLETIONS,
        }

        for param, attr in common_params.items():
            if param in kwargs and kwargs[param] is not None:
                attributes[attr] = kwargs[param]

        # Streaming
        if "stream" in kwargs:
            attributes[BrokleInstrumentationAttributes.STREAM_ENABLED] = kwargs[
                "stream"
            ]

        # Function calling / Tools
        if "functions" in kwargs:
            functions = kwargs["functions"]
            if isinstance(functions, list):
                attributes[BrokleInstrumentationAttributes.FUNCTION_COUNT] = len(
                    functions
                )
                function_names = [
                    f.get("name", "unknown") for f in functions if isinstance(f, dict)
                ]
                attributes[BrokleInstrumentationAttributes.FUNCTION_NAMES] = ",".join(
                    function_names
                )

        if "tools" in kwargs:
            tools = kwargs["tools"]
            if isinstance(tools, list):
                attributes[BrokleInstrumentationAttributes.TOOL_COUNT] = len(tools)
                tool_types = [
                    t.get("type", "unknown") for t in tools if isinstance(t, dict)
                ]
                attributes[BrokleInstrumentationAttributes.TOOL_TYPES] = ",".join(
                    set(tool_types)
                )

        # Estimate input tokens
        input_tokens = self.estimate_input_tokens(kwargs)
        attributes[BrokleInstrumentationAttributes.INPUT_TOKENS] = input_tokens

        return attributes

    def extract_response_attributes(self, response: Any) -> Dict[str, Any]:
        """Extract OpenAI response attributes for telemetry."""
        attributes = {}

        try:
            # Handle different response types
            if hasattr(response, "usage") and response.usage:
                usage = response.usage

                # Token usage
                if hasattr(usage, "prompt_tokens"):
                    attributes[BrokleInstrumentationAttributes.INPUT_TOKENS] = (
                        usage.prompt_tokens
                    )
                if hasattr(usage, "completion_tokens"):
                    attributes[BrokleInstrumentationAttributes.OUTPUT_TOKENS] = (
                        usage.completion_tokens
                    )
                if hasattr(usage, "total_tokens"):
                    attributes[BrokleInstrumentationAttributes.TOTAL_TOKENS] = (
                        usage.total_tokens
                    )

            # Model from response
            if hasattr(response, "model"):
                attributes[BrokleInstrumentationAttributes.MODEL_NAME_RESPONSE] = (
                    response.model
                )

            # Chat completion specific
            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]

                # Finish reason
                if hasattr(choice, "finish_reason"):
                    attributes[BrokleInstrumentationAttributes.FINISH_REASON] = (
                        choice.finish_reason
                    )

                # Response content
                if hasattr(choice, "message"):
                    message = choice.message
                    if hasattr(message, "content") and message.content:
                        content_length = len(message.content)
                        attributes[
                            BrokleInstrumentationAttributes.RESPONSE_CONTENT_LENGTH
                        ] = content_length

                    # Function calls
                    if hasattr(message, "function_call") and message.function_call:
                        attributes[
                            BrokleInstrumentationAttributes.FUNCTION_CALL_NAME
                        ] = message.function_call.name

                    if hasattr(message, "tool_calls") and message.tool_calls:
                        tool_call_names = [
                            tc.function.name
                            for tc in message.tool_calls
                            if hasattr(tc, "function")
                        ]
                        attributes[BrokleInstrumentationAttributes.TOOL_CALL_NAMES] = (
                            ",".join(tool_call_names)
                        )

                # Legacy completion specific
                elif hasattr(choice, "text"):
                    text_length = len(choice.text) if choice.text else 0
                    attributes[
                        BrokleInstrumentationAttributes.RESPONSE_CONTENT_LENGTH
                    ] = text_length

            # Embedding specific
            if hasattr(response, "data") and isinstance(response.data, list):
                attributes[BrokleInstrumentationAttributes.EMBEDDING_COUNT] = len(
                    response.data
                )
                if response.data and hasattr(response.data[0], "embedding"):
                    attributes[BrokleInstrumentationAttributes.EMBEDDING_DIMENSIONS] = (
                        len(response.data[0].embedding)
                    )

            # Image generation specific
            if hasattr(response, "data") and hasattr(response, "created"):
                # This is likely an image response
                if isinstance(response.data, list):
                    attributes[BrokleInstrumentationAttributes.IMAGE_COUNT] = len(
                        response.data
                    )

        except Exception as e:
            logger.warning(f"Failed to extract OpenAI response attributes: {e}")

        return attributes

    def normalize_model_name(self, model: str) -> str:
        """Normalize OpenAI model names for consistent telemetry."""
        # Remove version suffixes
        normalized = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model)  # Remove date suffixes
        normalized = re.sub(r"-\d+k$", "", normalized)  # Remove context length suffixes
        normalized = re.sub(r"-preview$", "", normalized)  # Remove preview suffix
        normalized = re.sub(r"-\d{4}$", "", normalized)  # Remove year suffixes

        # Map aliases to canonical names
        model_aliases = {
            "gpt-4-turbo-preview": "gpt-4-turbo",
            "gpt-4-0125-preview": "gpt-4-turbo",
            "gpt-4-1106-preview": "gpt-4-turbo",
            "gpt-4-vision-preview": "gpt-4-turbo",
            "gpt-3.5-turbo-0125": "gpt-3.5-turbo",
            "gpt-3.5-turbo-1106": "gpt-3.5-turbo",
        }

        return model_aliases.get(normalized, normalized)

    def get_error_mapping(self) -> Dict[str, str]:
        """Map OpenAI errors to Brokle error types."""
        return {
            "AuthenticationError": "AuthenticationError",
            "PermissionDeniedError": "AuthenticationError",
            "RateLimitError": "RateLimitError",
            "BadRequestError": "ValidationError",
            "InvalidRequestError": "ValidationError",
            "ConflictError": "ValidationError",
            "NotFoundError": "ValidationError",
            "UnprocessableEntityError": "ValidationError",
            "InternalServerError": "ProviderError",
            "APIConnectionError": "ProviderError",
            "APITimeoutError": "ProviderError",
            "APIError": "ProviderError",
        }
