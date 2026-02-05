"""
Provider configuration registry for LLM SDK wrappers.

Contains configuration for each supported LLM provider, including:
- Parameter mappings
- Response extractors
- Span name generators
- Provider-specific attribute builders

This enables consistent behavior across all providers while allowing
provider-specific customizations.
"""

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type

from ..streaming.wrappers import BrokleAsyncStreamWrapper, BrokleStreamWrapper
from ..types import Attrs, LLMProvider, OperationType, SpanType
from ..utils.attributes import serialize_messages
from ._extractors import (
    ANTHROPIC_PARAM_MAPPING,
    COHERE_PARAM_MAPPING,
    GOOGLE_PARAM_MAPPING,
    MISTRAL_PARAM_MAPPING,
    OPENAI_PARAM_MAPPING,
    build_base_attrs,
    add_messages_attrs,
    extract_params,
    extract_stop_sequences,
    extract_system_messages_openai,
    extract_openai_response,
    extract_anthropic_response,
    extract_mistral_response,
    extract_cohere_response,
    extract_google_response,
    extract_bedrock_response,
    serialize_cohere_connectors,
    serialize_cohere_documents,
    serialize_tools,
)


@dataclass
class ProviderConfig:
    """
    Configuration for a provider wrapper.

    Encapsulates all provider-specific behavior to enable the unified
    factory pattern.
    """

    provider_name: str
    param_mapping: Dict[str, str]
    response_extractor: Callable[[Any, Any, float], None]
    message_builder: Callable[[Dict[str, Any]], Dict[str, Any]]
    span_name_builder: Callable[[Dict[str, Any]], str]
    stream_checker: Callable[[Dict[str, Any]], bool]
    stream_wrapper_class: Optional[Type] = None
    async_stream_wrapper_class: Optional[Type] = None


# =============================================================================
# OpenAI Configuration
# =============================================================================

def build_openai_attrs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Build span attributes for OpenAI requests."""
    model = kwargs.get("model", "unknown")
    messages = kwargs.get("messages", [])
    stream = kwargs.get("stream", False)
    user = kwargs.get("user")

    # Extract system vs non-system messages
    non_system_msgs, system_msgs = extract_system_messages_openai(messages)

    # Build base attributes
    attrs = build_base_attrs(LLMProvider.OPENAI, model, stream)

    # Add messages
    add_messages_attrs(attrs, non_system_msgs, system_msgs)

    # Extract model parameters
    params = extract_params(kwargs, OPENAI_PARAM_MAPPING)
    attrs.update(params)

    # Handle stop sequences
    stop = extract_stop_sequences(kwargs, "stop")
    if stop:
        attrs[Attrs.GEN_AI_REQUEST_STOP_SEQUENCES] = stop

    # User attribution
    if user:
        attrs[Attrs.GEN_AI_REQUEST_USER] = user
        attrs[Attrs.USER_ID] = user

    return attrs


def openai_span_name(kwargs: Dict[str, Any]) -> str:
    """Generate span name for OpenAI requests."""
    model = kwargs.get("model", "unknown")
    return f"{OperationType.CHAT} {model}"


OPENAI_CONFIG = ProviderConfig(
    provider_name=LLMProvider.OPENAI,
    param_mapping=OPENAI_PARAM_MAPPING,
    response_extractor=extract_openai_response,
    message_builder=build_openai_attrs,
    span_name_builder=openai_span_name,
    stream_checker=lambda kw: kw.get("stream", False),
    stream_wrapper_class=BrokleStreamWrapper,
    async_stream_wrapper_class=BrokleAsyncStreamWrapper,
)


# =============================================================================
# Azure OpenAI Configuration
# =============================================================================

def build_azure_openai_attrs(
    kwargs: Dict[str, Any],
    azure_endpoint: Optional[str] = None,
    api_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Build span attributes for Azure OpenAI requests."""
    model = kwargs.get("model", "unknown")  # In Azure, this is deployment name
    messages = kwargs.get("messages", [])
    stream = kwargs.get("stream", False)
    user = kwargs.get("user")

    non_system_msgs, system_msgs = extract_system_messages_openai(messages)

    attrs = build_base_attrs(LLMProvider.AZURE_OPENAI, model, stream)

    # Azure-specific attributes
    attrs[Attrs.AZURE_OPENAI_DEPLOYMENT_NAME] = model
    if azure_endpoint:
        resource_name = azure_endpoint.replace("https://", "").split(".")[0]
        attrs[Attrs.AZURE_OPENAI_RESOURCE_NAME] = resource_name
    if api_version:
        attrs[Attrs.AZURE_OPENAI_API_VERSION] = api_version

    add_messages_attrs(attrs, non_system_msgs, system_msgs)

    # Extract parameters (same as OpenAI)
    params = extract_params(kwargs, OPENAI_PARAM_MAPPING)
    attrs.update(params)

    stop = extract_stop_sequences(kwargs, "stop")
    if stop:
        attrs[Attrs.GEN_AI_REQUEST_STOP_SEQUENCES] = stop

    if user:
        attrs[Attrs.GEN_AI_REQUEST_USER] = user
        attrs[Attrs.USER_ID] = user

    return attrs


def azure_openai_span_name(kwargs: Dict[str, Any]) -> str:
    """Generate span name for Azure OpenAI requests."""
    model = kwargs.get("model", "unknown")
    return f"{OperationType.CHAT} {model}"


AZURE_OPENAI_CONFIG = ProviderConfig(
    provider_name=LLMProvider.AZURE_OPENAI,
    param_mapping=OPENAI_PARAM_MAPPING,
    response_extractor=extract_openai_response,
    message_builder=build_azure_openai_attrs,
    span_name_builder=azure_openai_span_name,
    stream_checker=lambda kw: kw.get("stream", False),
    stream_wrapper_class=BrokleStreamWrapper,
    async_stream_wrapper_class=BrokleAsyncStreamWrapper,
)


# =============================================================================
# Anthropic Configuration
# =============================================================================

def build_anthropic_attrs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Build span attributes for Anthropic requests."""
    model = kwargs.get("model", "unknown")
    messages = kwargs.get("messages", [])
    system = kwargs.get("system")
    stream = kwargs.get("stream", False)

    attrs = build_base_attrs(LLMProvider.ANTHROPIC, model, stream)

    # Process messages - Anthropic separates system from messages
    if messages:
        serialized_msgs = []
        for msg in messages:
            if isinstance(msg, dict):
                serialized_msgs.append(msg)
            elif hasattr(msg, "role"):
                serialized_msgs.append({
                    "role": msg.role,
                    "content": getattr(msg, "content", ""),
                })
        if serialized_msgs:
            attrs[Attrs.GEN_AI_INPUT_MESSAGES] = serialize_messages(serialized_msgs)

    # System message (Anthropic uses separate 'system' param)
    if system:
        if isinstance(system, str):
            attrs[Attrs.GEN_AI_SYSTEM_INSTRUCTIONS] = system
        elif isinstance(system, list):
            # System can be list of content blocks
            attrs[Attrs.GEN_AI_SYSTEM_INSTRUCTIONS] = json.dumps(system)

    # Extract Anthropic-specific parameters
    params = extract_params(kwargs, ANTHROPIC_PARAM_MAPPING)
    attrs.update(params)

    return attrs


def anthropic_span_name(kwargs: Dict[str, Any]) -> str:
    """Generate span name for Anthropic requests."""
    model = kwargs.get("model", "unknown")
    return f"{OperationType.CHAT} {model}"


ANTHROPIC_CONFIG = ProviderConfig(
    provider_name=LLMProvider.ANTHROPIC,
    param_mapping=ANTHROPIC_PARAM_MAPPING,
    response_extractor=extract_anthropic_response,
    message_builder=build_anthropic_attrs,
    span_name_builder=anthropic_span_name,
    stream_checker=lambda kw: kw.get("stream", False),
    stream_wrapper_class=BrokleStreamWrapper,
    async_stream_wrapper_class=BrokleAsyncStreamWrapper,
)


# =============================================================================
# Mistral Configuration
# =============================================================================

def build_mistral_attrs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Build span attributes for Mistral requests."""
    model = kwargs.get("model", "unknown")
    messages = kwargs.get("messages", [])
    stream = kwargs.get("stream", False)

    non_system_msgs, system_msgs = extract_system_messages_openai(messages)

    attrs = build_base_attrs(LLMProvider.MISTRAL, model, stream)
    add_messages_attrs(attrs, non_system_msgs, system_msgs)

    # Extract Mistral-specific parameters
    params = extract_params(kwargs, MISTRAL_PARAM_MAPPING)
    attrs.update(params)

    # Tools
    tools = kwargs.get("tools")
    if tools:
        attrs[Attrs.GEN_AI_REQUEST_TOOLS] = json.dumps(serialize_tools(tools))

    # Tool choice
    tool_choice = kwargs.get("tool_choice")
    if tool_choice:
        if isinstance(tool_choice, str):
            attrs[Attrs.GEN_AI_REQUEST_TOOL_CHOICE] = tool_choice
        elif hasattr(tool_choice, "model_dump"):
            attrs[Attrs.GEN_AI_REQUEST_TOOL_CHOICE] = json.dumps(tool_choice.model_dump())
        elif isinstance(tool_choice, dict):
            attrs[Attrs.GEN_AI_REQUEST_TOOL_CHOICE] = json.dumps(tool_choice)

    return attrs


def mistral_span_name(kwargs: Dict[str, Any]) -> str:
    """Generate span name for Mistral requests."""
    model = kwargs.get("model", "unknown")
    return f"{OperationType.CHAT} {model}"


MISTRAL_CONFIG = ProviderConfig(
    provider_name=LLMProvider.MISTRAL,
    param_mapping=MISTRAL_PARAM_MAPPING,
    response_extractor=extract_mistral_response,
    message_builder=build_mistral_attrs,
    span_name_builder=mistral_span_name,
    stream_checker=lambda kw: kw.get("stream", False),
)


# =============================================================================
# Cohere Configuration
# =============================================================================

def build_cohere_attrs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Build span attributes for Cohere requests."""
    model = kwargs.get("model", "command")
    message = kwargs.get("message", "")
    stream = kwargs.get("stream", False)

    attrs = build_base_attrs(LLMProvider.COHERE, model, stream)

    # Input message
    if message:
        attrs[Attrs.GEN_AI_INPUT_MESSAGES] = serialize_messages([
            {"role": "user", "content": message}
        ])

    # Chat history
    chat_history = kwargs.get("chat_history")
    if chat_history:
        serialized_history = []
        for msg in chat_history:
            if isinstance(msg, dict):
                serialized_history.append(msg)
            elif hasattr(msg, "role"):
                serialized_history.append({
                    "role": msg.role,
                    "content": getattr(msg, "message", ""),
                })
        if serialized_history:
            attrs[Attrs.COHERE_REQUEST_CHAT_HISTORY] = json.dumps(serialized_history)

    # Preamble (system instruction)
    preamble = kwargs.get("preamble")
    if preamble:
        attrs[Attrs.GEN_AI_SYSTEM_INSTRUCTIONS] = preamble

    # Extract parameters
    params = extract_params(kwargs, COHERE_PARAM_MAPPING)
    attrs.update(params)

    # Connectors (RAG)
    connectors = kwargs.get("connectors")
    if connectors:
        attrs[Attrs.COHERE_REQUEST_CONNECTORS] = json.dumps(
            serialize_cohere_connectors(connectors)
        )

    # Documents (RAG)
    documents = kwargs.get("documents")
    if documents:
        attrs[Attrs.COHERE_REQUEST_DOCUMENTS] = json.dumps(
            serialize_cohere_documents(documents)
        )

    # Tools
    tools = kwargs.get("tools")
    if tools:
        attrs[Attrs.GEN_AI_REQUEST_TOOLS] = json.dumps(serialize_tools(tools))

    return attrs


def cohere_span_name(kwargs: Dict[str, Any]) -> str:
    """Generate span name for Cohere requests."""
    model = kwargs.get("model", "command")
    return f"{OperationType.CHAT} {model}"


COHERE_CONFIG = ProviderConfig(
    provider_name=LLMProvider.COHERE,
    param_mapping=COHERE_PARAM_MAPPING,
    response_extractor=extract_cohere_response,
    message_builder=build_cohere_attrs,
    span_name_builder=cohere_span_name,
    stream_checker=lambda kw: kw.get("stream", False),
)


# =============================================================================
# Google GenAI Configuration
# =============================================================================

def build_google_attrs(
    kwargs: Dict[str, Any],
    model_name: str = "unknown",
) -> Dict[str, Any]:
    """Build span attributes for Google GenAI requests."""
    # Google uses positional 'contents' argument
    contents = kwargs.get("contents", "")
    stream = kwargs.get("stream", False)

    attrs = build_base_attrs(LLMProvider.GOOGLE, model_name, stream)

    # Process contents
    if contents:
        if isinstance(contents, str):
            attrs[Attrs.GEN_AI_INPUT_MESSAGES] = serialize_messages([
                {"role": "user", "content": contents}
            ])
        elif isinstance(contents, list):
            serialized = []
            for item in contents:
                if isinstance(item, str):
                    serialized.append({"role": "user", "content": item})
                elif hasattr(item, "role"):
                    parts_text = ""
                    if hasattr(item, "parts"):
                        parts_text = "".join(
                            p.text for p in item.parts if hasattr(p, "text")
                        )
                    serialized.append({"role": item.role, "content": parts_text})
            if serialized:
                attrs[Attrs.GEN_AI_INPUT_MESSAGES] = serialize_messages(serialized)

    # Generation config (contains model parameters)
    generation_config = kwargs.get("generation_config")
    if generation_config:
        if hasattr(generation_config, "__dict__"):
            config_dict = generation_config.__dict__
        elif isinstance(generation_config, dict):
            config_dict = generation_config
        else:
            config_dict = {}

        params = extract_params(config_dict, GOOGLE_PARAM_MAPPING)
        attrs.update(params)

    # System instruction
    system_instruction = kwargs.get("system_instruction")
    if system_instruction:
        if isinstance(system_instruction, str):
            attrs[Attrs.GEN_AI_SYSTEM_INSTRUCTIONS] = system_instruction
        elif hasattr(system_instruction, "parts"):
            text = "".join(
                p.text for p in system_instruction.parts if hasattr(p, "text")
            )
            if text:
                attrs[Attrs.GEN_AI_SYSTEM_INSTRUCTIONS] = text

    return attrs


def google_span_name(kwargs: Dict[str, Any], model_name: str = "unknown") -> str:
    """Generate span name for Google GenAI requests."""
    return f"{OperationType.CHAT} {model_name}"


GOOGLE_CONFIG = ProviderConfig(
    provider_name=LLMProvider.GOOGLE,
    param_mapping=GOOGLE_PARAM_MAPPING,
    response_extractor=extract_google_response,
    message_builder=build_google_attrs,
    span_name_builder=lambda kw: google_span_name(kw),
    stream_checker=lambda kw: kw.get("stream", False),
)


# =============================================================================
# Bedrock Configuration
# =============================================================================

def build_bedrock_attrs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Build span attributes for Bedrock requests."""
    model_id = kwargs.get("modelId", "unknown")
    messages = kwargs.get("messages", [])
    system = kwargs.get("system")
    inference_config = kwargs.get("inferenceConfig", {})

    attrs = build_base_attrs(LLMProvider.BEDROCK, model_id, False)
    attrs[Attrs.BEDROCK_REQUEST_MODEL_ID] = model_id

    # Process messages (Bedrock uses list of dicts)
    if messages:
        serialized_msgs = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content_list = msg.get("content", [])
                text_parts = []
                for content in content_list:
                    if isinstance(content, dict) and "text" in content:
                        text_parts.append(content["text"])
                serialized_msgs.append({
                    "role": role,
                    "content": "".join(text_parts),
                })
        if serialized_msgs:
            attrs[Attrs.GEN_AI_INPUT_MESSAGES] = serialize_messages(serialized_msgs)

    # System prompts
    if system:
        system_text = []
        for sys_msg in system:
            if isinstance(sys_msg, dict) and "text" in sys_msg:
                system_text.append(sys_msg["text"])
        if system_text:
            attrs[Attrs.GEN_AI_SYSTEM_INSTRUCTIONS] = "\n".join(system_text)

    # Inference config
    if inference_config:
        if "temperature" in inference_config:
            attrs[Attrs.GEN_AI_REQUEST_TEMPERATURE] = inference_config["temperature"]
        if "maxTokens" in inference_config:
            attrs[Attrs.GEN_AI_REQUEST_MAX_TOKENS] = inference_config["maxTokens"]
        if "topP" in inference_config:
            attrs[Attrs.GEN_AI_REQUEST_TOP_P] = inference_config["topP"]
        if "stopSequences" in inference_config:
            attrs[Attrs.GEN_AI_REQUEST_STOP_SEQUENCES] = inference_config["stopSequences"]

    # Tool config
    tool_config = kwargs.get("toolConfig")
    if tool_config and "tools" in tool_config:
        attrs[Attrs.GEN_AI_REQUEST_TOOLS] = json.dumps(tool_config["tools"])

    # Guardrails
    guardrail_config = kwargs.get("guardrailConfig")
    if guardrail_config:
        attrs[Attrs.BEDROCK_REQUEST_GUARDRAIL_CONFIG] = json.dumps(guardrail_config)

    return attrs


def bedrock_span_name(kwargs: Dict[str, Any]) -> str:
    """Generate span name for Bedrock requests."""
    model_id = kwargs.get("modelId", "unknown")
    return f"{OperationType.CHAT} {model_id}"


BEDROCK_CONFIG = ProviderConfig(
    provider_name=LLMProvider.BEDROCK,
    param_mapping={},  # Bedrock uses nested config, handled in build_bedrock_attrs
    response_extractor=extract_bedrock_response,
    message_builder=build_bedrock_attrs,
    span_name_builder=bedrock_span_name,
    stream_checker=lambda kw: False,  # Bedrock has separate converse_stream method
)


# =============================================================================
# Provider Registry
# =============================================================================

PROVIDER_CONFIGS: Dict[str, ProviderConfig] = {
    LLMProvider.OPENAI: OPENAI_CONFIG,
    LLMProvider.AZURE_OPENAI: AZURE_OPENAI_CONFIG,
    LLMProvider.ANTHROPIC: ANTHROPIC_CONFIG,
    LLMProvider.MISTRAL: MISTRAL_CONFIG,
    LLMProvider.COHERE: COHERE_CONFIG,
    LLMProvider.GOOGLE: GOOGLE_CONFIG,
    LLMProvider.BEDROCK: BEDROCK_CONFIG,
}


def get_provider_config(provider: str) -> ProviderConfig:
    """
    Get configuration for a provider.

    Args:
        provider: Provider name (e.g., "openai", "anthropic")

    Returns:
        ProviderConfig for the specified provider

    Raises:
        ValueError: If provider is not supported
    """
    config = PROVIDER_CONFIGS.get(provider)
    if not config:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported: {list(PROVIDER_CONFIGS.keys())}"
        )
    return config
