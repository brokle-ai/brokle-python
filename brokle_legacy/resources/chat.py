"""
Chat resources for Brokle SDK.

Provides OpenAI-compatible chat completions interface with Brokle extensions.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from .base import AsyncBaseResource, BaseResource


class Message(BaseModel):
    """Chat message model."""

    role: str
    content: str
    name: Optional[str] = None


class Choice(BaseModel):
    """Chat completion choice."""

    index: int
    message: Message
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Chat completion response with Brokle metadata."""

    class BrokleMetadata(BaseModel):
        """Brokle-specific metadata."""

        provider: str
        request_id: str
        latency_ms: int
        cost_usd: Optional[float] = None
        tokens_used: Optional[int] = None
        cache_hit: bool = False
        cache_key: Optional[str] = None
        routing_strategy: Optional[str] = None
        quality_score: Optional[float] = None

    # Standard OpenAI fields
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

    # Brokle platform metadata (industry standard pattern)
    brokle: Optional[BrokleMetadata] = None


class ChatCompletions(BaseResource):
    """Sync chat completions resource."""

    def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: Optional[bool] = None,
        # Brokle extensions
        routing_strategy: Optional[str] = None,
        cache_strategy: Optional[str] = None,
        environment: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **kwargs,
    ) -> ChatCompletionResponse:
        """
        Create chat completion with OpenAI-compatible interface.

        Args:
            model: Model name
            messages: List of messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            stop: Stop sequences
            stream: Whether to stream response
            routing_strategy: Brokle routing strategy (e.g., "cost_optimized")
            cache_strategy: Brokle cache strategy (e.g., "semantic")
            environment: Environment override
            tags: Request tags
            **kwargs: Additional parameters

        Returns:
            Chat completion response with Brokle metadata
        """
        # Return None if client is disabled
        if self._client.is_disabled:
            return None

        # Prepare request data
        data = {
            "model": model,
            "messages": messages,
        }

        # Add OpenAI parameters if provided
        if temperature is not None:
            data["temperature"] = temperature
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        if top_p is not None:
            data["top_p"] = top_p
        if frequency_penalty is not None:
            data["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            data["presence_penalty"] = presence_penalty
        if stop is not None:
            data["stop"] = stop
        if stream is not None:
            data["stream"] = stream

        # Add Brokle extensions
        brokle_params = {}
        if routing_strategy is not None:
            brokle_params["routing_strategy"] = routing_strategy
        if cache_strategy is not None:
            brokle_params["cache_strategy"] = cache_strategy
        if environment is not None:
            brokle_params["environment"] = environment
        if tags is not None:
            brokle_params["tags"] = tags

        if brokle_params:
            data["brokle"] = brokle_params

        # Add any additional kwargs
        data.update(kwargs)

        # Make request to Go backend
        response_data = self._request("POST", "/v1/chat/completions", json=data)

        # Return typed response
        return ChatCompletionResponse(**response_data)


class AsyncChatCompletions(AsyncBaseResource):
    """Async chat completions resource."""

    async def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: Optional[bool] = None,
        # Brokle extensions
        routing_strategy: Optional[str] = None,
        cache_strategy: Optional[str] = None,
        environment: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **kwargs,
    ) -> ChatCompletionResponse:
        """
        Create async chat completion with OpenAI-compatible interface.

        Args:
            model: Model name
            messages: List of messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            stop: Stop sequences
            stream: Whether to stream response
            routing_strategy: Brokle routing strategy
            cache_strategy: Brokle cache strategy
            environment: Environment override
            tags: Request tags
            **kwargs: Additional parameters

        Returns:
            Chat completion response with Brokle metadata
        """
        # Return None if client is disabled
        if self._client.is_disabled:
            return None

        # Prepare request data (same as sync version)
        data = {
            "model": model,
            "messages": messages,
        }

        # Add OpenAI parameters if provided
        if temperature is not None:
            data["temperature"] = temperature
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        if top_p is not None:
            data["top_p"] = top_p
        if frequency_penalty is not None:
            data["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            data["presence_penalty"] = presence_penalty
        if stop is not None:
            data["stop"] = stop
        if stream is not None:
            data["stream"] = stream

        # Add Brokle extensions
        brokle_params = {}
        if routing_strategy is not None:
            brokle_params["routing_strategy"] = routing_strategy
        if cache_strategy is not None:
            brokle_params["cache_strategy"] = cache_strategy
        if environment is not None:
            brokle_params["environment"] = environment
        if tags is not None:
            brokle_params["tags"] = tags

        if brokle_params:
            data["brokle"] = brokle_params

        # Add any additional kwargs
        data.update(kwargs)

        # Make async request to Go backend
        response_data = await self._request("POST", "/v1/chat/completions", json=data)

        # Return typed response
        return ChatCompletionResponse(**response_data)


class ChatResource(BaseResource):
    """Sync chat resource with completions."""

    def __init__(self, client):
        super().__init__(client)
        self.completions = ChatCompletions(client)


class AsyncChatResource(AsyncBaseResource):
    """Async chat resource with completions."""

    def __init__(self, client):
        super().__init__(client)
        self.completions = AsyncChatCompletions(client)
