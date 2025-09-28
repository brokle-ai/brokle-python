"""
Embeddings resources for Brokle SDK.

Provides OpenAI-compatible embeddings interface with Brokle extensions.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from .base import AsyncBaseResource, BaseResource


class EmbeddingData(BaseModel):
    """Embedding data point."""

    object: str = "embedding"
    index: int
    embedding: List[float]


class EmbeddingUsage(BaseModel):
    """Embedding usage information."""

    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    """Embedding response with Brokle metadata."""

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

    # Standard OpenAI fields
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage

    # Brokle platform metadata (industry standard pattern)
    brokle: Optional[BrokleMetadata] = None


class EmbeddingsResource(BaseResource):
    """Sync embeddings resource."""

    def create(
        self,
        *,
        input: Union[str, List[str]],
        model: str,
        encoding_format: Optional[str] = None,
        dimensions: Optional[int] = None,
        user: Optional[str] = None,
        # Brokle extensions
        routing_strategy: Optional[str] = None,
        cache_strategy: Optional[str] = None,
        environment: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **kwargs,
    ) -> EmbeddingResponse:
        """
        Create embeddings with OpenAI-compatible interface.

        Args:
            input: Input text(s) to embed
            model: Model name
            encoding_format: Encoding format
            dimensions: Number of dimensions
            user: User identifier
            routing_strategy: Brokle routing strategy
            cache_strategy: Brokle cache strategy
            environment: Environment override
            tags: Request tags
            **kwargs: Additional parameters

        Returns:
            Embedding response with Brokle metadata
        """
        # Return None if client is disabled
        if self._client.is_disabled:
            return None

        # Prepare request data
        data = {
            "input": input,
            "model": model,
        }

        # Add OpenAI parameters if provided
        if encoding_format is not None:
            data["encoding_format"] = encoding_format
        if dimensions is not None:
            data["dimensions"] = dimensions
        if user is not None:
            data["user"] = user

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
        response_data = self._request("POST", "/v1/embeddings", json=data)

        # Return typed response
        return EmbeddingResponse(**response_data)


class AsyncEmbeddingsResource(AsyncBaseResource):
    """Async embeddings resource."""

    async def create(
        self,
        *,
        input: Union[str, List[str]],
        model: str,
        encoding_format: Optional[str] = None,
        dimensions: Optional[int] = None,
        user: Optional[str] = None,
        # Brokle extensions
        routing_strategy: Optional[str] = None,
        cache_strategy: Optional[str] = None,
        environment: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **kwargs,
    ) -> EmbeddingResponse:
        """
        Create async embeddings with OpenAI-compatible interface.

        Args:
            input: Input text(s) to embed
            model: Model name
            encoding_format: Encoding format
            dimensions: Number of dimensions
            user: User identifier
            routing_strategy: Brokle routing strategy
            cache_strategy: Brokle cache strategy
            environment: Environment override
            tags: Request tags
            **kwargs: Additional parameters

        Returns:
            Embedding response with Brokle metadata
        """
        # Return None if client is disabled
        if self._client.is_disabled:
            return None

        # Prepare request data (same as sync version)
        data = {
            "input": input,
            "model": model,
        }

        # Add OpenAI parameters if provided
        if encoding_format is not None:
            data["encoding_format"] = encoding_format
        if dimensions is not None:
            data["dimensions"] = dimensions
        if user is not None:
            data["user"] = user

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
        response_data = await self._request("POST", "/v1/embeddings", json=data)

        # Return typed response
        return EmbeddingResponse(**response_data)
