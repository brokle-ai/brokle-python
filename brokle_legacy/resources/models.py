"""
Models resources for Brokle SDK.

Provides OpenAI-compatible models interface with Brokle extensions.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from .base import AsyncBaseResource, BaseResource


class ModelInfo(BaseModel):
    """Model information."""

    id: str
    object: str = "model"
    created: int
    owned_by: str
    permission: Optional[List[Dict[str, Any]]] = None

    # Brokle extensions
    provider: Optional[str] = None
    category: Optional[str] = None  # "chat", "embedding", "completion"
    capabilities: Optional[List[str]] = None
    cost_per_token: Optional[float] = None
    context_length: Optional[int] = None
    availability: Optional[str] = None  # "available", "limited", "unavailable"


class ModelsResponse(BaseModel):
    """Models list response."""

    object: str = "list"
    data: List[ModelInfo]


class ModelsResource(BaseResource):
    """Sync models resource."""

    def list(
        self,
        *,
        provider: Optional[str] = None,
        category: Optional[str] = None,
        available_only: bool = True,
        **kwargs,
    ) -> ModelsResponse:
        """
        List available models with OpenAI-compatible interface.

        Args:
            provider: Filter by provider (e.g., "openai", "anthropic")
            category: Filter by category ("chat", "embedding", "completion")
            available_only: Only include available models
            **kwargs: Additional parameters

        Returns:
            Models response
        """
        # Return None if client is disabled
        if self._client.is_disabled:
            return None

        # Prepare query parameters
        params = {}
        if provider is not None:
            params["provider"] = provider
        if category is not None:
            params["category"] = category
        if not available_only:
            params["include_unavailable"] = "true"

        # Add any additional kwargs
        params.update(kwargs)

        # Make request to Go backend
        response_data = self._request("GET", "/v1/models", params=params)

        # Return typed response
        return ModelsResponse(**response_data)

    def retrieve(self, model_id: str, **kwargs) -> ModelInfo:
        """
        Retrieve specific model information.

        Args:
            model_id: Model ID
            **kwargs: Additional parameters

        Returns:
            Model information
        """
        # Return None if client is disabled
        if self._client.is_disabled:
            return None

        # Make request to Go backend
        response_data = self._request("GET", f"/v1/models/{model_id}", params=kwargs)

        # Return typed response
        return ModelInfo(**response_data)


class AsyncModelsResource(AsyncBaseResource):
    """Async models resource."""

    async def list(
        self,
        *,
        provider: Optional[str] = None,
        category: Optional[str] = None,
        available_only: bool = True,
        **kwargs,
    ) -> ModelsResponse:
        """
        List available models with OpenAI-compatible interface (async).

        Args:
            provider: Filter by provider
            category: Filter by category
            available_only: Only include available models
            **kwargs: Additional parameters

        Returns:
            Models response
        """
        # Return None if client is disabled
        if self._client.is_disabled:
            return None

        # Prepare query parameters
        params = {}
        if provider is not None:
            params["provider"] = provider
        if category is not None:
            params["category"] = category
        if not available_only:
            params["include_unavailable"] = "true"

        # Add any additional kwargs
        params.update(kwargs)

        # Make async request to Go backend
        response_data = await self._request("GET", "/v1/models", params=params)

        # Return typed response
        return ModelsResponse(**response_data)

    async def retrieve(self, model_id: str, **kwargs) -> ModelInfo:
        """
        Retrieve specific model information (async).

        Args:
            model_id: Model ID
            **kwargs: Additional parameters

        Returns:
            Model information
        """
        # Return None if client is disabled
        if self._client.is_disabled:
            return None

        # Make async request to Go backend
        response_data = await self._request(
            "GET", f"/v1/models/{model_id}", params=kwargs
        )

        # Return typed response
        return ModelInfo(**response_data)
