"""
HTTP Clients

Provides both synchronous and asynchronous HTTP clients for Brokle API communication.

Architecture:
- SyncHTTPClient: Uses httpx.Client for sync operations (no event loop)
- AsyncHTTPClient: Uses httpx.AsyncClient for async operations

This design eliminates event loop lifecycle issues that occur when trying to
bridge sync code to async code via asyncio.run().
"""

from typing import Any, Dict, Optional

import httpx


def unwrap_response(
    response: Dict[str, Any],
    resource_type: str,
    identifier: Optional[str] = None,
) -> Any:
    """
    Unwrap Brokle API envelope.

    Args:
        response: API response
        resource_type: Expected resource type
        identifier: Optional identifier for error messages

    Returns:
        Unwrapped data from response["data"]

    Raises:
        ValueError: If response format is invalid
        KeyError: If required fields are missing
    """
    if not response.get("success"):
        error = response.get("error", {})
        error_msg = error.get("message", "Unknown error")
        if identifier:
            raise ValueError(f"{resource_type} '{identifier}': {error_msg}")
        raise ValueError(f"{resource_type}: {error_msg}")

    return response["data"]


class SyncHTTPClient:
    """
    Synchronous HTTP client for Brokle API.

    Uses httpx.Client (sync) - no event loop involvement.
    This is the correct approach for sync operations.
    """

    def __init__(self, config):
        """
        Initialize sync HTTP client.

        Args:
            config: BrokleConfig instance
        """
        self._config = config
        self._client: Optional[httpx.Client] = None

    def _get_client(self) -> httpx.Client:
        """Get or create httpx sync client."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self._config.base_url,
                timeout=self._config.timeout,
                headers={
                    "X-API-Key": self._config.api_key,
                    "Content-Type": "application/json",
                },
            )
        return self._client

    def get(
        self, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send sync GET request.

        Args:
            path: API path (e.g., "/v1/prompts/greeting")
            params: Optional query parameters

        Returns:
            Response JSON (always returns JSON, even for error responses)
        """
        response = self._get_client().get(path, params=params)
        return response.json()

    def post(
        self, path: str, json: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send sync POST request.

        Args:
            path: API path
            json: Request body

        Returns:
            Response JSON (always returns JSON, even for error responses)
        """
        response = self._get_client().post(path, json=json)
        return response.json()

    def close(self):
        """Close sync HTTP client."""
        if self._client:
            self._client.close()
            self._client = None


class AsyncHTTPClient:
    """
    Asynchronous HTTP client for Brokle API.

    Uses httpx.AsyncClient - requires async context.
    Uses the caller's event loop, never creates its own.
    """

    def __init__(self, config):
        """
        Initialize async HTTP client.

        Args:
            config: BrokleConfig instance
        """
        self._config = config
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create httpx async client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._config.base_url,
                timeout=self._config.timeout,
                headers={
                    "X-API-Key": self._config.api_key,
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def get(
        self, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send async GET request.

        Args:
            path: API path (e.g., "/v1/prompts/greeting")
            params: Optional query parameters

        Returns:
            Response JSON (always returns JSON, even for error responses)
        """
        response = await self._get_client().get(path, params=params)
        return response.json()

    async def post(
        self, path: str, json: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send async POST request.

        Args:
            path: API path
            json: Request body

        Returns:
            Response JSON (always returns JSON, even for error responses)
        """
        response = await self._get_client().post(path, json=json)
        return response.json()

    async def close(self):
        """Close async HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
