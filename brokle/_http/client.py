"""
HTTP Client

Async HTTP client for Brokle API communication.
"""

from typing import Any, Dict, Optional

import httpx


class AsyncHTTPClient:
    """
    Async HTTP client for Brokle API.

    Wraps httpx.AsyncClient with Brokle-specific authentication and error handling.
    """

    def __init__(self, config):
        """
        Initialize HTTP client.

        Args:
            config: BrokleConfig instance
        """
        self._config = config
        self._session: Optional[httpx.AsyncClient] = None

    def _get_session(self) -> httpx.AsyncClient:
        """Get or create httpx session."""
        if self._session is None:
            self._session = httpx.AsyncClient(
                base_url=self._config.base_url,
                timeout=self._config.timeout,
                headers={
                    "X-API-Key": self._config.api_key,
                    "Content-Type": "application/json",
                },
            )
        return self._session

    async def get(
        self, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send GET request.

        Args:
            path: API path (e.g., "/v1/prompts/greeting")
            params: Optional query parameters

        Returns:
            Response JSON

        Raises:
            httpx.HTTPError: On request failure
        """
        session = self._get_session()
        response = await session.get(path, params=params)
        response.raise_for_status()
        return response.json()

    async def post(
        self, path: str, json: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send POST request.

        Args:
            path: API path
            json: Request body

        Returns:
            Response JSON

        Raises:
            httpx.HTTPError: On request failure
        """
        session = self._get_session()
        response = await session.post(path, json=json)
        response.raise_for_status()
        return response.json()

    async def close(self):
        """Close HTTP session."""
        if self._session:
            await self._session.aclose()
            self._session = None

    @staticmethod
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
