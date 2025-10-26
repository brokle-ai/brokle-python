"""
Base resource classes for Brokle SDK resources.

Provides common functionality for sync and async resource implementations.
"""

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from ..client import AsyncBrokle, Brokle


class BaseResource:
    """Base class for sync resources."""

    def __init__(self, client: "Brokle"):
        """
        Initialize base resource.

        Args:
            client: Sync Brokle client instance
        """
        self._client = client

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make HTTP request via client.

        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Request kwargs

        Returns:
            Response data
        """
        return self._client.request(method, endpoint, **kwargs)


class AsyncBaseResource:
    """Base class for async resources."""

    def __init__(self, client: "AsyncBrokle"):
        """
        Initialize async base resource.

        Args:
            client: Async Brokle client instance
        """
        self._client = client

    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make async HTTP request via client.

        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Request kwargs

        Returns:
            Response data
        """
        return await self._client.request(method, endpoint, **kwargs)
