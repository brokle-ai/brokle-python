"""
HTTP Base class for Brokle SDK.

Provides shared HTTP functionality for both sync and async clients.
Centralizes auth, headers, and request preparation logic.
"""

import time
from typing import Any, Dict, Optional, Union
from urllib.parse import urljoin

import httpx
from pydantic import BaseModel

from .._version import __version__
from ..config import Config
from ..exceptions import APIError, AuthenticationError, NetworkError


class HTTPBase:
    """
    Shared HTTP base class for Brokle clients.

    Centralizes:
    - Authentication and headers
    - Request preparation
    - Error handling
    - Configuration management
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        environment: Optional[str] = None,
        timeout: Optional[float] = None,
        config: Optional[Config] = None,
        **kwargs,
    ):
        """
        Initialize HTTP base with configuration.

        Args:
            api_key: Brokle API key (bk_...)
            host: Brokle host URL
            environment: Environment name
            timeout: Request timeout in seconds
            config: Pre-configured Config object (overrides individual parameters)
            **kwargs: Additional configuration
        """
        # If config object provided, use it directly
        if config is not None:
            self.config = config
        elif (
            api_key is None
            and host is None
            and environment is None
            and timeout is None
            and not kwargs
        ):
            # If no parameters provided, use environment variables
            self.config = Config.from_env()
        else:
            # Override specific parameters while falling back to env vars
            from_env = Config.from_env()
            self.config = Config(
                api_key=api_key or from_env.api_key,
                host=host or from_env.host,
                environment=environment or from_env.environment,
                timeout=timeout or from_env.timeout,
                **kwargs,
            )

        # Handle missing API key with warning
        if not self.config.api_key:
            import logging
            logger = logging.getLogger("brokle")
            logger.warning(
                "Authentication error: Brokle client initialized without api_key. "
                "Client will be disabled. Provide an api_key parameter or set BROKLE_API_KEY environment variable."
            )
            self._disabled = True
        else:
            self._disabled = False

        # Build default headers (safe headers even when disabled)
        self.default_headers = self._build_headers()

    def _build_headers(self) -> Dict[str, str]:
        """
        Build default headers for all requests.

        Returns:
            Dictionary of headers
        """
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"brokle-python/{__version__}",
            "X-SDK-Version": __version__,
        }

        # Only add auth headers if we have an API key
        if self.config.api_key:
            headers["X-API-Key"] = self.config.api_key

        # Always include environment (falls back to "default")
        headers["X-Environment"] = self.config.environment

        return headers

    def _prepare_url(self, endpoint: str) -> str:
        """
        Prepare full URL for endpoint.

        Args:
            endpoint: API endpoint (e.g., '/v1/chat/completions')

        Returns:
            Full URL
        """
        return urljoin(self.config.host.rstrip("/") + "/", endpoint.lstrip("/"))

    def _prepare_request_kwargs(self, **kwargs) -> Dict[str, Any]:
        """
        Prepare request kwargs with default headers and timeout.

        Args:
            **kwargs: Request kwargs

        Returns:
            Prepared request kwargs
        """
        # Start with default headers
        headers = self.default_headers.copy()

        # Merge with provided headers
        if "headers" in kwargs:
            headers.update(kwargs["headers"])
        kwargs["headers"] = headers

        # Set timeout if not provided
        if "timeout" not in kwargs:
            kwargs["timeout"] = self.config.timeout

        # Add timestamp for debugging
        kwargs.setdefault("headers", {})["X-Request-Timestamp"] = str(int(time.time()))

        return kwargs

    def _handle_http_error(self, response: httpx.Response) -> None:
        """
        Handle HTTP errors and convert to appropriate exceptions.

        Args:
            response: HTTP response

        Raises:
            AuthenticationError: For 401 errors
            APIError: For other HTTP errors
        """
        if response.status_code == 401:
            raise AuthenticationError(f"Authentication failed: {response.text}")
        elif response.status_code == 429:
            raise APIError(f"Rate limit exceeded: {response.text}")
        elif response.status_code >= 400:
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", response.text)
            except Exception:
                error_msg = response.text

            raise APIError(f"API error ({response.status_code}): {error_msg}")

    def _process_response(self, response: httpx.Response) -> Dict[str, Any]:
        """
        Process HTTP response and return JSON data.

        Args:
            response: HTTP response

        Returns:
            Response JSON data

        Raises:
            APIError: For HTTP errors or invalid JSON
        """
        # Handle HTTP errors
        if response.status_code >= 400:
            self._handle_http_error(response)

        # Parse JSON response
        try:
            return response.json()
        except Exception as e:
            raise APIError(f"Failed to parse response JSON: {e}")


class BrokleResponse(BaseModel):
    """
    Base response model with Brokle metadata.
    """

    class BrokleMetadata(BaseModel):
        """Brokle-specific response metadata."""

        provider: str
        request_id: str
        latency_ms: int
        cost_usd: Optional[float] = None
        tokens_used: Optional[int] = None
        cache_hit: bool = False
        cache_key: Optional[str] = None
        routing_strategy: Optional[str] = None
        quality_score: Optional[float] = None

    brokle: Optional[BrokleMetadata] = None
