"""
New Brokle clients with OpenAI-compatible interface.

Provides sync and async clients that mirror OpenAI SDK structure
while adding Brokle-specific features.
"""

from typing import Optional, Any, Dict, List
import httpx

from .http.base import HTTPBase
from .exceptions import NetworkError


class Brokle(HTTPBase):
    """
    Sync Brokle client with OpenAI-compatible interface.

    Usage:
        with Brokle(api_key="ak_...", host="http://localhost:8080") as client:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello!"}],
                routing_strategy="cost_optimized"  # Brokle extension
            )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        project_id: Optional[str] = None,
        environment: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize sync Brokle client.

        Args:
            api_key: Brokle API key
            host: Brokle host URL
            project_id: Project ID
            environment: Environment name
            timeout: Request timeout in seconds
            **kwargs: Additional configuration
        """
        super().__init__(
            api_key=api_key,
            host=host,
            project_id=project_id,
            environment=environment,
            timeout=timeout,
            **kwargs
        )

        # Initialize HTTP client
        self._client: Optional[httpx.Client] = None

        # Initialize resources (will be created in next step)
        from .resources.chat import ChatResource
        from .resources.embeddings import EmbeddingsResource
        from .resources.models import ModelsResource

        self.chat = ChatResource(self)
        self.embeddings = EmbeddingsResource(self)
        self.models = ModelsResource(self)

    def span(self, name: str, **kwargs):
        """Create a span for observability."""
        from .observability.spans import create_span
        return create_span(name=name, **kwargs)

    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                timeout=httpx.Timeout(self.config.timeout),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
            )
        return self._client

    def request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make HTTP request to Brokle backend.

        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Request kwargs

        Returns:
            Response data

        Raises:
            NetworkError: For connection errors
        """
        client = self._get_client()
        url = self._prepare_url(endpoint)
        kwargs = self._prepare_request_kwargs(**kwargs)

        try:
            response = client.request(method, url, **kwargs)
            return self._process_response(response)
        except httpx.ConnectError as e:
            raise NetworkError(f"Failed to connect to Brokle backend: {e}")
        except httpx.TimeoutException as e:
            raise NetworkError(f"Request timeout: {e}")
        except httpx.HTTPError as e:
            raise NetworkError(f"HTTP error: {e}")

    def close(self) -> None:
        """Close HTTP client and cleanup resources."""
        if hasattr(self, '_client') and self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()

    def __del__(self):
        """Cleanup on deletion (fallback)."""
        self.close()


class AsyncBrokle(HTTPBase):
    """
    Async Brokle client with OpenAI-compatible interface.

    Usage:
        client = AsyncBrokle(api_key="ak_...")
        try:
            response = await client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello!"}],
                routing_strategy="cost_optimized"
            )
        finally:
            await client.close()
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        project_id: Optional[str] = None,
        environment: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize async Brokle client.

        Args:
            api_key: Brokle API key
            host: Brokle host URL
            project_id: Project ID
            environment: Environment name
            timeout: Request timeout in seconds
            **kwargs: Additional configuration
        """
        super().__init__(
            api_key=api_key,
            host=host,
            project_id=project_id,
            environment=environment,
            timeout=timeout,
            **kwargs
        )

        # Initialize persistent HTTP client (performance optimization)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.timeout),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
        )

        # Initialize async resources (will be created in next step)
        from .resources.chat import AsyncChatResource
        from .resources.embeddings import AsyncEmbeddingsResource
        from .resources.models import AsyncModelsResource

        self.chat = AsyncChatResource(self)
        self.embeddings = AsyncEmbeddingsResource(self)
        self.models = AsyncModelsResource(self)

    async def request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make async HTTP request to Brokle backend.

        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Request kwargs

        Returns:
            Response data

        Raises:
            NetworkError: For connection errors
        """
        url = self._prepare_url(endpoint)
        kwargs = self._prepare_request_kwargs(**kwargs)

        try:
            response = await self._client.request(method, url, **kwargs)
            return self._process_response(response)
        except httpx.ConnectError as e:
            raise NetworkError(f"Failed to connect to Brokle backend: {e}")
        except httpx.TimeoutException as e:
            raise NetworkError(f"Request timeout: {e}")
        except httpx.HTTPError as e:
            raise NetworkError(f"HTTP error: {e}")

    async def close(self) -> None:
        """Close async HTTP client and cleanup resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.close()


# Global singleton for clean architecture
_client_singleton: Optional[Brokle] = None

def get_client() -> Brokle:
    """
    Get or create a singleton Brokle client instance from environment variables.

    This is the clean API for Pattern 1/2/3 integration:
    - Pattern 1 (Wrappers): Use this for observability context
    - Pattern 2 (Decorator): Use this for telemetry
    - Pattern 3 (Native): Use Brokle() for explicit config, get_client() for env config

    Configuration is read from environment variables:
    - BROKLE_API_KEY
    - BROKLE_HOST
    - BROKLE_PROJECT_ID
    - BROKLE_ENVIRONMENT
    - BROKLE_OTEL_ENABLED
    - etc.

    Returns:
        Singleton Brokle client instance

    Example:
        ```python
        # For explicit configuration, use Brokle() directly
        client = Brokle(api_key="ak_your_key", project_id="proj_123")

        # For environment-based configuration, use get_client()
        client = get_client()  # Reads from BROKLE_* env vars
        ```
    """
    global _client_singleton

    if _client_singleton is None:
        # Create singleton from environment variables
        _client_singleton = Brokle()

    return _client_singleton