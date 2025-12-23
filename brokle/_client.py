"""
Brokle Client

Provides both synchronous and asynchronous clients for Brokle's
OpenTelemetry-native LLM observability platform.

Core: Telemetry (traces, spans, metrics, logs)
Features: Prompts, datasets, and scores APIs

Sync Usage:
    >>> from brokle import Brokle
    >>> with Brokle(api_key="bk_...") as client:
    ...     # Core: Create traces and spans
    ...     with client.start_as_current_span("my-operation") as span:
    ...         span.set_attribute("output", "Hello, world!")
    ...
    ...     # Feature: Prompt management (optional)
    ...     prompt = client.prompts.get("greeting", label="production")

Async Usage:
    >>> from brokle import AsyncBrokle
    >>> async with AsyncBrokle(api_key="bk_...") as client:
    ...     # Core: Create traces and spans
    ...     with client.start_as_current_span("process") as span:
    ...         result = await do_work()
    ...
    ...     # Feature: Prompt management (optional)
    ...     prompt = await client.prompts.get("greeting")

Singleton Pattern:
    >>> from brokle import get_client
    >>> client = get_client()  # Reads from BROKLE_* env vars
"""

from typing import Optional

from ._base_client import BaseBrokleClient
from ._http import AsyncHTTPClient, SyncHTTPClient
from .config import BrokleConfig
from .datasets import AsyncDatasetsManager, DatasetsManager
from .prompts import AsyncPromptManager, PromptManager
from .scores import AsyncScoresManager, ScoresManager


class Brokle(BaseBrokleClient):
    """
    Synchronous Brokle client for OpenTelemetry-native LLM observability.

    Core responsibility: Telemetry (traces, spans, metrics, logs)
    Feature APIs: Prompts, datasets, and scores

    This client provides synchronous methods for all operations.
    Uses SyncHTTPClient (httpx.Client) internally - no event loop involvement.

    Example:
        >>> from brokle import Brokle
        >>>
        >>> # Context manager (recommended)
        >>> with Brokle(api_key="bk_...") as client:
        ...     # Core: Telemetry - traces and spans
        ...     with client.start_as_current_span("process") as span:
        ...         result = do_work()
        ...         span.set_attribute("result", result)
        ...
        ...     # Feature: Prompt management (optional)
        ...     prompt = client.prompts.get("greeting")
        ...     messages = prompt.to_openai_messages({"name": "Alice"})
    """

    def __init__(self, *args, **kwargs):
        """Initialize sync Brokle client with SyncHTTPClient."""
        super().__init__(*args, **kwargs)
        self._http_client: Optional[SyncHTTPClient] = None

    @property
    def _http(self) -> SyncHTTPClient:
        """Lazy-init sync HTTP client."""
        if self._http_client is None:
            self._http_client = SyncHTTPClient(self.config)
        return self._http_client

    @property
    def prompts(self) -> PromptManager:
        """
        Access prompt management operations.

        Returns a PromptManager for fetching and managing prompts.
        All methods are synchronous.

        Returns:
            PromptManager instance

        Example:
            >>> prompt = client.prompts.get("greeting", label="production")
            >>> compiled = prompt.compile({"name": "Alice"})
        """
        if self._prompts_manager is None:
            self._prompts_manager = PromptManager(
                http_client=self._http,
                config=self.config,
                prompt_config=self._prompt_config,
            )
        return self._prompts_manager

    @property
    def datasets(self) -> DatasetsManager:
        """
        Access dataset management operations.

        Returns a DatasetsManager for creating and managing datasets.
        All methods are synchronous.

        Returns:
            DatasetsManager instance

        Example:
            >>> dataset = client.datasets.create(name="qa-pairs")
            >>> dataset.insert([{"input": {"q": "2+2?"}, "expected": {"a": "4"}}])
            >>> for item in dataset:
            ...     print(item.input, item.expected)
        """
        if self._datasets_manager is None:
            self._datasets_manager = DatasetsManager(
                http_client=self._http,
                config=self.config,
            )
        return self._datasets_manager

    @property
    def scores(self) -> ScoresManager:
        """
        Access score submission operations.

        Returns a ScoresManager for submitting quality scores.
        All methods are synchronous.

        Returns:
            ScoresManager instance

        Example:
            >>> client.scores.submit(
            ...     trace_id="abc123",
            ...     name="quality",
            ...     value=0.9,
            ... )
        """
        if self._scores_manager is None:
            self._scores_manager = ScoresManager(
                http_client=self._http,
                config=self.config,
            )
        return self._scores_manager

    def auth_check(self) -> bool:
        """
        Verify connection to Brokle server.

        Makes a synchronous request to validate API key.
        Use for development/testing only - adds latency.

        Returns:
            True if authenticated, False otherwise

        Example:
            >>> if client.auth_check():
            ...     print("Connected!")
        """
        try:
            response = self._http.post("/v1/auth/validate-key", json={})
            return response.get("success", False)
        except Exception:
            return False

    def shutdown(self, timeout_seconds: int = 30) -> bool:
        """Shutdown with manager cleanup."""
        success = super().shutdown(timeout_seconds)

        if self._http_client:
            self._http_client.close()
        if self._prompts_manager:
            self._prompts_manager._shutdown()

        return success

    def close(self):
        """Close the client (alias for shutdown)."""
        self.shutdown()

    def __enter__(self) -> "Brokle":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class AsyncBrokle(BaseBrokleClient):
    """
    Asynchronous Brokle client for OpenTelemetry-native LLM observability.

    Core responsibility: Telemetry (traces, spans, metrics, logs)
    Feature APIs: Prompts, datasets, and scores

    This client provides async methods for all operations.
    Uses AsyncHTTPClient (httpx.AsyncClient) internally.

    Example:
        >>> from brokle import AsyncBrokle
        >>>
        >>> # Context manager (recommended)
        >>> async with AsyncBrokle(api_key="bk_...") as client:
        ...     # Core: Telemetry - traces and spans
        ...     with client.start_as_current_span("process") as span:
        ...         result = await do_work()
        ...         span.set_attribute("result", result)
        ...
        ...     # Feature: Prompt management (optional)
        ...     prompt = await client.prompts.get("greeting")
        ...     messages = prompt.to_openai_messages({"name": "Alice"})
    """

    def __init__(self, *args, **kwargs):
        """Initialize async Brokle client with AsyncHTTPClient."""
        super().__init__(*args, **kwargs)
        self._http_client: Optional[AsyncHTTPClient] = None

    @property
    def _http(self) -> AsyncHTTPClient:
        """Lazy-init async HTTP client."""
        if self._http_client is None:
            self._http_client = AsyncHTTPClient(self.config)
        return self._http_client

    @property
    def prompts(self) -> AsyncPromptManager:
        """
        Access prompt management operations.

        Returns an AsyncPromptManager for fetching and managing prompts.
        All methods are async and must be awaited.

        Returns:
            AsyncPromptManager instance

        Example:
            >>> prompt = await client.prompts.get("greeting", label="production")
            >>> compiled = prompt.compile({"name": "Alice"})
        """
        if self._prompts_manager is None:
            self._prompts_manager = AsyncPromptManager(
                http_client=self._http,
                config=self.config,
                prompt_config=self._prompt_config,
            )
        return self._prompts_manager

    @property
    def datasets(self) -> AsyncDatasetsManager:
        """
        Access dataset management operations.

        Returns an AsyncDatasetsManager for creating and managing datasets.
        All methods are async and must be awaited.

        Returns:
            AsyncDatasetsManager instance

        Example:
            >>> dataset = await client.datasets.create(name="qa-pairs")
            >>> await dataset.insert([{"input": {"q": "2+2?"}, "expected": {"a": "4"}}])
            >>> async for item in dataset:
            ...     print(item.input, item.expected)
        """
        if self._datasets_manager is None:
            self._datasets_manager = AsyncDatasetsManager(
                http_client=self._http,
                config=self.config,
            )
        return self._datasets_manager

    @property
    def scores(self) -> AsyncScoresManager:
        """
        Access score submission operations.

        Returns an AsyncScoresManager for submitting quality scores.
        All methods are async and must be awaited.

        Returns:
            AsyncScoresManager instance

        Example:
            >>> await client.scores.submit(
            ...     trace_id="abc123",
            ...     name="quality",
            ...     value=0.9,
            ... )
        """
        if self._scores_manager is None:
            self._scores_manager = AsyncScoresManager(
                http_client=self._http,
                config=self.config,
            )
        return self._scores_manager

    async def auth_check(self) -> bool:
        """
        Verify connection to Brokle server.

        Makes an async request to validate API key.
        Use for development/testing only - adds latency.

        Returns:
            True if authenticated, False otherwise

        Example:
            >>> if await client.auth_check():
            ...     print("Connected!")
        """
        try:
            response = await self._http.post("/v1/auth/validate-key", json={})
            return response.get("success", False)
        except Exception:
            return False

    async def shutdown(self, timeout_seconds: int = 30) -> bool:
        """Shutdown with manager cleanup."""
        success = super().shutdown(timeout_seconds)

        if self._http_client:
            await self._http_client.close()
        if self._prompts_manager:
            await self._prompts_manager._shutdown()

        return success

    async def close(self):
        """Close the client (alias for shutdown)."""
        await self.shutdown()

    async def __aenter__(self) -> "AsyncBrokle":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


_global_client: Optional[Brokle] = None


def get_client(**overrides) -> Brokle:
    """
    Get or create global singleton Brokle client.

    Configuration is read from environment variables on first call.
    Subsequent calls return the same instance.

    Args:
        **overrides: Override specific configuration values

    Returns:
        Singleton Brokle instance

    Raises:
        ValueError: If BROKLE_API_KEY environment variable is missing

    Example:
        >>> from brokle import get_client
        >>> client = get_client()
        >>> prompt = client.prompts.get("greeting")
    """
    global _global_client

    if _global_client is None:
        config = BrokleConfig.from_env(**overrides)
        _global_client = Brokle(config=config)

    return _global_client


def reset_client():
    """
    Reset global singleton client.

    Useful for testing. Should not be used in production code.
    """
    global _global_client
    if _global_client:
        _global_client.close()
    _global_client = None


_global_async_client: Optional[AsyncBrokle] = None


async def get_async_client(**overrides) -> AsyncBrokle:
    """
    Get or create global singleton AsyncBrokle client.

    Configuration is read from environment variables on first call.
    Subsequent calls return the same instance.

    Args:
        **overrides: Override specific configuration values

    Returns:
        Singleton AsyncBrokle instance

    Raises:
        ValueError: If BROKLE_API_KEY environment variable is missing

    Example:
        >>> from brokle import get_async_client
        >>> client = await get_async_client()
        >>> prompt = await client.prompts.get("greeting")
    """
    global _global_async_client

    if _global_async_client is None:
        config = BrokleConfig.from_env(**overrides)
        _global_async_client = AsyncBrokle(config=config)

    return _global_async_client


async def reset_async_client():
    """
    Reset global singleton async client.

    Useful for testing. Should not be used in production code.
    """
    global _global_async_client
    if _global_async_client:
        await _global_async_client.close()
    _global_async_client = None
