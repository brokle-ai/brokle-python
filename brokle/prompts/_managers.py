"""
Prompts Manager

Provides both synchronous and asynchronous prompt operations for Brokle.

Sync Usage:
    >>> with Brokle(api_key="bk_...") as client:
    ...     prompt = client.prompts.get("greeting", label="production")
    ...     messages = prompt.to_openai_messages({"name": "Alice"})

Async Usage:
    >>> async with AsyncBrokle(api_key="bk_...") as client:
    ...     prompt = await client.prompts.get("greeting", label="production")
    ...     messages = prompt.to_openai_messages({"name": "Alice"})
"""

from typing import List, Optional, Union

from ._base import BaseAsyncPromptsManager, BaseSyncPromptsManager
from .prompt import Prompt
from .types import ChatMessage, Fallback, PaginatedResponse, UpsertPromptRequest


class PromptManager(BaseSyncPromptsManager):
    """
    Sync prompts manager for Brokle.

    All methods are synchronous. Uses SyncHTTPClient (httpx.Client) internally -
    no event loop involvement.

    Example:
        >>> with Brokle(api_key="bk_...") as client:
        ...     prompt = client.prompts.get("greeting", label="production")
        ...     messages = prompt.to_openai_messages({"name": "Alice"})
    """

    def get(
        self,
        name: str,
        *,
        label: Optional[str] = None,
        version: Optional[int] = None,
        cache_ttl: Optional[int] = None,
        force_refresh: bool = False,
        fallback: Optional[Union[str, List[ChatMessage]]] = None,
    ) -> Prompt:
        """
        Get a prompt by name with guaranteed availability.

        Fallback ensures prompts are always available even when the API is unreachable.
        Priority: fresh cache → fetch → stale cache → fallback → raise

        Args:
            name: Prompt name
            label: Optional label filter
            version: Optional version filter
            cache_ttl: Optional cache TTL override
            force_refresh: Skip cache and fetch fresh
            fallback: Fallback content - string for text prompts, list of messages for chat

        Returns:
            Prompt instance (check prompt.is_fallback to detect if fallback was used)

        Raises:
            PromptNotFoundError: If prompt is not found and no fallback provided
            PromptFetchError: If request fails and no fallback provided

        Example:
            >>> # Text prompt with fallback
            >>> prompt = client.prompts.get(
            ...     "greeting",
            ...     label="production",
            ...     fallback="Hello {{name}}!"
            ... )
            >>> if prompt.is_fallback:
            ...     logger.warning("Using fallback prompt")
            >>> messages = prompt.to_openai_messages({"name": "Alice"})

            >>> # Chat prompt with fallback
            >>> prompt = client.prompts.get(
            ...     "assistant",
            ...     fallback=[
            ...         {"role": "system", "content": "You are a helpful assistant."},
            ...         {"role": "user", "content": "{{query}}"}
            ...     ]
            ... )
        """
        return self._get(
            name,
            label=label,
            version=version,
            cache_ttl=cache_ttl,
            force_refresh=force_refresh,
            fallback=fallback,
        )

    def list(
        self,
        *,
        type: Optional[str] = None,
        limit: int = 20,
        page: int = 1,
    ) -> PaginatedResponse:
        """
        List prompts.

        Args:
            type: Optional prompt type filter
            limit: Maximum number of prompts to return
            page: Page number (1-indexed)

        Returns:
            Paginated response with prompt summaries

        Raises:
            PromptFetchError: If request fails

        Example:
            >>> result = client.prompts.list(type="chat", limit=10)
            >>> for summary in result.data:
            ...     print(f"{summary.name} v{summary.latest_version}")
        """
        return self._list(
            type=type,
            limit=limit,
            page=page,
        )

    def upsert(self, request: UpsertPromptRequest) -> Prompt:
        """
        Create or update a prompt.

        Args:
            request: Upsert request with prompt details

        Returns:
            Created/updated prompt

        Raises:
            PromptFetchError: If request fails

        Example:
            >>> from brokle.prompts.types import UpsertPromptRequest, PromptType
            >>> request = UpsertPromptRequest(
            ...     name="greeting",
            ...     type=PromptType.TEXT,
            ...     template={"content": "Hello {{name}}!"},
            ...     commit_message="Initial version",
            ... )
            >>> prompt = client.prompts.upsert(request)
        """
        return self._upsert(request)


class AsyncPromptManager(BaseAsyncPromptsManager):
    """
    Async prompts manager for AsyncBrokle.

    All methods are async and return coroutines that must be awaited.
    Uses AsyncHTTPClient (httpx.AsyncClient) internally.

    Example:
        >>> async with AsyncBrokle(api_key="bk_...") as client:
        ...     prompt = await client.prompts.get("greeting", label="production")
        ...     messages = prompt.to_openai_messages({"name": "Alice"})
    """

    async def get(
        self,
        name: str,
        *,
        label: Optional[str] = None,
        version: Optional[int] = None,
        cache_ttl: Optional[int] = None,
        force_refresh: bool = False,
        fallback: Optional[Union[str, List[ChatMessage]]] = None,
    ) -> Prompt:
        """
        Get a prompt by name with guaranteed availability.

        Fallback ensures prompts are always available even when the API is unreachable.
        Priority: fresh cache → fetch → stale cache → fallback → raise

        Args:
            name: Prompt name
            label: Optional label filter
            version: Optional version filter
            cache_ttl: Optional cache TTL override
            force_refresh: Skip cache and fetch fresh
            fallback: Fallback content - string for text prompts, list of messages for chat

        Returns:
            Prompt instance (check prompt.is_fallback to detect if fallback was used)

        Raises:
            PromptNotFoundError: If prompt is not found and no fallback provided
            PromptFetchError: If request fails and no fallback provided

        Example:
            >>> # Text prompt with fallback
            >>> prompt = await client.prompts.get(
            ...     "greeting",
            ...     label="production",
            ...     fallback="Hello {{name}}!"
            ... )
            >>> if prompt.is_fallback:
            ...     logger.warning("Using fallback prompt")
            >>> messages = prompt.to_openai_messages({"name": "Alice"})

            >>> # Chat prompt with fallback
            >>> prompt = await client.prompts.get(
            ...     "assistant",
            ...     fallback=[
            ...         {"role": "system", "content": "You are a helpful assistant."},
            ...         {"role": "user", "content": "{{query}}"}
            ...     ]
            ... )
        """
        return await self._get(
            name,
            label=label,
            version=version,
            cache_ttl=cache_ttl,
            force_refresh=force_refresh,
            fallback=fallback,
        )

    async def list(
        self,
        *,
        type: Optional[str] = None,
        limit: int = 20,
        page: int = 1,
    ) -> PaginatedResponse:
        """
        List prompts.

        Args:
            type: Optional prompt type filter
            limit: Maximum number of prompts to return
            page: Page number (1-indexed)

        Returns:
            Paginated response with prompt summaries

        Raises:
            PromptFetchError: If request fails

        Example:
            >>> result = await client.prompts.list(type="chat", limit=10)
            >>> for summary in result.data:
            ...     print(f"{summary.name} v{summary.latest_version}")
        """
        return await self._list(
            type=type,
            limit=limit,
            page=page,
        )

    async def upsert(self, request: UpsertPromptRequest) -> Prompt:
        """
        Create or update a prompt.

        Args:
            request: Upsert request with prompt details

        Returns:
            Created/updated prompt

        Raises:
            PromptFetchError: If request fails

        Example:
            >>> from brokle.prompts.types import UpsertPromptRequest, PromptType
            >>> request = UpsertPromptRequest(
            ...     name="greeting",
            ...     type=PromptType.TEXT,
            ...     template={"content": "Hello {{name}}!"},
            ...     commit_message="Initial version",
            ... )
            >>> prompt = await client.prompts.upsert(request)
        """
        return await self._upsert(request)
