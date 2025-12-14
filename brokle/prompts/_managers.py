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

from typing import Optional

from .._utils import run_sync
from ._base import BasePromptsManager
from .prompt import Prompt
from .types import FallbackConfig, PaginatedResponse, UpsertPromptRequest


class PromptManager(BasePromptsManager):
    """
    Sync prompts manager for Brokle.

    All methods are synchronous. Internally uses run_sync() to execute
    the async implementations.

    Note:
        This client cannot be used inside an async event loop.
        Use AsyncBrokle instead for async contexts.

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
        fallback_config: Optional[FallbackConfig] = None,
    ) -> Prompt:
        """
        Get a prompt by name.

        Args:
            name: Prompt name
            label: Optional label filter
            version: Optional version filter
            cache_ttl: Optional cache TTL override
            force_refresh: Skip cache and fetch fresh
            fallback_config: Fallback configuration (not implemented)

        Returns:
            Prompt instance

        Raises:
            PromptNotFoundError: If prompt is not found
            PromptFetchError: If request fails
            RuntimeError: If called inside an async event loop

        Example:
            >>> prompt = client.prompts.get("greeting", label="production")
            >>> messages = prompt.to_openai_messages({"name": "Alice"})
        """
        return run_sync(
            self._get(
                name,
                label=label,
                version=version,
                cache_ttl=cache_ttl,
                force_refresh=force_refresh,
            )
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
            Paginated response with prompts

        Raises:
            PromptFetchError: If request fails
            RuntimeError: If called inside an async event loop

        Example:
            >>> result = client.prompts.list(type="chat", limit=10)
            >>> for prompt in result.data:
            ...     print(f"{prompt.name} v{prompt.version}")
        """
        return run_sync(
            self._list(
                type=type,
                limit=limit,
                page=page,
            )
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
            RuntimeError: If called inside an async event loop

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
        return run_sync(self._upsert(request))


class AsyncPromptManager(BasePromptsManager):
    """
    Async prompts manager for AsyncBrokle.

    All methods are async and return coroutines that must be awaited.

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
        fallback_config: Optional[FallbackConfig] = None,
    ) -> Prompt:
        """
        Get a prompt by name.

        Args:
            name: Prompt name
            label: Optional label filter
            version: Optional version filter
            cache_ttl: Optional cache TTL override
            force_refresh: Skip cache and fetch fresh
            fallback_config: Fallback configuration (not implemented)

        Returns:
            Prompt instance

        Raises:
            PromptNotFoundError: If prompt is not found
            PromptFetchError: If request fails

        Example:
            >>> prompt = await client.prompts.get("greeting", label="production")
            >>> messages = prompt.to_openai_messages({"name": "Alice"})
        """
        return await self._get(
            name,
            label=label,
            version=version,
            cache_ttl=cache_ttl,
            force_refresh=force_refresh,
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
            Paginated response with prompts

        Raises:
            PromptFetchError: If request fails

        Example:
            >>> result = await client.prompts.list(type="chat", limit=10)
            >>> for prompt in result.data:
            ...     print(f"{prompt.name} v{prompt.version}")
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
