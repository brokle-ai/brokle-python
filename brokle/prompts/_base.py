"""
Base Prompts Manager

Provides the async-first internal implementation for prompt operations.
Both AsyncPromptManager and PromptManager inherit from this class.
"""

import asyncio
import threading
from typing import Any, Dict, List, Optional

from .._http import AsyncHTTPClient
from ..config import BrokleConfig
from .cache import CacheOptions, PromptCache
from .exceptions import PromptFetchError, PromptNotFoundError
from .prompt import Prompt
from .types import (
    FallbackConfig,
    GetPromptOptions,
    PaginatedResponse,
    Pagination,
    PromptConfig,
    PromptData,
    UpsertPromptRequest,
)


class BasePromptsManager:
    """
    Base class for prompts manager with async internal implementation.

    This class provides all the internal async methods that both
    AsyncPromptManager and PromptManager use.
    """

    def __init__(
        self,
        http_client: AsyncHTTPClient,
        config: BrokleConfig,
        prompt_config: Optional[PromptConfig] = None,
    ):
        """
        Initialize the prompts manager.

        Args:
            http_client: Shared async HTTP client
            config: Brokle configuration
            prompt_config: Optional prompt-specific configuration
        """
        self._http = http_client
        self._config = config
        self._prompt_config = prompt_config or PromptConfig()

        # Initialize cache
        if self._prompt_config.cache_enabled:
            cache_opts = CacheOptions(
                max_size=self._prompt_config.cache_max_size,
                default_ttl=self._prompt_config.cache_ttl_seconds,
            )
            self._cache: PromptCache[PromptData] = PromptCache(cache_opts)
        else:
            self._cache = PromptCache(CacheOptions(max_size=0))

    def _log(self, message: str, *args: Any) -> None:
        """Log debug messages."""
        if self._config.debug:
            print(f"[Brokle PromptManager] {message}", *args)

    async def _fetch_prompt(
        self, name: str, options: Optional[GetPromptOptions] = None
    ) -> PromptData:
        """
        Fetch a single prompt from the API.

        Args:
            name: Prompt name
            options: Optional fetch options

        Returns:
            PromptData

        Raises:
            PromptNotFoundError: If prompt is not found
            PromptFetchError: If request fails
        """
        params: Dict[str, Any] = {}
        if options:
            if options.label:
                params["label"] = options.label
            if options.version is not None:
                params["version"] = options.version

        try:
            raw_response = await self._http.get(f"/v1/prompts/{name}", params)
            data = AsyncHTTPClient.unwrap_response(
                raw_response,
                resource_type="Prompt",
                identifier=name,
            )
            return PromptData.from_dict(data)
        except ValueError as e:
            # Check if it's a not found error
            if "not found" in str(e).lower():
                raise PromptNotFoundError(
                    name,
                    version=options.version if options else None,
                    label=options.label if options else None,
                )
            raise PromptFetchError(str(e))
        except Exception as e:
            raise PromptFetchError(f"Failed to fetch prompt: {e}")

    async def _get(
        self,
        name: str,
        label: Optional[str] = None,
        version: Optional[int] = None,
        cache_ttl: Optional[int] = None,
        force_refresh: bool = False,
    ) -> Prompt:
        """
        Get a prompt with caching and SWR support.

        Args:
            name: Prompt name
            label: Optional label filter
            version: Optional version filter
            cache_ttl: Optional TTL override
            force_refresh: Skip cache and fetch fresh

        Returns:
            Prompt instance
        """
        options = GetPromptOptions(label=label, version=version)
        cache_key = PromptCache.generate_key(name, label, version)
        ttl = cache_ttl if cache_ttl is not None else self._prompt_config.cache_ttl_seconds

        # Force refresh - skip cache
        if force_refresh:
            self._log(f"Force refresh: {cache_key}")
            data = await self._fetch_prompt(name, options)
            self._cache.set(cache_key, data, ttl)
            return Prompt.from_data(data)

        # Fresh cache - return immediately
        cached = self._cache.get(cache_key)
        if cached and self._cache.is_fresh(cache_key):
            self._log(f"Cache hit (fresh): {cache_key}")
            return Prompt.from_data(cached)

        # Stale cache - return stale and refresh in background
        if cached and self._cache.is_stale(cache_key):
            self._log(f"Cache hit (stale): {cache_key}")

            # Trigger background refresh if not already in progress
            if not self._cache.is_refreshing(cache_key):
                self._cache.start_refresh(cache_key)

                # Run refresh in background thread (works for both sync/async)
                def _thread_refresh():
                    """Run refresh in dedicated thread with thread-local HTTP client."""
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    async def refresh_with_thread_local_client():
                        # Create thread-local HTTP client bound to this thread's event loop
                        thread_http = AsyncHTTPClient(self._config)

                        try:
                            # Fetch using thread-local client (thread-safe)
                            params: Dict[str, Any] = {}
                            if options:
                                if options.label:
                                    params["label"] = options.label
                                if options.version is not None:
                                    params["version"] = options.version

                            raw_response = await thread_http.get(
                                f"/v1/prompts/{name}", params
                            )
                            data = AsyncHTTPClient.unwrap_response(
                                raw_response,
                                resource_type="Prompt",
                                identifier=name,
                            )
                            prompt_data = PromptData.from_dict(data)

                            # Update cache with fresh data
                            self._cache.set(cache_key, prompt_data, ttl)
                            self._log(f"Background refresh complete: {cache_key}")
                        except Exception as e:
                            self._log(f"Background refresh failed: {e}")
                        finally:
                            self._cache.end_refresh(cache_key)
                            # Clean up thread-local HTTP client
                            await thread_http.close()

                    try:
                        loop.run_until_complete(refresh_with_thread_local_client())
                    finally:
                        loop.close()

                thread = threading.Thread(target=_thread_refresh, daemon=True)
                thread.start()

            return Prompt.from_data(cached)

        self._log(f"Cache miss: {cache_key}")
        data = await self._fetch_prompt(name, options)
        self._cache.set(cache_key, data, ttl)
        return Prompt.from_data(data)

    async def _list(
        self,
        type: Optional[str] = None,
        limit: int = 20,
        page: int = 1,
    ) -> PaginatedResponse:
        """
        List prompts from the API.

        Args:
            type: Optional prompt type filter
            limit: Maximum number of prompts to return
            page: Page number (1-indexed)

        Returns:
            Paginated response with prompts and pagination info
        """
        params: Dict[str, Any] = {
            "limit": limit,
            "page": page,
        }
        if type:
            params["type"] = type

        try:
            raw_response = await self._http.get("/v1/prompts", params)

            # Handle error response
            if not raw_response.get("success"):
                error = raw_response.get("error", {})
                error_msg = error.get("message", "Unknown error")
                raise PromptFetchError(f"Failed to list prompts: {error_msg}")

            # Parse response - pagination is in meta.pagination
            data = [PromptData.from_dict(p) for p in raw_response.get("data", [])]
            pagination_data = raw_response.get("meta", {}).get("pagination", {})
            pagination = Pagination(
                total=pagination_data.get("total", 0),
                page=pagination_data.get("page", page),
                limit=pagination_data.get("limit", limit),
                pages=pagination_data.get("total_pages", 1),
            )

            return PaginatedResponse(
                data=[Prompt.from_data(p) for p in data],
                pagination=pagination,
            )
        except PromptFetchError:
            raise
        except Exception as e:
            raise PromptFetchError(f"Failed to list prompts: {e}")

    async def _upsert(self, request: UpsertPromptRequest) -> Prompt:
        """
        Create or update a prompt.

        Args:
            request: Upsert request

        Returns:
            Created/updated prompt
        """
        try:
            raw_response = await self._http.post(
                "/v1/prompts",
                json=request.to_dict(),
            )
            data = AsyncHTTPClient.unwrap_response(
                raw_response,
                resource_type="Prompt",
                identifier=request.name,
            )

            # Invalidate cache for this prompt
            self.invalidate(request.name)

            return Prompt.from_data(PromptData.from_dict(data))
        except Exception as e:
            raise PromptFetchError(f"Failed to upsert prompt: {e}")

    def invalidate(self, name: str) -> None:
        """
        Invalidate all cached entries for a prompt.

        Removes all cached entries for the prompt name, regardless of
        label or version.

        Args:
            name: Prompt name
        """
        count = self._cache.delete_by_prompt(name)
        self._log(f"Invalidated {count} cache entries for: {name}")

    def clear_cache(self) -> None:
        """Clear the entire cache."""
        self._cache.clear()
        self._log("Cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return self._cache.get_stats()

    async def _shutdown(self) -> None:
        """
        Internal cleanup method.

        Called by parent client during shutdown.
        """
        pass  # Cache cleanup handled by cache itself
