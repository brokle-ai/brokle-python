"""
Prompt Client

Main client for fetching and managing prompts from the Brokle API.
Supports both sync and async operations with caching and stale-while-revalidate.
"""

import atexit
import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

import httpx

from .cache import CacheOptions, PromptCache
from .exceptions import PromptFetchError, PromptNotFoundError
from .prompt import Prompt
from .types import (
    FallbackConfig,
    GetPromptOptions,
    ListPromptsOptions,
    Pagination,
    PaginatedResponse,
    PromptConfig,
    PromptData,
    UpsertPromptRequest,
)


class PromptClient:
    """
    Prompt API client with caching and SWR support.

    Supports both synchronous and asynchronous operations.

    Example:
        >>> client = PromptClient(
        ...     api_key="bk_...",
        ...     base_url="https://api.brokle.ai"
        ... )
        >>> prompt = await client.get("greeting", label="production")
        >>> messages = prompt.to_openai_messages({"name": "Alice"})
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "http://localhost:8080",
        config: Optional[PromptConfig] = None,
        debug: bool = False,
        timeout: float = 30.0,
    ):
        """
        Initialize the prompt client.

        Args:
            api_key: Brokle API key
            base_url: Base URL for the API
            config: Client configuration with cache and retry settings
            debug: Enable debug logging
            timeout: HTTP timeout in seconds
        """
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._debug = debug
        self._timeout = timeout
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._closed = False

        # Register cleanup on process exit
        atexit.register(self._atexit_cleanup)

        self._config = config or PromptConfig()

        if self._config.cache_enabled:
            cache_opts = CacheOptions(
                max_size=self._config.cache_max_size,
                default_ttl=self._config.cache_ttl_seconds,
            )
            self._cache: PromptCache[PromptData] = PromptCache(cache_opts)
        else:
            self._cache = PromptCache(CacheOptions(max_size=0))

        self._max_retries = self._config.max_retries
        self._retry_delay = self._config.retry_delay
        self._cache_ttl_seconds = self._config.cache_ttl_seconds

    def _log(self, message: str, *args: Any) -> None:
        """Log debug messages."""
        if self._debug:
            print(f"[Brokle Prompt] {message}", *args)

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for requests."""
        return {
            "X-API-Key": self._api_key,
            "Content-Type": "application/json",
        }

    def _unwrap_response(
        self,
        response: Dict[str, Any],
        prompt_name: Optional[str] = None,
        label: Optional[str] = None,
        version: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Unwrap API response envelope.

        Backend returns: {"success": bool, "data": {...}, "error": {...}, "meta": {...}}
        This extracts the data or raises appropriate error.

        Args:
            response: Raw API response
            prompt_name: Prompt name for error context
            label: Label for error context
            version: Version for error context

        Returns:
            Unwrapped data from response

        Raises:
            PromptNotFoundError: When prompt is not found
            PromptFetchError: When request fails or response is invalid
        """
        if not isinstance(response, dict):
            raise PromptFetchError("Invalid response format: expected dict")

        # Check for error response
        if not response.get("success", True):
            error = response.get("error", {})
            error_code = error.get("code", "unknown_error")
            error_message = error.get("message", "Unknown error occurred")
            error_type = error.get("type", error_code)

            # Map error codes to exceptions
            if error_type == "not_found" and prompt_name:
                raise PromptNotFoundError(prompt_name, version, label)

            raise PromptFetchError(
                f"{error_code}: {error_message}",
                status_code=self._error_code_to_status(error_code),
            )

        # Extract data from successful response
        data = response.get("data")
        if data is None:
            raise PromptFetchError("Response missing 'data' field")

        return data

    def _unwrap_paginated_response(
        self,
        response: Dict[str, Any],
    ) -> tuple:
        """
        Unwrap paginated API response.

        Args:
            response: Raw API response with pagination

        Returns:
            Tuple of (data_list, pagination_dict)

        Raises:
            PromptFetchError: When request fails or response is invalid
        """
        if not isinstance(response, dict):
            raise PromptFetchError("Invalid response format: expected dict")

        # Check for error response
        if not response.get("success", True):
            error = response.get("error", {})
            error_code = error.get("code", "unknown_error")
            error_message = error.get("message", "Unknown error occurred")

            raise PromptFetchError(
                f"{error_code}: {error_message}",
                status_code=self._error_code_to_status(error_code),
            )

        data = response.get("data", [])
        meta = response.get("meta", {})
        pagination = meta.get("pagination", {})
        return data, pagination

    def _error_code_to_status(self, code: str) -> int:
        """Map error code to HTTP status code."""
        mapping = {
            "not_found": 404,
            "validation_error": 400,
            "unauthorized": 401,
            "forbidden": 403,
            "conflict": 409,
            "rate_limit": 429,
            "internal_error": 500,
        }
        return mapping.get(code, 500)

    def _http_get_sync(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        prompt_name: Optional[str] = None,
        label: Optional[str] = None,
        version: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Make a synchronous HTTP GET request with retry logic.

        Args:
            path: API path
            params: Query parameters
            prompt_name: Prompt name for error context
            label: Label for error context
            version: Version for error context

        Returns:
            JSON response

        Raises:
            PromptNotFoundError: When prompt is not found (404)
            PromptFetchError: When request fails
        """
        url = f"{self._base_url}{path}"
        if params:
            params = {k: v for k, v in params.items() if v is not None}

        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            try:
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.get(
                        url, headers=self._get_headers(), params=params
                    )
                    response.raise_for_status()
                    return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404 and prompt_name:
                    raise PromptNotFoundError(prompt_name, version, label)
                last_error = e
                # Don't retry on 4xx errors (except 429)
                if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                    raise PromptFetchError(
                        f"HTTP {e.response.status_code}: {str(e)}",
                        status_code=e.response.status_code,
                    )
            except httpx.RequestError as e:
                last_error = e

            # Wait before retry (exponential backoff)
            if attempt < self._max_retries:
                delay = self._retry_delay * (2 ** attempt)
                self._log(f"Request failed, retrying in {delay}s (attempt {attempt + 1})")
                time.sleep(delay)

        raise PromptFetchError(f"Request failed after {self._max_retries + 1} attempts: {last_error}")

    def _http_post_sync(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a synchronous HTTP POST request with retry logic.

        Args:
            path: API path
            body: Request body

        Returns:
            JSON response

        Raises:
            PromptFetchError: When request fails
        """
        url = f"{self._base_url}{path}"
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            try:
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.post(
                        url, headers=self._get_headers(), json=body
                    )
                    response.raise_for_status()
                    return response.json()
            except httpx.HTTPStatusError as e:
                last_error = e
                # Don't retry on 4xx errors (except 429)
                if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                    raise PromptFetchError(
                        f"HTTP {e.response.status_code}: {str(e)}",
                        status_code=e.response.status_code,
                    )
            except httpx.RequestError as e:
                last_error = e

            # Wait before retry (exponential backoff)
            if attempt < self._max_retries:
                delay = self._retry_delay * (2 ** attempt)
                self._log(f"Request failed, retrying in {delay}s (attempt {attempt + 1})")
                time.sleep(delay)

        raise PromptFetchError(f"Request failed after {self._max_retries + 1} attempts: {last_error}")

    def _fetch_prompt_sync(
        self,
        name: str,
        options: Optional[GetPromptOptions] = None
    ) -> PromptData:
        """
        Fetch a prompt from the API synchronously.

        Args:
            name: Prompt name
            options: Fetch options

        Returns:
            PromptData from API

        Raises:
            PromptNotFoundError: When prompt is not found
            PromptFetchError: When request fails
        """
        params: Dict[str, Any] = {}
        label = options.label if options else None
        version = options.version if options else None

        if options:
            if options.label:
                params["label"] = options.label
            if options.version is not None:
                params["version"] = options.version

        self._log(f"Fetching prompt: {name}", params)
        raw_response = self._http_get_sync(
            f"/v1/prompts/{name}",
            params,
            prompt_name=name,
            label=label,
            version=version,
        )
        data = self._unwrap_response(raw_response, prompt_name=name, label=label, version=version)
        return PromptData.from_dict(data)

    def get_sync(
        self,
        name: str,
        label: Optional[str] = None,
        version: Optional[int] = None,
        cache_ttl: Optional[int] = None,
        force_refresh: bool = False,
    ) -> Prompt:
        """
        Get a prompt by name synchronously with caching.

        Args:
            name: Prompt name
            label: Fetch by label (e.g., 'production')
            version: Fetch by specific version number
            cache_ttl: Cache TTL in seconds (uses config default if not provided)
            force_refresh: Force refresh from API

        Returns:
            Prompt instance

        Example:
            >>> prompt = client.get_sync("greeting", label="production")
        """
        ttl = cache_ttl if cache_ttl is not None else self._cache_ttl_seconds

        options = GetPromptOptions(
            label=label,
            version=version,
            cache_ttl=ttl,
            force_refresh=force_refresh,
        )
        cache_key = PromptCache.generate_key(name, label, version)

        if force_refresh:
            self._log(f"Force refresh: {cache_key}")
            data = self._fetch_prompt_sync(name, options)
            self._cache.set(cache_key, data, ttl)
            return Prompt.from_data(data)

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

                def refresh():
                    try:
                        data = self._fetch_prompt_sync(name, options)
                        self._cache.set(cache_key, data, ttl)
                        self._log(f"Background refresh complete: {cache_key}")
                    except Exception as e:
                        self._log(f"Background refresh failed: {e}")
                    finally:
                        self._cache.end_refresh(cache_key)

                self._executor.submit(refresh)

            return Prompt.from_data(cached)

        self._log(f"Cache miss: {cache_key}")
        data = self._fetch_prompt_sync(name, options)
        self._cache.set(cache_key, data, ttl)
        return Prompt.from_data(data)

    def get_with_fallback_sync(
        self,
        name: str,
        fallback: FallbackConfig,
        label: Optional[str] = None,
        version: Optional[int] = None,
        cache_ttl: Optional[int] = None,
    ) -> Prompt:
        """
        Get a prompt with fallback on failure (synchronous).

        Args:
            name: Prompt name
            fallback: Fallback configuration
            label: Fetch by label
            version: Fetch by version
            cache_ttl: Cache TTL in seconds

        Returns:
            Prompt instance (real or fallback)
        """
        try:
            return self.get_sync(name, label, version, cache_ttl)
        except Exception as e:
            self._log(f"Fetch failed, using fallback: {e}")
            return Prompt.create_fallback(
                name,
                fallback.template,
                fallback.type,
                fallback.config,
            )

    def list_sync(
        self,
        type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        search: Optional[str] = None,
        page: int = 1,
        limit: int = 20,
    ) -> PaginatedResponse:
        """
        List prompts with optional filtering (synchronous).

        Args:
            type: Filter by prompt type
            tags: Filter by tags
            search: Search in name and description
            page: Page number
            limit: Items per page

        Returns:
            Paginated list of prompts
        """
        params: Dict[str, Any] = {
            "page": page,
            "limit": limit,
        }
        if type:
            params["type"] = type
        if search:
            params["search"] = search
        if tags:
            params["tags"] = ",".join(tags)

        self._log("Listing prompts", params)
        raw_response = self._http_get_sync("/v1/prompts", params)
        data_list, pagination_data = self._unwrap_paginated_response(raw_response)

        prompts = [Prompt.from_dict(d) for d in data_list]

        return PaginatedResponse(
            data=prompts,
            pagination=Pagination(
                total=pagination_data.get("total", 0),
                page=pagination_data.get("page", page),
                limit=pagination_data.get("limit", limit),
                pages=pagination_data.get("total_pages", 0),
            ),
        )

    def upsert_sync(self, request: UpsertPromptRequest) -> Prompt:
        """
        Create or update a prompt (synchronous).

        Args:
            request: Prompt data

        Returns:
            Created/updated prompt
        """
        self._log(f"Upserting prompt: {request.name}")
        raw_response = self._http_post_sync("/v1/prompts", request.to_dict())
        data = self._unwrap_response(raw_response, prompt_name=request.name)

        self.invalidate_prompt(request.name)

        return Prompt.from_dict(data)


    async def _http_get_async(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        prompt_name: Optional[str] = None,
        label: Optional[str] = None,
        version: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Make an asynchronous HTTP GET request with retry logic.

        Args:
            path: API path
            params: Query parameters
            prompt_name: Prompt name for error context
            label: Label for error context
            version: Version for error context

        Returns:
            JSON response

        Raises:
            PromptNotFoundError: When prompt is not found (404)
            PromptFetchError: When request fails
        """
        url = f"{self._base_url}{path}"
        if params:
            params = {k: v for k, v in params.items() if v is not None}

        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.get(
                        url, headers=self._get_headers(), params=params
                    )
                    response.raise_for_status()
                    return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404 and prompt_name:
                    raise PromptNotFoundError(prompt_name, version, label)
                last_error = e
                # Don't retry on 4xx errors (except 429)
                if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                    raise PromptFetchError(
                        f"HTTP {e.response.status_code}: {str(e)}",
                        status_code=e.response.status_code,
                    )
            except httpx.RequestError as e:
                last_error = e

            # Wait before retry (exponential backoff)
            if attempt < self._max_retries:
                delay = self._retry_delay * (2 ** attempt)
                self._log(f"Request failed, retrying in {delay}s (attempt {attempt + 1})")
                await asyncio.sleep(delay)

        raise PromptFetchError(f"Request failed after {self._max_retries + 1} attempts: {last_error}")

    async def _http_post_async(
        self, path: str, body: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make an asynchronous HTTP POST request with retry logic.

        Args:
            path: API path
            body: Request body

        Returns:
            JSON response

        Raises:
            PromptFetchError: When request fails
        """
        url = f"{self._base_url}{path}"
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(
                        url, headers=self._get_headers(), json=body
                    )
                    response.raise_for_status()
                    return response.json()
            except httpx.HTTPStatusError as e:
                last_error = e
                # Don't retry on 4xx errors (except 429)
                if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                    raise PromptFetchError(
                        f"HTTP {e.response.status_code}: {str(e)}",
                        status_code=e.response.status_code,
                    )
            except httpx.RequestError as e:
                last_error = e

            # Wait before retry (exponential backoff)
            if attempt < self._max_retries:
                delay = self._retry_delay * (2 ** attempt)
                self._log(f"Request failed, retrying in {delay}s (attempt {attempt + 1})")
                await asyncio.sleep(delay)

        raise PromptFetchError(f"Request failed after {self._max_retries + 1} attempts: {last_error}")

    async def _fetch_prompt_async(
        self,
        name: str,
        options: Optional[GetPromptOptions] = None
    ) -> PromptData:
        """
        Fetch a prompt from the API asynchronously.

        Args:
            name: Prompt name
            options: Fetch options

        Returns:
            PromptData from API

        Raises:
            PromptNotFoundError: When prompt is not found
            PromptFetchError: When request fails
        """
        params: Dict[str, Any] = {}
        label = options.label if options else None
        version = options.version if options else None

        if options:
            if options.label:
                params["label"] = options.label
            if options.version is not None:
                params["version"] = options.version

        self._log(f"Fetching prompt: {name}", params)
        raw_response = await self._http_get_async(
            f"/v1/prompts/{name}",
            params,
            prompt_name=name,
            label=label,
            version=version,
        )
        data = self._unwrap_response(raw_response, prompt_name=name, label=label, version=version)
        return PromptData.from_dict(data)

    async def get(
        self,
        name: str,
        label: Optional[str] = None,
        version: Optional[int] = None,
        cache_ttl: Optional[int] = None,
        force_refresh: bool = False,
    ) -> Prompt:
        """
        Get a prompt by name with caching (async).

        Args:
            name: Prompt name
            label: Fetch by label (e.g., 'production')
            version: Fetch by specific version number
            cache_ttl: Cache TTL in seconds (uses config default if not provided)
            force_refresh: Force refresh from API

        Returns:
            Prompt instance

        Example:
            >>> prompt = await client.get("greeting", label="production")
        """
        ttl = cache_ttl if cache_ttl is not None else self._cache_ttl_seconds

        options = GetPromptOptions(
            label=label,
            version=version,
            cache_ttl=ttl,
            force_refresh=force_refresh,
        )
        cache_key = PromptCache.generate_key(name, label, version)

        if force_refresh:
            self._log(f"Force refresh: {cache_key}")
            data = await self._fetch_prompt_async(name, options)
            self._cache.set(cache_key, data, ttl)
            return Prompt.from_data(data)

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

                async def refresh():
                    try:
                        data = await self._fetch_prompt_async(name, options)
                        self._cache.set(cache_key, data, ttl)
                        self._log(f"Background refresh complete: {cache_key}")
                    except Exception as e:
                        self._log(f"Background refresh failed: {e}")
                    finally:
                        self._cache.end_refresh(cache_key)

                asyncio.create_task(refresh())

            return Prompt.from_data(cached)

        self._log(f"Cache miss: {cache_key}")
        data = await self._fetch_prompt_async(name, options)
        self._cache.set(cache_key, data, ttl)
        return Prompt.from_data(data)

    async def get_with_fallback(
        self,
        name: str,
        fallback: FallbackConfig,
        label: Optional[str] = None,
        version: Optional[int] = None,
        cache_ttl: Optional[int] = None,
    ) -> Prompt:
        """
        Get a prompt with fallback on failure (async).

        Args:
            name: Prompt name
            fallback: Fallback configuration
            label: Fetch by label
            version: Fetch by version
            cache_ttl: Cache TTL in seconds

        Returns:
            Prompt instance (real or fallback)
        """
        try:
            return await self.get(name, label, version, cache_ttl)
        except Exception as e:
            self._log(f"Fetch failed, using fallback: {e}")
            return Prompt.create_fallback(
                name,
                fallback.template,
                fallback.type,
                fallback.config,
            )

    async def list(
        self,
        type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        search: Optional[str] = None,
        page: int = 1,
        limit: int = 20,
    ) -> PaginatedResponse:
        """
        List prompts with optional filtering (async).

        Args:
            type: Filter by prompt type
            tags: Filter by tags
            search: Search in name and description
            page: Page number
            limit: Items per page

        Returns:
            Paginated list of prompts
        """
        params: Dict[str, Any] = {
            "page": page,
            "limit": limit,
        }
        if type:
            params["type"] = type
        if search:
            params["search"] = search
        if tags:
            params["tags"] = ",".join(tags)

        self._log("Listing prompts", params)
        raw_response = await self._http_get_async("/v1/prompts", params)
        data_list, pagination_data = self._unwrap_paginated_response(raw_response)

        prompts = [Prompt.from_dict(d) for d in data_list]

        return PaginatedResponse(
            data=prompts,
            pagination=Pagination(
                total=pagination_data.get("total", 0),
                page=pagination_data.get("page", page),
                limit=pagination_data.get("limit", limit),
                pages=pagination_data.get("total_pages", 0),
            ),
        )

    async def upsert(self, request: UpsertPromptRequest) -> Prompt:
        """
        Create or update a prompt (async).

        Args:
            request: Prompt data

        Returns:
            Created/updated prompt
        """
        self._log(f"Upserting prompt: {request.name}")
        raw_response = await self._http_post_async("/v1/prompts", request.to_dict())
        data = self._unwrap_response(raw_response, prompt_name=request.name)

        self.invalidate_prompt(request.name)

        return Prompt.from_dict(data)


    def invalidate_prompt(self, name: str) -> None:
        """
        Invalidate all cached entries for a prompt.

        Removes all cached entries for the prompt name, regardless of
        label or version. This ensures stale data is not served after
        an upsert operation.

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

    # =========================================================================
    # Resource Cleanup Methods
    # =========================================================================

    def _atexit_cleanup(self) -> None:
        """Cleanup handler called on process exit."""
        if not self._closed:
            self._shutdown_executor()

    def _shutdown_executor(self) -> None:
        """Shutdown the thread pool executor."""
        if self._executor:
            self._executor.shutdown(wait=False)

    def close(self) -> None:
        """
        Close the client and release resources.

        This shuts down the background thread pool used for cache refresh.
        After calling close(), the client should not be used.

        Example:
            >>> client = PromptClient(api_key="bk_...")
            >>> try:
            ...     prompt = client.get_sync("greeting")
            ... finally:
            ...     client.close()
        """
        if self._closed:
            return

        self._closed = True
        self._shutdown_executor()
        self._cache.clear()
        self._log("Client closed")

    def __enter__(self) -> "PromptClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - ensures cleanup."""
        self.close()

    @classmethod
    def from_env(
        cls,
        config: Optional[PromptConfig] = None,
        debug: Optional[bool] = None,
    ) -> "PromptClient":
        """
        Create a prompt client from environment variables.

        Args:
            config: Client configuration with cache and retry settings
            debug: Enable debug logging

        Returns:
            PromptClient instance

        Raises:
            ValueError: If BROKLE_API_KEY is not set
        """
        api_key = os.environ.get("BROKLE_API_KEY")
        if not api_key:
            raise ValueError("BROKLE_API_KEY environment variable not set")

        base_url = os.environ.get("BROKLE_BASE_URL", "http://localhost:8080")
        debug_env = os.environ.get("BROKLE_DEBUG", "false").lower() == "true"

        return cls(
            api_key=api_key,
            base_url=base_url,
            config=config,
            debug=debug if debug is not None else debug_env,
        )
