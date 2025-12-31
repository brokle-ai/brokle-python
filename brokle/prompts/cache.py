"""
LRU Cache with TTL and Stale-While-Revalidate support.

Provides efficient caching for prompt data with automatic expiration
and background refresh capabilities.
"""

import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Dict, Generic, Optional, Set, TypeVar

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with data and metadata."""

    data: T
    fetched_at: float
    ttl: int


@dataclass
class CacheOptions:
    """Cache configuration options."""

    max_size: int = 100
    default_ttl: int = 60  # seconds
    stale_while_revalidate: bool = True
    stale_grace_period: int = 30  # seconds


class PromptCache(Generic[T]):
    """
    LRU Cache with TTL and SWR support.

    Thread-safe cache implementation with:
    - Least Recently Used eviction policy
    - Time-to-Live expiration
    - Stale-while-revalidate pattern
    - Background refresh tracking
    """

    def __init__(self, options: Optional[CacheOptions] = None):
        """
        Initialize the cache.

        Args:
            options: Cache configuration options
        """
        opts = options or CacheOptions()
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._max_size = opts.max_size
        self._default_ttl = opts.default_ttl
        self._stale_while_revalidate = opts.stale_while_revalidate
        self._stale_grace_period = opts.stale_grace_period
        self._refreshing: Set[str] = set()
        self._lock = Lock()

    def get(self, key: str) -> Optional[T]:
        """
        Get an entry from the cache.

        Args:
            key: Cache key

        Returns:
            Cached data or None if not found/expired
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None

            now = time.time()
            age = now - entry.fetched_at

            if age < entry.ttl:
                self._cache.move_to_end(key)
                return entry.data

            if (
                self._stale_while_revalidate
                and age < entry.ttl + self._stale_grace_period
            ):
                self._cache.move_to_end(key)
                return entry.data

            del self._cache[key]
            return None

    def is_fresh(self, key: str) -> bool:
        """
        Check if an entry exists and is fresh.

        Args:
            key: Cache key

        Returns:
            True if entry exists and is not stale
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            age = time.time() - entry.fetched_at
            return age < entry.ttl

    def is_stale(self, key: str) -> bool:
        """
        Check if an entry is stale but within SWR grace period.

        Args:
            key: Cache key

        Returns:
            True if stale but usable
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            age = time.time() - entry.fetched_at
            return entry.ttl <= age < entry.ttl + self._stale_grace_period

    def is_refreshing(self, key: str) -> bool:
        """
        Check if a key is currently being refreshed.

        Args:
            key: Cache key

        Returns:
            True if refresh in progress
        """
        with self._lock:
            return key in self._refreshing

    def start_refresh(self, key: str) -> None:
        """
        Mark a key as being refreshed.

        Args:
            key: Cache key
        """
        with self._lock:
            self._refreshing.add(key)

    def end_refresh(self, key: str) -> None:
        """
        Mark a key as done refreshing.

        Args:
            key: Cache key
        """
        with self._lock:
            self._refreshing.discard(key)

    def set(self, key: str, data: T, ttl: Optional[int] = None) -> None:
        """
        Set an entry in the cache.

        Args:
            key: Cache key
            data: Data to cache
            ttl: TTL in seconds (optional, uses default)
        """
        with self._lock:
            if self._max_size <= 0:
                return

            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)

            self._cache[key] = CacheEntry(
                data=data,
                fetched_at=time.time(),
                ttl=ttl if ttl is not None else self._default_ttl,
            )

    def delete(self, key: str) -> bool:
        """
        Delete an entry from the cache.

        Args:
            key: Cache key

        Returns:
            True if entry was deleted
        """
        with self._lock:
            self._refreshing.discard(key)
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def delete_by_prompt(self, name: str) -> int:
        """
        Delete all cache entries for a specific prompt by name.

        Removes all entries matching the pattern: prompt:{name}:*
        This includes all labels (latest, production, beta, canary, etc.)
        and all versions (v1, v2, v3, etc.)

        Args:
            name: Prompt name

        Returns:
            Number of entries deleted
        """
        with self._lock:
            prefix = f"prompt:{name}:"
            keys_to_delete = [k for k in self._cache.keys() if k.startswith(prefix)]

            for key in keys_to_delete:
                del self._cache[key]
                self._refreshing.discard(key)

            return len(keys_to_delete)

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()
            self._refreshing.clear()

    @property
    def size(self) -> int:
        """Get the current cache size."""
        with self._lock:
            return len(self._cache)

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "refreshing_count": len(self._refreshing),
            }

    @staticmethod
    def generate_key(
        name: str, label: Optional[str] = None, version: Optional[int] = None
    ) -> str:
        """
        Generate a cache key for a prompt.

        Args:
            name: Prompt name
            label: Optional label
            version: Optional version number

        Returns:
            Cache key string
        """
        if version is not None:
            return f"prompt:{name}:v{version}"
        if label:
            return f"prompt:{name}:{label}"
        return f"prompt:{name}:latest"
