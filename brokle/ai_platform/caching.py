"""
Semantic caching configuration and local cache management.

This module provides client-side caching configuration that integrates
with Brokle's backend semantic caching while providing local caching
capabilities for improved performance and resilience.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Caching strategies supported by Brokle platform."""
    SEMANTIC = "semantic"      # Vector similarity-based caching
    EXACT = "exact"           # Exact string matching
    FUZZY = "fuzzy"           # Fuzzy string matching
    CUSTOM = "custom"         # Custom similarity function
    DISABLED = "disabled"     # No caching


class CacheLevel(Enum):
    """Cache levels for different scopes."""
    USER = "user"             # Per-user caching
    SESSION = "session"       # Per-session caching
    GLOBAL = "global"         # Global caching
    ORGANIZATION = "organization"  # Per-organization caching


@dataclass
class SemanticCacheConfig:
    """Configuration for semantic caching using vector similarity."""
    enabled: bool = True

    # Similarity thresholds
    similarity_threshold: float = 0.85  # 0.0-1.0
    min_similarity_for_cache: float = 0.7

    # Embedding configuration
    embedding_model: str = "text-embedding-ada-002"
    embedding_dimensions: int = 1536
    normalize_embeddings: bool = True

    # Cache behavior
    cache_level: CacheLevel = CacheLevel.USER
    ttl_hours: int = 24
    max_cache_size: int = 10000

    # Performance
    enable_async_embedding: bool = True
    batch_embedding_size: int = 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "enabled": self.enabled,
            "similarity_threshold": self.similarity_threshold,
            "min_similarity_for_cache": self.min_similarity_for_cache,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "normalize_embeddings": self.normalize_embeddings,
            "cache_level": self.cache_level.value,
            "ttl_hours": self.ttl_hours,
            "max_cache_size": self.max_cache_size,
            "enable_async_embedding": self.enable_async_embedding,
            "batch_embedding_size": self.batch_embedding_size
        }


@dataclass
class LocalCacheConfig:
    """Configuration for local client-side caching."""
    enabled: bool = True

    # Local cache settings
    max_size: int = 1000
    ttl_seconds: int = 3600  # 1 hour
    cleanup_interval_seconds: int = 300  # 5 minutes

    # Cache strategies
    use_lru_eviction: bool = True
    compress_data: bool = False
    encrypt_data: bool = False

    # Persistence
    persist_to_disk: bool = False
    cache_file_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "cleanup_interval_seconds": self.cleanup_interval_seconds,
            "use_lru_eviction": self.use_lru_eviction,
            "compress_data": self.compress_data,
            "encrypt_data": self.encrypt_data,
            "persist_to_disk": self.persist_to_disk,
            "cache_file_path": self.cache_file_path
        }


@dataclass
class CacheConfig:
    """Complete caching configuration for Brokle AI platform."""
    strategy: CacheStrategy = CacheStrategy.SEMANTIC

    # Backend semantic caching
    semantic: SemanticCacheConfig = field(default_factory=SemanticCacheConfig)

    # Local client-side caching
    local: LocalCacheConfig = field(default_factory=LocalCacheConfig)

    # Fallback behavior
    fallback_to_local: bool = True
    fallback_to_exact: bool = True

    # Cache warming
    enable_cache_warming: bool = False
    warm_cache_patterns: List[str] = field(default_factory=list)

    # Analytics
    track_cache_performance: bool = True
    track_cost_savings: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "strategy": self.strategy.value,
            "semantic": self.semantic.to_dict(),
            "local": self.local.to_dict(),
            "fallback_to_local": self.fallback_to_local,
            "fallback_to_exact": self.fallback_to_exact,
            "enable_cache_warming": self.enable_cache_warming,
            "warm_cache_patterns": self.warm_cache_patterns,
            "track_cache_performance": self.track_cache_performance,
            "track_cost_savings": self.track_cost_savings
        }


@dataclass
class CacheEntry:
    """Local cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.utcnow)
    ttl_seconds: int = 3600
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)

    # Metadata
    similarity_score: Optional[float] = None
    cost_saved: Optional[float] = None
    original_request: Optional[Dict[str, Any]] = None

    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        expiry_time = self.created_at + timedelta(seconds=self.ttl_seconds)
        return datetime.utcnow() > expiry_time

    def touch(self) -> None:
        """Update access time and count."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1


class LocalCache:
    """
    Local client-side cache implementation.

    Provides fast local caching with LRU eviction and optional persistence.
    """

    def __init__(self, config: LocalCacheConfig):
        self.config = config
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []  # For LRU

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.config.enabled:
            return None

        entry = self._cache.get(key)
        if not entry:
            self._misses += 1
            return None

        if entry.is_expired:
            self._remove(key)
            self._misses += 1
            return None

        # Update LRU order
        entry.touch()
        if self.config.use_lru_eviction:
            self._access_order.remove(key)
            self._access_order.append(key)

        self._hits += 1
        logger.debug(f"Cache hit for key: {key[:50]}...")
        return entry.value

    def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        similarity_score: Optional[float] = None,
        cost_saved: Optional[float] = None
    ) -> None:
        """Put value in cache."""
        if not self.config.enabled:
            return

        ttl = ttl_seconds or self.config.ttl_seconds

        # Check if we need to evict
        if len(self._cache) >= self.config.max_size:
            self._evict_if_needed()

        entry = CacheEntry(
            key=key,
            value=value,
            ttl_seconds=ttl,
            similarity_score=similarity_score,
            cost_saved=cost_saved
        )

        self._cache[key] = entry
        if self.config.use_lru_eviction:
            self._access_order.append(key)

        logger.debug(f"Cache put for key: {key[:50]}...")

    def _evict_if_needed(self) -> None:
        """Evict entries if cache is full."""
        if len(self._cache) < self.config.max_size:
            return

        if self.config.use_lru_eviction:
            # Remove least recently used
            while len(self._cache) >= self.config.max_size and self._access_order:
                oldest_key = self._access_order.pop(0)
                self._remove(oldest_key)
        else:
            # Remove oldest by creation time
            oldest_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].created_at
            )
            self._remove(oldest_key)

    def _remove(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self._cache:
            del self._cache[key]
            self._evictions += 1

        if key in self._access_order:
            self._access_order.remove(key)

    def cleanup_expired(self) -> int:
        """Remove expired entries."""
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired
        ]

        for key in expired_keys:
            self._remove(key)

        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
        return len(expired_keys)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._access_order.clear()
        logger.info("Local cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0

        return {
            "enabled": self.config.enabled,
            "size": len(self._cache),
            "max_size": self.config.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }


class CacheManager:
    """
    Client-side cache manager.

    Coordinates between local caching and backend semantic caching,
    providing fallback capabilities and performance optimization.
    """

    def __init__(self):
        self.config = CacheConfig()
        self.local_cache = LocalCache(self.config.local)
        self._backend_available = True

    def configure(self, config: CacheConfig) -> None:
        """Configure caching settings."""
        self.config = config
        self.local_cache = LocalCache(config.local)
        logger.info(f"Cache configured with strategy: {config.strategy.value}")

    def get_config(self) -> CacheConfig:
        """Get current cache configuration."""
        return self.config

    def generate_cache_key(
        self,
        prompt: str,
        model: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate cache key for a request."""
        # Create deterministic key
        key_data = {
            "prompt": prompt,
            "model": model or "default",
            "parameters": parameters or {}
        }

        # Sort parameters for consistent key generation
        if isinstance(key_data["parameters"], dict):
            key_data["parameters"] = dict(sorted(key_data["parameters"].items()))

        key_string = json.dumps(key_data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get_cached_response(
        self,
        prompt: str,
        model: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Get cached response if available."""
        if self.config.strategy == CacheStrategy.DISABLED:
            return None

        # Generate cache key
        cache_key = self.generate_cache_key(prompt, model, parameters)

        # Try local cache first (fastest)
        if self.config.local.enabled:
            local_result = self.local_cache.get(cache_key)
            if local_result is not None:
                logger.debug("Cache hit: local cache")
                return local_result

        # Try backend semantic cache
        if (self.config.strategy == CacheStrategy.SEMANTIC and
            self.config.semantic.enabled and
            self._backend_available):
            try:
                # This would call backend API for semantic similarity search
                # For now, return None (backend integration placeholder)
                backend_result = None  # await self._query_backend_cache(prompt, model, parameters)
                if backend_result:
                    # Cache locally for faster future access
                    self.local_cache.put(cache_key, backend_result)
                    logger.debug("Cache hit: backend semantic cache")
                    return backend_result
            except Exception as e:
                logger.warning(f"Backend cache unavailable: {e}")
                self._backend_available = False

        return None

    def cache_response(
        self,
        prompt: str,
        response: Any,
        model: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        cost_usd: Optional[float] = None
    ) -> None:
        """Cache a response."""
        if self.config.strategy == CacheStrategy.DISABLED:
            return

        cache_key = self.generate_cache_key(prompt, model, parameters)

        # Cache locally
        if self.config.local.enabled:
            self.local_cache.put(
                cache_key,
                response,
                cost_saved=cost_usd
            )

        # Send to backend for semantic caching
        if (self.config.strategy == CacheStrategy.SEMANTIC and
            self.config.semantic.enabled and
            self._backend_available):
            try:
                # This would call backend API to store in semantic cache
                # For now, just log (backend integration placeholder)
                logger.debug("Would cache in backend semantic cache")
            except Exception as e:
                logger.warning(f"Backend cache storage failed: {e}")

    def create_semantic_cache_config(
        self,
        similarity_threshold: float = 0.85,
        ttl_hours: int = 24
    ) -> CacheConfig:
        """Create semantic cache configuration."""
        config = CacheConfig(strategy=CacheStrategy.SEMANTIC)
        config.semantic.similarity_threshold = similarity_threshold
        config.semantic.ttl_hours = ttl_hours
        return config

    def create_exact_cache_config(
        self,
        ttl_hours: int = 24
    ) -> CacheConfig:
        """Create exact match cache configuration."""
        config = CacheConfig(strategy=CacheStrategy.EXACT)
        config.local.ttl_seconds = ttl_hours * 3600
        return config

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        local_stats = self.local_cache.get_stats()

        return {
            "strategy": self.config.strategy.value,
            "backend_available": self._backend_available,
            "local_cache": local_stats,
            "semantic_config": self.config.semantic.to_dict() if self.config.semantic.enabled else None,
            "total_cost_saved": self._calculate_total_cost_saved()
        }

    def _calculate_total_cost_saved(self) -> float:
        """Calculate total cost saved through caching."""
        total_saved = 0.0
        for entry in self.local_cache._cache.values():
            if entry.cost_saved:
                total_saved += entry.cost_saved
        return total_saved

    def cleanup(self) -> None:
        """Cleanup expired cache entries."""
        self.local_cache.cleanup_expired()

    def clear_cache(self) -> None:
        """Clear all cache data."""
        self.local_cache.clear()


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager

    if _cache_manager is None:
        _cache_manager = CacheManager()

    return _cache_manager


def configure_caching(config: CacheConfig) -> None:
    """Configure global caching settings."""
    manager = get_cache_manager()
    manager.configure(config)


def get_cache_config() -> CacheConfig:
    """Get current cache configuration."""
    manager = get_cache_manager()
    return manager.get_config()


def create_semantic_cache_config(
    similarity_threshold: float = 0.85,
    ttl_hours: int = 24
) -> CacheConfig:
    """Create semantic cache configuration."""
    manager = get_cache_manager()
    return manager.create_semantic_cache_config(similarity_threshold, ttl_hours)


def create_exact_cache_config(ttl_hours: int = 24) -> CacheConfig:
    """Create exact match cache configuration."""
    manager = get_cache_manager()
    return manager.create_exact_cache_config(ttl_hours)


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    manager = get_cache_manager()
    return manager.get_cache_stats()