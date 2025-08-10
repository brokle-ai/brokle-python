"""
Cache client for semantic caching and performance optimization via cache-service.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union

from ..types.requests import (
    CacheGetRequest,
    CacheSetRequest,
    CacheInvalidateRequest,
    EmbeddingGenerationRequest,
    SemanticSearchRequest,
)
from ..types.responses import (
    CacheResponse,
    CacheStatsResponse,
    EmbeddingGenerationResponse,
    SemanticSearchResponse,
)

logger = logging.getLogger(__name__)


class CacheClient:
    """Client for semantic caching and performance optimization via cache-service."""
    
    def __init__(self, brokle_client: 'Brokle'):
        self.brokle_client = brokle_client
    
    # Core Cache Operations
    async def get(
        self,
        *,
        key: Optional[str] = None,
        query: Optional[str] = None,
        similarity_threshold: Optional[float] = 0.8,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> CacheResponse:
        """Get cached response by key or semantic similarity."""
        request = CacheGetRequest(
            key=key,
            query=query,
            similarity_threshold=similarity_threshold,
            provider=provider,
            model=model
        )
        
        return await self.brokle_client._make_request(
            "POST",
            "/api/v1/cache/get",
            request.model_dump(exclude_none=True),
            response_model=CacheResponse
        )
    
    def get_sync(self, **kwargs) -> CacheResponse:
        """Get cache synchronously."""
        return asyncio.run(self.get(**kwargs))
    
    async def set(
        self,
        key: str,
        value: Dict[str, Any],
        *,
        ttl: Optional[int] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Set cache entry."""
        request = CacheSetRequest(
            key=key,
            value=value,
            ttl=ttl,
            embedding=embedding,
            metadata=metadata
        )
        
        return await self.brokle_client._make_request(
            "POST",
            "/api/v1/cache/set",
            request.model_dump(exclude_none=True)
        )
    
    def set_sync(self, key: str, value: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Set cache synchronously."""
        return asyncio.run(self.set(key, value, **kwargs))
    
    async def delete(self, key: str) -> Dict[str, Any]:
        """Delete cache entry by key."""
        return await self.brokle_client._make_request(
            "DELETE",
            f"/api/v1/cache/{key}"
        )
    
    def delete_sync(self, key: str) -> Dict[str, Any]:
        """Delete cache synchronously."""
        return asyncio.run(self.delete(key))
    
    async def invalidate(
        self,
        *,
        keys: Optional[List[str]] = None,
        pattern: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Invalidate cache entries by keys, pattern, or filters."""
        request = CacheInvalidateRequest(
            keys=keys,
            pattern=pattern,
            provider=provider,
            model=model
        )
        
        return await self.brokle_client._make_request(
            "POST",
            "/api/v1/cache/invalidate",
            request.model_dump(exclude_none=True)
        )
    
    def invalidate_sync(self, **kwargs) -> Dict[str, Any]:
        """Invalidate cache synchronously."""
        return asyncio.run(self.invalidate(**kwargs))
    
    async def get_stats(
        self,
        *,
        provider: Optional[str] = None,
        detailed: bool = False,
        **kwargs
    ) -> CacheStatsResponse:
        """Get cache statistics."""
        params = {
            "provider": provider,
            "detailed": detailed,
            **kwargs
        }
        
        return await self.brokle_client._make_request(
            "GET",
            "/api/v1/cache/stats",
            {k: v for k, v in params.items() if v is not None},
            response_model=CacheStatsResponse
        )
    
    def get_stats_sync(self, **kwargs) -> CacheStatsResponse:
        """Get cache stats synchronously."""
        return asyncio.run(self.get_stats(**kwargs))
    
    async def get_by_provider(
        self,
        provider: str,
        *,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        **kwargs
    ) -> List[CacheResponse]:
        """Get cache entries by provider."""
        params = {
            "limit": limit,
            "offset": offset,
            **kwargs
        }
        
        response = await self.brokle_client._make_request(
            "GET",
            f"/api/v1/cache/provider/{provider}",
            {k: v for k, v in params.items() if v is not None}
        )
        
        if isinstance(response, dict) and "entries" in response:
            return [CacheResponse(**entry) for entry in response["entries"]]
        return []
    
    def get_by_provider_sync(self, provider: str, **kwargs) -> List[CacheResponse]:
        """Get cache by provider synchronously."""
        return asyncio.run(self.get_by_provider(provider, **kwargs))
    
    async def cleanup(
        self,
        *,
        expired_only: bool = True,
        provider: Optional[str] = None,
        older_than: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Cleanup cache entries."""
        params = {
            "expired_only": expired_only,
            "provider": provider,
            "older_than": older_than,
            **kwargs
        }
        
        return await self.brokle_client._make_request(
            "POST",
            "/api/v1/cache/cleanup",
            {k: v for k, v in params.items() if v is not None}
        )
    
    def cleanup_sync(self, **kwargs) -> Dict[str, Any]:
        """Cleanup cache synchronously."""
        return asyncio.run(self.cleanup(**kwargs))
    
    # Bulk Operations
    async def get_bulk(
        self,
        requests: List[Dict[str, Any]]
    ) -> List[CacheResponse]:
        """Get multiple cache entries in bulk."""
        response = await self.brokle_client._make_request(
            "POST",
            "/api/v1/cache/bulk/get",
            {"requests": requests}
        )
        
        if isinstance(response, dict) and "results" in response:
            return [CacheResponse(**result) for result in response["results"]]
        return []
    
    def get_bulk_sync(self, requests: List[Dict[str, Any]]) -> List[CacheResponse]:
        """Get bulk cache synchronously."""
        return asyncio.run(self.get_bulk(requests))
    
    async def delete_bulk(
        self,
        keys: List[str]
    ) -> Dict[str, Any]:
        """Delete multiple cache entries in bulk."""
        return await self.brokle_client._make_request(
            "POST",
            "/api/v1/cache/bulk/delete",
            {"keys": keys}
        )
    
    def delete_bulk_sync(self, keys: List[str]) -> Dict[str, Any]:
        """Delete bulk cache synchronously."""
        return asyncio.run(self.delete_bulk(keys))
    
    # Embedding Operations
    async def generate_embeddings(
        self,
        text: Union[str, List[str]],
        *,
        model: Optional[str] = "text-embedding-ada-002",
        provider: Optional[str] = None,
        **kwargs
    ) -> EmbeddingGenerationResponse:
        """Generate embeddings for text."""
        request = EmbeddingGenerationRequest(
            text=text,
            model=model,
            provider=provider
        )
        
        return await self.brokle_client._make_request(
            "POST",
            "/api/v1/embeddings/generate",
            request.model_dump(exclude_none=True),
            response_model=EmbeddingGenerationResponse
        )
    
    def generate_embeddings_sync(self, text: Union[str, List[str]], **kwargs) -> EmbeddingGenerationResponse:
        """Generate embeddings synchronously."""
        return asyncio.run(self.generate_embeddings(text, **kwargs))
    
    async def semantic_search(
        self,
        query: str,
        *,
        embedding: Optional[List[float]] = None,
        top_k: Optional[int] = 10,
        similarity_threshold: Optional[float] = 0.7,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> SemanticSearchResponse:
        """Perform semantic search."""
        request = SemanticSearchRequest(
            query=query,
            embedding=embedding,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            filters=filters
        )
        
        return await self.brokle_client._make_request(
            "POST",
            "/api/v1/embeddings/search",
            request.model_dump(exclude_none=True),
            response_model=SemanticSearchResponse
        )
    
    def semantic_search_sync(self, query: str, **kwargs) -> SemanticSearchResponse:
        """Semantic search synchronously."""
        return asyncio.run(self.semantic_search(query, **kwargs))
    
    async def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding statistics."""
        return await self.brokle_client._make_request(
            "GET",
            "/api/v1/embeddings/stats"
        )
    
    def get_embedding_stats_sync(self) -> Dict[str, Any]:
        """Get embedding stats synchronously."""
        return asyncio.run(self.get_embedding_stats())
    
    # Cache Strategy Helpers
    async def check_semantic_similarity(
        self,
        query: str,
        cached_queries: List[str],
        *,
        threshold: float = 0.8,
        **kwargs
    ) -> Dict[str, Any]:
        """Check semantic similarity between query and cached queries."""
        # Generate embedding for the query
        query_embedding_response = await self.generate_embeddings(query, **kwargs)
        query_embedding = query_embedding_response.embeddings[0]
        
        # Generate embeddings for cached queries
        cached_embeddings_response = await self.generate_embeddings(cached_queries, **kwargs)
        cached_embeddings = cached_embeddings_response.embeddings
        
        # Calculate similarities (simplified - in practice would use cosine similarity)
        similarities = []
        for i, cached_embedding in enumerate(cached_embeddings):
            # This would be proper cosine similarity calculation
            similarity = 0.0  # Placeholder - implement actual similarity calculation
            similarities.append({
                "query": cached_queries[i],
                "similarity": similarity,
                "above_threshold": similarity >= threshold
            })
        
        return {
            "query": query,
            "threshold": threshold,
            "similarities": similarities,
            "best_match": max(similarities, key=lambda x: x["similarity"]) if similarities else None
        }
    
    def check_semantic_similarity_sync(self, query: str, cached_queries: List[str], **kwargs) -> Dict[str, Any]:
        """Check semantic similarity synchronously."""
        return asyncio.run(self.check_semantic_similarity(query, cached_queries, **kwargs))
    
    async def smart_cache_lookup(
        self,
        query: str,
        *,
        exact_key: Optional[str] = None,
        similarity_threshold: float = 0.8,
        fallback_to_semantic: bool = True,
        **kwargs
    ) -> CacheResponse:
        """Smart cache lookup with exact and semantic fallback."""
        # Try exact key lookup first if provided
        if exact_key:
            try:
                result = await self.get(key=exact_key, **kwargs)
                if result.hit:
                    return result
            except Exception as e:
                logger.debug(f"Exact cache lookup failed: {e}")
        
        # Fallback to semantic search if enabled
        if fallback_to_semantic:
            try:
                return await self.get(
                    query=query,
                    similarity_threshold=similarity_threshold,
                    **kwargs
                )
            except Exception as e:
                logger.debug(f"Semantic cache lookup failed: {e}")
        
        # Return cache miss
        return CacheResponse(hit=False, key=exact_key, query=query)
    
    def smart_cache_lookup_sync(self, query: str, **kwargs) -> CacheResponse:
        """Smart cache lookup synchronously."""
        return asyncio.run(self.smart_cache_lookup(query, **kwargs))
    
    async def cache_with_embedding(
        self,
        key: str,
        value: Dict[str, Any],
        text_for_embedding: str,
        *,
        ttl: Optional[int] = None,
        embedding_model: str = "text-embedding-ada-002",
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Cache value with auto-generated embedding for semantic search."""
        # Generate embedding for the text
        embedding_response = await self.generate_embeddings(
            text_for_embedding,
            model=embedding_model,
            **kwargs
        )
        embedding = embedding_response.embeddings[0]
        
        # Cache with embedding
        return await self.set(
            key=key,
            value=value,
            ttl=ttl,
            embedding=embedding,
            metadata={
                **(metadata or {}),
                "embedding_text": text_for_embedding,
                "embedding_model": embedding_model
            }
        )
    
    def cache_with_embedding_sync(
        self,
        key: str,
        value: Dict[str, Any],
        text_for_embedding: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Cache with embedding synchronously."""
        return asyncio.run(self.cache_with_embedding(key, value, text_for_embedding, **kwargs))