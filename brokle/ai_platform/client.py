"""
Unified AI client configuration and backend integration.

This module provides a simplified AI client that coordinates Brokle platform
configuration and delegates intelligence decisions to the backend. The SDK
handles configuration management and provides fallbacks when offline.
"""

import logging
from typing import Dict, List, Optional, Any, Union, AsyncIterator
from dataclasses import dataclass, field
import asyncio
import time

from .routing import RoutingManager, RoutingConfig, ProviderConfig
from .caching import CacheManager, CacheConfig
from .quality import QualityEvaluator, QualityConfig, QualityScore
from .optimization import CostTracker, CostOptimizationConfig, CostBreakdown
from .providers import ProviderMonitor, ProviderHealth, record_provider_request

logger = logging.getLogger(__name__)


@dataclass
class AIRequest:
    """AI request with platform metadata."""
    # Core request data
    prompt: str
    model: Optional[str] = None
    provider: Optional[str] = None

    # Request parameters
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Platform preferences
    routing_strategy: Optional[str] = None
    cache_enabled: bool = True
    quality_evaluation: bool = True
    cost_tracking: bool = True

    # Metadata
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class AIResponse:
    """AI response with platform metadata."""
    # Core response data
    content: str
    model: str
    provider: str

    # Platform metadata
    request_id: str
    cached: bool = False
    cost: float = 0.0
    tokens: int = 0
    latency_ms: float = 0.0

    # Quality assessment
    quality_score: Optional[QualityScore] = None

    # Routing information
    routing_decision: Optional[Dict[str, Any]] = None
    fallback_used: bool = False

    # Cache information
    cache_hit: bool = False
    cache_key: Optional[str] = None

    # Cost optimization
    cost_optimization: Optional[Dict[str, Any]] = None
    budget_status: Optional[Dict[str, Any]] = None

    # Provider health
    provider_health: Optional[ProviderHealth] = None


class AIClient:
    """
    Unified AI client for Brokle platform configuration.

    Coordinates platform configuration and delegates AI intelligence
    decisions to the Brokle backend. Provides simple fallbacks when offline.
    """

    def __init__(
        self,
        routing_config: Optional[RoutingConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quality_config: Optional[QualityConfig] = None,
        optimization_config: Optional[CostOptimizationConfig] = None
    ):
        # Initialize platform managers
        self.routing_manager = RoutingManager()
        self.cache_manager = CacheManager()
        self.quality_evaluator = QualityEvaluator()
        self.cost_tracker = CostTracker()
        self.provider_monitor = ProviderMonitor()

        # Configure platform features
        if routing_config:
            self.routing_manager.configure(routing_config)
        if cache_config:
            self.cache_manager.configure(cache_config)
        if quality_config:
            self.quality_evaluator.configure(quality_config)
        if optimization_config:
            self.cost_tracker.configure(optimization_config)

        # Request tracking
        self._request_counter = 0

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs
    ) -> AIResponse:
        """Generate AI response with full platform integration."""
        request = AIRequest(
            prompt=prompt,
            model=model,
            provider=provider,
            parameters=kwargs,
            request_id=self._generate_request_id()
        )

        return await self._process_request(request)

    async def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate streaming AI response with platform integration."""
        request = AIRequest(
            prompt=prompt,
            model=model,
            provider=provider,
            parameters={**kwargs, "stream": True},
            request_id=self._generate_request_id()
        )

        # Check cache first
        if request.cache_enabled:
            cached_response = self.cache_manager.get_cached_response(
                request.prompt, request.model, request.parameters
            )
            if cached_response:
                # Yield cached response as stream
                for chunk in cached_response.split():
                    yield chunk + " "
                return

        # Stream from provider (placeholder implementation)
        response_chunks = []
        async for chunk in self._stream_from_provider(request):
            response_chunks.append(chunk)
            yield chunk

        # Cache complete response
        complete_response = "".join(response_chunks)
        if request.cache_enabled:
            self.cache_manager.cache_response(
                request.prompt, complete_response, request.model, request.parameters
            )

    async def _process_request(self, request: AIRequest) -> AIResponse:
        """Process AI request via backend with simple fallbacks."""
        start_time = time.time()

        try:
            # Step 1: Check local cache first (fast path)
            if request.cache_enabled:
                cached_response = self.cache_manager.get_cached_response(
                    request.prompt, request.model, request.parameters
                )
                if cached_response:
                    return self._create_cached_response(request, cached_response, start_time)

            # Step 2: Check local budget limits before making backend call
            estimated_cost = 0.01  # Basic estimate
            cost_check = self.cost_tracker.track_request_cost(
                estimated_cost, request.provider or "unknown", request.model or "unknown", 1000
            )
            if not cost_check["allowed"]:
                raise Exception(f"Request blocked by budget: {cost_check['reason']}")

            # Step 3: Send request to backend AI service
            response = await self._call_backend_ai_service(request)

            if response:
                # Step 4: Track metrics and cache locally
                latency_ms = (time.time() - start_time) * 1000

                # Record provider metrics
                record_provider_request(
                    response.provider,
                    success=True,
                    latency_ms=latency_ms,
                    cost=response.cost,
                    tokens=response.tokens
                )

                # Cache response if enabled
                if request.cache_enabled:
                    self.cache_manager.cache_response(
                        request.prompt, response.content, response.model, request.parameters, response.cost
                    )

                return response
            else:
                # Step 5: Fallback to simple local response
                return await self._create_fallback_response(request, start_time)

        except Exception as e:
            logger.error(f"Request failed: {e}")
            # Try fallback response
            try:
                return await self._create_fallback_response(request, start_time)
            except:
                raise e

    async def _call_backend_ai_service(self, request: AIRequest) -> Optional[AIResponse]:
        """Call Brokle backend AI service with configuration."""
        try:
            # TODO: Implement actual backend API call
            # Example: POST /api/v1/ai/generate with config and request

            payload = {
                "prompt": request.prompt,
                "model": request.model,
                "provider": request.provider,
                "parameters": request.parameters,
                "routing_config": self.routing_manager.get_config().to_dict(),
                "cache_config": self.cache_manager.get_config().to_dict(),
                "quality_config": self.quality_evaluator.get_config().to_dict(),
                "optimization_config": self.cost_tracker.get_config().to_dict(),
                "request_id": request.request_id,
                "routing_strategy": request.routing_strategy,
                "cache_enabled": request.cache_enabled,
                "quality_evaluation": request.quality_evaluation,
                "cost_tracking": request.cost_tracking
            }

            logger.debug(f"Would call backend API: POST /api/v1/ai/generate")
            logger.debug(f"Payload size: {len(str(payload))} chars")

            # Return None to trigger fallback
            return None

        except Exception as e:
            logger.warning(f"Backend AI service call failed: {e}")
            return None

    async def _create_fallback_response(self, request: AIRequest, start_time: float) -> AIResponse:
        """Create simple fallback response when backend unavailable."""
        # Very basic fallback response
        latency_ms = (time.time() - start_time) * 1000

        fallback_content = f"Fallback response for: {request.prompt[:50]}..."
        if len(request.prompt) > 50:
            fallback_content += " (truncated)"

        # Basic cost and token estimates
        estimated_cost = 0.005
        estimated_tokens = len(request.prompt.split()) * 2  # Simple estimate

        return AIResponse(
            content=fallback_content,
            model=request.model or "fallback-model",
            provider=request.provider or "local-fallback",
            request_id=request.request_id or self._generate_request_id(),
            cost=estimated_cost,
            tokens=estimated_tokens,
            latency_ms=latency_ms,
            cached=False,
            fallback_used=True
        )

    def _create_cached_response(
        self,
        request: AIRequest,
        cached_content: str,
        start_time: float
    ) -> AIResponse:
        """Create response from cached content."""
        latency_ms = (time.time() - start_time) * 1000

        return AIResponse(
            content=cached_content,
            model=request.model or "cached",
            provider=request.provider or "cache",
            request_id=request.request_id,
            cached=True,
            cache_hit=True,
            latency_ms=latency_ms,
            cost=0.0,  # No cost for cached responses
            tokens=self._count_tokens(request.prompt, cached_content)
        )

    async def _stream_from_provider(self, request: AIRequest) -> AsyncIterator[str]:
        """Simple streaming fallback when backend unavailable."""
        # Create simple streaming response by breaking fallback into chunks
        fallback_content = f"Streaming fallback response for: {request.prompt[:50]}..."
        if len(request.prompt) > 50:
            fallback_content += " (truncated)"

        # Split into word chunks for streaming effect
        words = fallback_content.split()
        for word in words:
            yield word + " "
            await asyncio.sleep(0.1)  # Simulate streaming delay

    def _count_tokens(self, prompt: str, response: str) -> int:
        """Simple token estimation for SDK fallback scenarios."""
        # Basic word-based token estimation (GPT-style: ~1.3 tokens per word)
        prompt_words = len(prompt.split()) if prompt else 0
        response_words = len(response.split()) if response else 0
        return int((prompt_words + response_words) * 1.3)

    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        self._request_counter += 1
        return f"req_{int(time.time())}_{self._request_counter}"

    # Configuration methods
    def configure_routing(self, config: RoutingConfig) -> None:
        """Configure routing settings."""
        self.routing_manager.configure(config)

    def configure_caching(self, config: CacheConfig) -> None:
        """Configure caching settings."""
        self.cache_manager.configure(config)

    def configure_quality(self, config: QualityConfig) -> None:
        """Configure quality evaluation."""
        self.quality_evaluator.configure(config)

    def configure_optimization(self, config: CostOptimizationConfig) -> None:
        """Configure cost optimization."""
        self.cost_tracker.configure(config)

    # Analytics and monitoring methods
    def get_platform_stats(self) -> Dict[str, Any]:
        """Get comprehensive platform statistics."""
        return {
            "routing": {
                "available_providers": len(self.routing_manager.get_available_providers()),
                "circuit_breaker_status": self.routing_manager.get_circuit_breaker_status()
            },
            "caching": self.cache_manager.get_cache_stats(),
            "quality": self.quality_evaluator.get_quality_stats(),
            "optimization": self.cost_tracker.get_optimization_stats(),
            "providers": self.provider_monitor.get_monitor_stats(),
            "requests_processed": self._request_counter
        }

    def get_cost_breakdown(self, period: Optional[str] = None) -> CostBreakdown:
        """Get detailed cost breakdown."""
        return self.cost_tracker.get_cost_breakdown(period)

    def get_provider_rankings(self) -> List[Dict[str, Any]]:
        """Get providers ranked by performance."""
        return self.provider_monitor.get_provider_rankings()

    def get_cache_performance(self) -> Dict[str, Any]:
        """Get cache performance metrics."""
        stats = self.cache_manager.get_cache_stats()
        return {
            "hit_rate": stats.get("local_cache", {}).get("hit_rate", 0),
            "total_requests": stats.get("local_cache", {}).get("total_requests", 0),
            "cache_size": stats.get("local_cache", {}).get("size", 0),
            "cost_saved": stats.get("total_cost_saved", 0)
        }

    # Convenience methods for common configurations
    def setup_cost_optimized(self) -> None:
        """Setup for maximum cost savings."""
        from .routing import create_cost_optimized_routing
        from .optimization import create_aggressive_optimization

        self.configure_routing(create_cost_optimized_routing())
        self.configure_optimization(create_aggressive_optimization())

    def setup_quality_optimized(self) -> None:
        """Setup for maximum quality."""
        from .routing import create_quality_optimized_routing
        from .quality import create_comprehensive_quality

        self.configure_routing(create_quality_optimized_routing())
        self.configure_quality(create_comprehensive_quality())

    def setup_balanced(self) -> None:
        """Setup for balanced cost and quality."""
        from .routing import RoutingStrategy
        from .optimization import create_balanced_optimization

        # Create balanced routing config
        routing_config = RoutingConfig(strategy=RoutingStrategy.BALANCED)
        routing_config.add_primary_provider("openai", "gpt-3.5-turbo")
        routing_config.add_primary_provider("anthropic", "claude-3-haiku-20240307")

        self.configure_routing(routing_config)
        self.configure_optimization(create_balanced_optimization())


# Global AI client instance
_ai_client: Optional[AIClient] = None


def get_ai_client() -> AIClient:
    """Get global AI client instance."""
    global _ai_client

    if _ai_client is None:
        _ai_client = AIClient()

    return _ai_client


def configure_ai_platform(
    routing_config: Optional[RoutingConfig] = None,
    cache_config: Optional[CacheConfig] = None,
    quality_config: Optional[QualityConfig] = None,
    optimization_config: Optional[CostOptimizationConfig] = None
) -> None:
    """Configure global AI platform settings."""
    client = get_ai_client()

    if routing_config:
        client.configure_routing(routing_config)
    if cache_config:
        client.configure_caching(cache_config)
    if quality_config:
        client.configure_quality(quality_config)
    if optimization_config:
        client.configure_optimization(optimization_config)


async def generate(
    prompt: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    **kwargs
) -> AIResponse:
    """Generate AI response using global client."""
    client = get_ai_client()
    return await client.generate(prompt, model, provider, **kwargs)


async def generate_stream(
    prompt: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    **kwargs
) -> AsyncIterator[str]:
    """Generate streaming AI response using global client."""
    client = get_ai_client()
    async for chunk in client.generate_stream(prompt, model, provider, **kwargs):
        yield chunk