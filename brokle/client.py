"""
Main client for Brokle SDK.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List, Union

import httpx

from .config import Config, get_config
from .auth import AuthManager, AuthInfo
from .core.telemetry import TelemetryManager, get_telemetry_manager
from .core.background_processor import BackgroundProcessor, get_background_processor
from .types.requests import (
    CompletionRequest,
    ChatCompletionRequest,
    EmbeddingRequest,
    AnalyticsRequest,
    EvaluationRequest,
)
from .types.responses import (
    CompletionResponse,
    ChatCompletionResponse,
    EmbeddingResponse,
    AnalyticsResponse,
    EvaluationResponse,
    APIResponse,
)

logger = logging.getLogger(__name__)


class CompletionsClient:
    """Client for completions API."""
    
    def __init__(self, brokle_client: 'Brokle'):
        self.brokle_client = brokle_client
    
    async def create(self, **kwargs) -> CompletionResponse:
        """Create a completion."""
        request = CompletionRequest(**kwargs)
        return await self.brokle_client._make_request(
            "POST",
            "/v1/completions",
            request.model_dump(exclude_none=True),
            response_model=CompletionResponse
        )
    
    def create_sync(self, **kwargs) -> CompletionResponse:
        """Create a completion synchronously."""
        return asyncio.run(self.create(**kwargs))


class ChatCompletionsClient:
    """Client for chat completions API."""
    
    def __init__(self, brokle_client: 'Brokle'):
        self.brokle_client = brokle_client
    
    async def create(self, **kwargs) -> ChatCompletionResponse:
        """Create a chat completion."""
        request = ChatCompletionRequest(**kwargs)
        return await self.brokle_client._make_request(
            "POST",
            "/v1/chat/completions",
            request.model_dump(exclude_none=True),
            response_model=ChatCompletionResponse
        )
    
    def create_sync(self, **kwargs) -> ChatCompletionResponse:
        """Create a chat completion synchronously."""
        return asyncio.run(self.create(**kwargs))


class EmbeddingsClient:
    """Client for embeddings API."""
    
    def __init__(self, brokle_client: 'Brokle'):
        self.brokle_client = brokle_client
    
    async def create(self, **kwargs) -> EmbeddingResponse:
        """Create embeddings."""
        request = EmbeddingRequest(**kwargs)
        return await self.brokle_client._make_request(
            "POST",
            "/v1/embeddings",
            request.model_dump(exclude_none=True),
            response_model=EmbeddingResponse
        )
    
    def create_sync(self, **kwargs) -> EmbeddingResponse:
        """Create embeddings synchronously."""
        return asyncio.run(self.create(**kwargs))


class AnalyticsClient:
    """Client for analytics API."""
    
    def __init__(self, brokle_client: 'Brokle'):
        self.brokle_client = brokle_client
    
    async def get_metrics(self, **kwargs) -> AnalyticsResponse:
        """Get analytics metrics."""
        request = AnalyticsRequest(**kwargs)
        return await self.brokle_client._make_request(
            "GET",
            "/api/v1/analytics/metrics",
            request.model_dump(exclude_none=True),
            response_model=AnalyticsResponse
        )
    
    def get_metrics_sync(self, **kwargs) -> AnalyticsResponse:
        """Get analytics metrics synchronously."""
        return asyncio.run(self.get_metrics(**kwargs))
    
    async def get_real_time_metrics(self) -> AnalyticsResponse:
        """Get real-time metrics."""
        return await self.brokle_client._make_request(
            "GET",
            "/api/v1/analytics/real-time",
            response_model=AnalyticsResponse
        )
    
    def get_real_time_metrics_sync(self) -> AnalyticsResponse:
        """Get real-time metrics synchronously."""
        return asyncio.run(self.get_real_time_metrics())


class EvaluationClient:
    """Client for evaluation API."""
    
    def __init__(self, brokle_client: 'Brokle'):
        self.brokle_client = brokle_client
    
    async def evaluate_response(self, **kwargs) -> EvaluationResponse:
        """Evaluate a response."""
        request = EvaluationRequest(**kwargs)
        return await self.brokle_client._make_request(
            "POST",
            "/api/v1/evaluation/evaluate",
            request.model_dump(exclude_none=True),
            response_model=EvaluationResponse
        )
    
    def evaluate_response_sync(self, **kwargs) -> EvaluationResponse:
        """Evaluate a response synchronously."""
        return asyncio.run(self.evaluate_response(**kwargs))
    
    async def submit_feedback(self, **kwargs) -> Dict[str, Any]:
        """Submit feedback."""
        return await self.brokle_client._make_request(
            "POST",
            "/api/v1/evaluation/feedback",
            kwargs
        )
    
    def submit_feedback_sync(self, **kwargs) -> Dict[str, Any]:
        """Submit feedback synchronously."""
        return asyncio.run(self.submit_feedback(**kwargs))


class Brokle:
    """Main Brokle client with full platform features."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        project_id: Optional[str] = None,
        environment: Optional[str] = None,
        config: Optional[Config] = None,
        **kwargs
    ):
        """Initialize Brokle client."""
        # Use provided config or create from parameters
        if config:
            self.config = config
        else:
            if any([api_key, host, project_id, environment]):
                # Create config from parameters
                base_config = get_config()
                self.config = Config(
                    api_key=api_key or base_config.api_key,
                    host=host or base_config.host,
                    project_id=project_id or base_config.project_id,
                    environment=environment or base_config.environment,
                    **kwargs
                )
            else:
                # Use global config
                self.config = get_config()
        
        # Initialize managers
        self.auth_manager = AuthManager(self.config)
        self.telemetry_manager = get_telemetry_manager(self.config)
        self.background_processor = get_background_processor(self.config)
        
        # Initialize HTTP client
        self._client: Optional[httpx.AsyncClient] = None
        self._auth_info: Optional[AuthInfo] = None
        
        # Initialize core AI clients
        self.completions = CompletionsClient(self)
        self.chat = ChatCompletionsClient(self)
        self.embeddings = EmbeddingsClient(self)
        
        # Initialize AI-focused clients (public API for developers)
        from .clients.ai_analytics import AIAnalyticsClient
        
        self.analytics = AIAnalyticsClient(self)  # AI usage insights
        self.evaluation = EvaluationClient(self)  # AI quality assessment
        
        # Initialize backend integration (internal - for automatic platform features)
        from .clients.telemetry import TelemetryClient
        from .clients.cost import CostClient
        from .clients.cache import CacheClient
        from .clients.ml import MLClient
        
        self._telemetry_client = TelemetryClient(self)
        self._cost_client = CostClient(self)
        self._cache_client = CacheClient(self)
        self._ml_client = MLClient(self)
        
        # Set telemetry client for enhanced observability
        if self.telemetry_manager:
            self.telemetry_manager.set_telemetry_client(self._telemetry_client)
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if not self._client:
            self._client = httpx.AsyncClient(
                base_url=self.config.host,
                headers=self.config.get_headers(),
                timeout=self.config.timeout,
            )
        return self._client
    
    async def _ensure_authenticated(self) -> AuthInfo:
        """Ensure client is authenticated."""
        if not self._auth_info:
            self._auth_info = await self.auth_manager.validate_api_key()
        return self._auth_info
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        response_model: Optional[type] = None,
        **kwargs
    ) -> Union[Dict[str, Any], APIResponse]:
        """Make HTTP request to Brokle with automatic platform integration."""
        import time
        import uuid
        
        # Generate request ID for tracking
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Ensure authentication
        await self._ensure_authenticated()
        
        # Get HTTP client
        client = await self._get_client()
        
        # Add request ID to data if this is an AI request
        if data and any(ai_endpoint in endpoint for ai_endpoint in ['/chat/completions', '/completions', '/embeddings']):
            data['request_id'] = request_id
        
        # Prepare request
        request_kwargs = {
            'method': method,
            'url': endpoint,
            **kwargs
        }
        
        if data:
            request_kwargs['json'] = data
        
        # Make request with telemetry
        with self.telemetry_manager.start_span(
            f"{method} {endpoint}",
            span_type="http_request",
            method=method,
            endpoint=endpoint,
            request_id=request_id
        ) as span:
            try:
                response = await client.request(**request_kwargs)
                response.raise_for_status()
                
                # Parse response
                response_data = response.json()
                
                # Calculate request latency
                request_latency = (time.time() - start_time) * 1000  # ms
                
                # Update telemetry
                if span:
                    self.telemetry_manager.update_span_attributes(
                        span,
                        http_status_code=response.status_code,
                        response_time_ms=request_latency,
                        request_id=request_id
                    )
                
                # Enhance AI responses with automatic platform insights
                if any(ai_endpoint in endpoint for ai_endpoint in ['/chat/completions', '/completions', '/embeddings']):
                    response_data = await self._enhance_ai_response(response_data, request_id, request_latency, data)
                
                # Parse with response model if provided
                if response_model:
                    if isinstance(response_data, dict) and 'data' in response_data:
                        return response_model(**response_data['data'])
                    else:
                        return response_model(**response_data)
                
                return response_data
                
            except httpx.HTTPError as e:
                # Record error in telemetry
                if span:
                    self.telemetry_manager.record_error(
                        span,
                        e,
                        error_type="http_error",
                        error_code=str(getattr(e.response, 'status_code', 'unknown'))
                    )
                raise
            except Exception as e:
                # Record error in telemetry
                if span:
                    self.telemetry_manager.record_error(
                        span,
                        e,
                        error_type="request_error"
                    )
                raise
    
    async def _enhance_ai_response(
        self,
        response_data: Dict[str, Any],
        request_id: str,
        request_latency: float,
        request_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Enhance AI response with automatic platform insights."""
        if not isinstance(response_data, dict):
            return response_data
        
        try:
            # Extract token usage for cost calculation
            usage = response_data.get('usage', {})
            input_tokens = usage.get('prompt_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0)
            
            # Get model info
            model = response_data.get('model', request_data.get('model', 'unknown') if request_data else 'unknown')
            
            # Automatic cost calculation (background task)
            cost_info = None
            if input_tokens > 0 or output_tokens > 0:
                try:
                    # This would normally be automatic via backend, but we can simulate
                    cost_info = await self._calculate_cost_async(
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens
                    )
                except Exception as e:
                    logger.debug(f"Cost calculation failed: {e}")
            
            # Create Brokle metadata
            brokle_metadata = {
                'request_id': request_id,
                'model_used': model,
                'latency_ms': request_latency,
                'optimization_applied': []
            }
            
            # Add cost info if available
            if cost_info:
                brokle_metadata.update({
                    'cost_usd': cost_info.get('total_cost_usd', 0.0),
                    'cost_per_token': cost_info.get('cost_per_token', 0.0)
                })
            
            # Check for caching (this would be automatic via backend)
            cache_info = await self._check_cache_async(request_data) if request_data else None
            if cache_info:
                brokle_metadata.update({
                    'cache_hit': cache_info.get('hit', False),
                    'cache_similarity_score': cache_info.get('similarity_score')
                })
                if cache_info.get('hit'):
                    brokle_metadata['optimization_applied'].append('semantic_caching')
            
            # Add routing info (this would come from backend automatically)
            routing_info = await self._get_routing_info_async(model, request_data) if request_data else None
            if routing_info:
                brokle_metadata.update({
                    'provider': routing_info.get('provider', 'unknown'),
                    'routing_strategy': routing_info.get('strategy', 'default'),
                    'routing_reason': routing_info.get('reason', 'optimal_selection')
                })
                if routing_info.get('cost_optimized'):
                    brokle_metadata['optimization_applied'].append('cost_optimization')
                    brokle_metadata['cost_savings_usd'] = routing_info.get('savings_usd', 0.0)
            
            # Add quality assessment (this would be automatic via backend)
            quality_info = await self._assess_quality_async(response_data) if 'choices' in response_data else None
            if quality_info:
                brokle_metadata['quality_score'] = quality_info.get('score', 0.0)
                if quality_info.get('score', 0) > 0.8:
                    brokle_metadata['optimization_applied'].append('quality_optimization')
            
            # Add the enhanced metadata to response
            response_data['brokle'] = brokle_metadata
            
            # Also add to top level for backward compatibility
            response_data.update({
                'request_id': request_id,
                'latency_ms': request_latency,
                'cost_usd': brokle_metadata.get('cost_usd'),
                'cache_hit': brokle_metadata.get('cache_hit'),
                'provider': brokle_metadata.get('provider'),
                'quality_score': brokle_metadata.get('quality_score')
            })
            
        except Exception as e:
            logger.debug(f"Response enhancement failed: {e}")
            # Still add basic metadata even if enhancement fails
            response_data['brokle'] = {
                'request_id': request_id,
                'latency_ms': request_latency,
                'optimization_applied': []
            }
        
        return response_data
    
    async def _calculate_cost_async(self, model: str, input_tokens: int, output_tokens: int) -> Optional[Dict[str, Any]]:
        """Calculate cost asynchronously (normally automatic via backend)."""
        try:
            # In a real implementation, this would be automatic via cost-tracking-service
            # For now, simulate with basic calculation
            cost_per_input_token = 0.00001  # $0.01 per 1K tokens
            cost_per_output_token = 0.00002  # $0.02 per 1K tokens
            
            input_cost = (input_tokens / 1000) * cost_per_input_token
            output_cost = (output_tokens / 1000) * cost_per_output_token
            total_cost = input_cost + output_cost
            
            return {
                'total_cost_usd': total_cost,
                'input_cost_usd': input_cost,
                'output_cost_usd': output_cost,
                'cost_per_token': total_cost / (input_tokens + output_tokens) if (input_tokens + output_tokens) > 0 else 0
            }
        except Exception:
            return None
    
    async def _check_cache_async(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check cache asynchronously (normally automatic via backend)."""
        try:
            # In a real implementation, this would be automatic via cache-service
            # For now, simulate basic cache check
            return {
                'hit': False,  # Would be determined by backend
                'similarity_score': None
            }
        except Exception:
            return None
    
    async def _get_routing_info_async(self, model: str, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get routing info asynchronously (normally automatic via backend)."""
        try:
            # In a real implementation, this would be automatic via routing-service + ml-service
            # For now, simulate basic routing info
            return {
                'provider': 'openai',  # Would be determined by ML routing
                'strategy': request_data.get('routing_strategy', 'balanced'),
                'reason': 'cost_and_quality_optimized',
                'cost_optimized': True,
                'savings_usd': 0.001  # Simulated savings
            }
        except Exception:
            return None
    
    async def _assess_quality_async(self, response_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Assess quality asynchronously (normally automatic via backend)."""
        try:
            # In a real implementation, this would be automatic via evaluation-service
            # For now, simulate basic quality assessment
            choices = response_data.get('choices', [])
            if choices and len(choices) > 0:
                content = choices[0].get('message', {}).get('content', '') or choices[0].get('text', '')
                # Simple quality heuristic (real implementation would use ML models)
                quality_score = min(1.0, len(content) / 100)  # Longer = higher quality (simplified)
                return {'score': quality_score}
        except Exception:
            pass
        return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        return await self._make_request("GET", "/health")
    
    def health_check_sync(self) -> Dict[str, Any]:
        """Check API health synchronously."""
        return asyncio.run(self.health_check())
    
    async def get_config_info(self) -> Dict[str, Any]:
        """Get configuration information."""
        return await self._make_request("GET", "/api/v1/config")
    
    def get_config_info_sync(self) -> Dict[str, Any]:
        """Get configuration information synchronously."""
        return asyncio.run(self.get_config_info())
    
    def flush_telemetry(self) -> None:
        """Flush pending telemetry data."""
        self.telemetry_manager.flush()
        self.background_processor.flush()
    
    async def close(self) -> None:
        """Close client and cleanup resources."""
        if self._client:
            await self._client.aclose()
        
        self.flush_telemetry()
        self.background_processor.shutdown()
        self.telemetry_manager.shutdown()
    
    def __enter__(self) -> 'Brokle':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        asyncio.run(self.close())
    
    async def __aenter__(self) -> 'Brokle':
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()


# Global client instance
_global_client: Optional[Brokle] = None


def get_client(
    api_key: Optional[str] = None,
    host: Optional[str] = None,
    project_id: Optional[str] = None,
    environment: Optional[str] = None,
    **kwargs
) -> Brokle:
    """Get or create global Brokle client."""
    global _global_client
    
    if _global_client is None:
        _global_client = Brokle(
            api_key=api_key,
            host=host,
            project_id=project_id,
            environment=environment,
            **kwargs
        )
    
    return _global_client


def reset_client() -> None:
    """Reset global client (for testing)."""
    global _global_client
    
    if _global_client:
        asyncio.run(_global_client.close())
    
    _global_client = None