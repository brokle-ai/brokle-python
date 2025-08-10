"""
ML client for intelligent provider routing via ml-service.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional

from ..types.requests import (
    MLRoutingRequest,
    MLTrainingDataRequest,
)
from ..types.responses import (
    MLRoutingResponse,
    MLModelInfoResponse,
)

logger = logging.getLogger(__name__)


class MLClient:
    """Client for intelligent provider routing via ml-service."""
    
    def __init__(self, brokle_client: 'Brokle'):
        self.brokle_client = brokle_client
    
    # ML Routing Operations
    async def get_routing_recommendation(
        self,
        model: str,
        *,
        input_text: Optional[str] = None,
        routing_strategy: Optional[str] = "balanced",
        constraints: Optional[Dict[str, Any]] = None,
        historical_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> MLRoutingResponse:
        """Get ML-powered routing recommendation."""
        request = MLRoutingRequest(
            model=model,
            input_text=input_text,
            routing_strategy=routing_strategy,
            constraints=constraints,
            historical_data=historical_data
        )
        
        return await self.brokle_client._make_request(
            "POST",
            "/routing/recommend",
            request.model_dump(exclude_none=True),
            response_model=MLRoutingResponse
        )
    
    def get_routing_recommendation_sync(self, model: str, **kwargs) -> MLRoutingResponse:
        """Get routing recommendation synchronously."""
        return asyncio.run(self.get_routing_recommendation(model, **kwargs))
    
    async def submit_training_data(
        self,
        request_id: str,
        provider: str,
        model: str,
        input_features: Dict[str, Any],
        performance_metrics: Dict[str, float],
        outcome: str,
        *,
        feedback_score: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Submit training data for model improvement."""
        request = MLTrainingDataRequest(
            request_id=request_id,
            provider=provider,
            model=model,
            input_features=input_features,
            performance_metrics=performance_metrics,
            outcome=outcome,
            feedback_score=feedback_score
        )
        
        return await self.brokle_client._make_request(
            "POST",
            "/routing/training-data",
            request.model_dump(exclude_none=True)
        )
    
    def submit_training_data_sync(
        self,
        request_id: str,
        provider: str,
        model: str,
        input_features: Dict[str, Any],
        performance_metrics: Dict[str, float],
        outcome: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Submit training data synchronously."""
        return asyncio.run(self.submit_training_data(
            request_id, provider, model, input_features, performance_metrics, outcome, **kwargs
        ))
    
    async def get_model_info(self) -> MLModelInfoResponse:
        """Get current ML model information."""
        return await self.brokle_client._make_request(
            "GET",
            "/routing/model-info",
            response_model=MLModelInfoResponse
        )
    
    def get_model_info_sync(self) -> MLModelInfoResponse:
        """Get model info synchronously."""
        return asyncio.run(self.get_model_info())
    
    async def trigger_retrain(
        self,
        *,
        force: bool = False,
        model_type: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Manually trigger model retraining."""
        params = {
            "force": force,
            "model_type": model_type,
            **kwargs
        }
        
        return await self.brokle_client._make_request(
            "POST",
            "/routing/retrain",
            {k: v for k, v in params.items() if v is not None}
        )
    
    def trigger_retrain_sync(self, **kwargs) -> Dict[str, Any]:
        """Trigger retrain synchronously."""
        return asyncio.run(self.trigger_retrain(**kwargs))
    
    # Advanced Routing Strategies
    async def get_cost_optimized_routing(
        self,
        model: str,
        *,
        max_cost_per_token: Optional[float] = None,
        quality_threshold: Optional[float] = None,
        **kwargs
    ) -> MLRoutingResponse:
        """Get cost-optimized routing recommendation."""
        constraints = {
            "strategy": "cost_optimized",
            "max_cost_per_token": max_cost_per_token,
            "quality_threshold": quality_threshold,
            **kwargs
        }
        
        return await self.get_routing_recommendation(
            model=model,
            routing_strategy="cost_optimized",
            constraints=constraints
        )
    
    def get_cost_optimized_routing_sync(self, model: str, **kwargs) -> MLRoutingResponse:
        """Get cost-optimized routing synchronously."""
        return asyncio.run(self.get_cost_optimized_routing(model, **kwargs))
    
    async def get_quality_optimized_routing(
        self,
        model: str,
        *,
        min_quality_score: Optional[float] = None,
        max_latency_ms: Optional[float] = None,
        **kwargs
    ) -> MLRoutingResponse:
        """Get quality-optimized routing recommendation."""
        constraints = {
            "strategy": "quality_optimized",
            "min_quality_score": min_quality_score,
            "max_latency_ms": max_latency_ms,
            **kwargs
        }
        
        return await self.get_routing_recommendation(
            model=model,
            routing_strategy="quality_optimized",
            constraints=constraints
        )
    
    def get_quality_optimized_routing_sync(self, model: str, **kwargs) -> MLRoutingResponse:
        """Get quality-optimized routing synchronously."""
        return asyncio.run(self.get_quality_optimized_routing(model, **kwargs))
    
    async def get_latency_optimized_routing(
        self,
        model: str,
        *,
        max_latency_ms: float,
        acceptable_quality_drop: Optional[float] = None,
        **kwargs
    ) -> MLRoutingResponse:
        """Get latency-optimized routing recommendation."""
        constraints = {
            "strategy": "latency_optimized",
            "max_latency_ms": max_latency_ms,
            "acceptable_quality_drop": acceptable_quality_drop,
            **kwargs
        }
        
        return await self.get_routing_recommendation(
            model=model,
            routing_strategy="latency_optimized",
            constraints=constraints
        )
    
    def get_latency_optimized_routing_sync(self, model: str, **kwargs) -> MLRoutingResponse:
        """Get latency-optimized routing synchronously."""
        return asyncio.run(self.get_latency_optimized_routing(model, **kwargs))
    
    async def get_balanced_routing(
        self,
        model: str,
        *,
        cost_weight: float = 0.33,
        quality_weight: float = 0.33,
        latency_weight: float = 0.34,
        **kwargs
    ) -> MLRoutingResponse:
        """Get balanced routing recommendation with custom weights."""
        constraints = {
            "strategy": "balanced",
            "cost_weight": cost_weight,
            "quality_weight": quality_weight,
            "latency_weight": latency_weight,
            **kwargs
        }
        
        return await self.get_routing_recommendation(
            model=model,
            routing_strategy="balanced",
            constraints=constraints
        )
    
    def get_balanced_routing_sync(self, model: str, **kwargs) -> MLRoutingResponse:
        """Get balanced routing synchronously."""
        return asyncio.run(self.get_balanced_routing(model, **kwargs))
    
    # Provider Health and Performance
    async def get_provider_health_scores(
        self,
        *,
        provider: Optional[str] = None,
        time_window: str = "1h",
        **kwargs
    ) -> Dict[str, Any]:
        """Get provider health scores from ML model."""
        params = {
            "provider": provider,
            "time_window": time_window,
            **kwargs
        }
        
        # This would typically be part of the routing recommendation
        # but could be a separate endpoint for health monitoring
        return await self.brokle_client._make_request(
            "GET",
            "/routing/provider-health",
            {k: v for k, v in params.items() if v is not None}
        )
    
    def get_provider_health_scores_sync(self, **kwargs) -> Dict[str, Any]:
        """Get provider health scores synchronously."""
        return asyncio.run(self.get_provider_health_scores(**kwargs))
    
    async def predict_provider_performance(
        self,
        provider: str,
        model: str,
        *,
        input_characteristics: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Predict provider performance for given input."""
        params = {
            "provider": provider,
            "model": model,
            "input_characteristics": input_characteristics or {},
            **kwargs
        }
        
        return await self.brokle_client._make_request(
            "POST",
            "/routing/predict-performance",
            params
        )
    
    def predict_provider_performance_sync(self, provider: str, model: str, **kwargs) -> Dict[str, Any]:
        """Predict provider performance synchronously."""
        return asyncio.run(self.predict_provider_performance(provider, model, **kwargs))
    
    # Feedback and Learning
    async def submit_routing_feedback(
        self,
        request_id: str,
        recommended_provider: str,
        actual_provider: str,
        *,
        satisfaction_score: Optional[float] = None,
        performance_metrics: Optional[Dict[str, float]] = None,
        issues_encountered: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Submit feedback on routing decisions."""
        feedback_data = {
            "request_id": request_id,
            "recommended_provider": recommended_provider,
            "actual_provider": actual_provider,
            "satisfaction_score": satisfaction_score,
            "performance_metrics": performance_metrics or {},
            "issues_encountered": issues_encountered or [],
            **kwargs
        }
        
        return await self.brokle_client._make_request(
            "POST",
            "/routing/feedback",
            feedback_data
        )
    
    def submit_routing_feedback_sync(
        self,
        request_id: str,
        recommended_provider: str,
        actual_provider: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Submit routing feedback synchronously."""
        return asyncio.run(self.submit_routing_feedback(
            request_id, recommended_provider, actual_provider, **kwargs
        ))
    
    async def get_routing_accuracy_metrics(
        self,
        *,
        time_period: str = "24h",
        strategy: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Get routing accuracy metrics."""
        params = {
            "time_period": time_period,
            "strategy": strategy,
            **kwargs
        }
        
        return await self.brokle_client._make_request(
            "GET",
            "/routing/accuracy-metrics",
            {k: v for k, v in params.items() if v is not None}
        )
    
    def get_routing_accuracy_metrics_sync(self, **kwargs) -> Dict[str, Any]:
        """Get routing accuracy metrics synchronously."""
        return asyncio.run(self.get_routing_accuracy_metrics(**kwargs))
    
    # A/B Testing Support
    async def create_routing_experiment(
        self,
        name: str,
        strategies: List[Dict[str, Any]],
        *,
        traffic_split: Optional[Dict[str, float]] = None,
        duration_hours: Optional[int] = None,
        success_metrics: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a routing A/B test experiment."""
        experiment_data = {
            "name": name,
            "strategies": strategies,
            "traffic_split": traffic_split or {},
            "duration_hours": duration_hours,
            "success_metrics": success_metrics or [],
            **kwargs
        }
        
        return await self.brokle_client._make_request(
            "POST",
            "/routing/experiments",
            experiment_data
        )
    
    def create_routing_experiment_sync(
        self,
        name: str,
        strategies: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Create routing experiment synchronously."""
        return asyncio.run(self.create_routing_experiment(name, strategies, **kwargs))
    
    async def get_experiment_results(
        self,
        experiment_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Get A/B test experiment results."""
        return await self.brokle_client._make_request(
            "GET",
            f"/routing/experiments/{experiment_id}/results",
            kwargs
        )
    
    def get_experiment_results_sync(self, experiment_id: str, **kwargs) -> Dict[str, Any]:
        """Get experiment results synchronously."""
        return asyncio.run(self.get_experiment_results(experiment_id, **kwargs))