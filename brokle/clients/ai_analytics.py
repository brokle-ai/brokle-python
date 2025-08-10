"""
AI-focused analytics client for developers using the Brokle Platform.
Provides simple insights relevant to AI development, not platform administration.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta

from ..types.requests import AnalyticsRequest
from ..types.responses import AnalyticsResponse

logger = logging.getLogger(__name__)


class AIAnalyticsClient:
    """Simplified analytics client focused on AI usage insights for developers."""
    
    def __init__(self, brokle_client: 'Brokle'):
        self.brokle_client = brokle_client
    
    # Core AI Usage Analytics
    async def get_usage_summary(
        self,
        *,
        period: str = "7d",
        **kwargs
    ) -> Dict[str, Any]:
        """Get simple AI usage summary for developers."""
        request = AnalyticsRequest(
            start_date=(datetime.now() - timedelta(days=7)).isoformat() if period == "7d" else None,
            end_date=datetime.now().isoformat(),
            metrics=["total_requests", "total_cost", "avg_quality", "top_models", "top_providers"],
            granularity="daily"
        )
        
        response = await self.brokle_client._make_request(
            "GET",
            "/api/v1/metrics/usage",
            request.model_dump(exclude_none=True)
        )
        
        # Simplify response for developers
        if isinstance(response, dict):
            return {
                "period": period,
                "total_requests": response.get("total_requests", 0),
                "total_cost_usd": response.get("total_cost", 0.0),
                "average_cost_per_request": response.get("avg_cost_per_request", 0.0),
                "average_quality_score": response.get("avg_quality_score"),
                "most_used_models": response.get("top_models", [])[:5],
                "provider_breakdown": response.get("provider_breakdown", {}),
                "daily_usage": response.get("daily_breakdown", [])
            }
        return {}
    
    def get_usage_summary_sync(self, **kwargs) -> Dict[str, Any]:
        """Get usage summary synchronously."""
        return asyncio.run(self.get_usage_summary(**kwargs))
    
    async def get_cost_trends(
        self,
        *,
        days: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """Get cost trends for budgeting and planning."""
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        end_date = datetime.now().isoformat()
        
        response = await self.brokle_client._make_request(
            "GET",
            "/api/v1/metrics/cost",
            {
                "start_date": start_date,
                "end_date": end_date,
                "granularity": "daily",
                "breakdown_by": "provider,model"
            }
        )
        
        if isinstance(response, dict):
            return {
                "period_days": days,
                "total_cost_usd": response.get("total_cost", 0.0),
                "daily_costs": response.get("daily_breakdown", []),
                "cost_by_provider": response.get("provider_breakdown", {}),
                "cost_by_model": response.get("model_breakdown", {}),
                "projected_monthly_cost": response.get("projected_monthly", 0.0),
                "cost_trend": response.get("trend", "stable")  # increasing, decreasing, stable
            }
        return {}
    
    def get_cost_trends_sync(self, **kwargs) -> Dict[str, Any]:
        """Get cost trends synchronously."""
        return asyncio.run(self.get_cost_trends(**kwargs))
    
    async def get_performance_metrics(
        self,
        *,
        period: str = "24h",
        **kwargs
    ) -> Dict[str, Any]:
        """Get AI performance insights for optimization."""
        response = await self.brokle_client._make_request(
            "GET",
            "/api/v1/metrics/performance",
            {
                "time_range": period,
                "metrics": ["latency", "cache_hit_rate", "error_rate", "quality_scores"]
            }
        )
        
        if isinstance(response, dict):
            return {
                "period": period,
                "average_latency_ms": response.get("avg_latency_ms", 0),
                "cache_hit_rate": response.get("cache_hit_rate", 0.0),
                "error_rate": response.get("error_rate", 0.0),
                "average_quality_score": response.get("avg_quality_score"),
                "fastest_provider": response.get("fastest_provider"),
                "most_reliable_provider": response.get("most_reliable_provider"),
                "performance_by_model": response.get("model_performance", {})
            }
        return {}
    
    def get_performance_metrics_sync(self, **kwargs) -> Dict[str, Any]:
        """Get performance metrics synchronously."""
        return asyncio.run(self.get_performance_metrics(**kwargs))
    
    async def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time AI usage metrics."""
        response = await self.brokle_client._make_request(
            "GET",
            "/api/v1/metrics/real-time"
        )
        
        if isinstance(response, dict):
            return {
                "current_requests_per_minute": response.get("rpm", 0),
                "current_cost_per_hour": response.get("cost_per_hour", 0.0),
                "active_models": response.get("active_models", []),
                "current_cache_hit_rate": response.get("cache_hit_rate", 0.0),
                "current_average_latency": response.get("avg_latency_ms", 0),
                "current_error_rate": response.get("error_rate", 0.0),
                "provider_health": response.get("provider_health", {})
            }
        return {}
    
    def get_real_time_metrics_sync(self) -> Dict[str, Any]:
        """Get real-time metrics synchronously."""
        return asyncio.run(self.get_real_time_metrics())
    
    # AI Model and Provider Insights
    async def get_model_comparison(
        self,
        models: List[str],
        *,
        metric: str = "cost_efficiency",
        period: str = "7d",
        **kwargs
    ) -> Dict[str, Any]:
        """Compare AI models on cost, performance, and quality."""
        response = await self.brokle_client._make_request(
            "GET",
            "/api/v1/providers/performance/comparison",
            {
                "models": models,
                "metric": metric,
                "time_range": period
            }
        )
        
        if isinstance(response, dict):
            return {
                "models_compared": models,
                "comparison_metric": metric,
                "period": period,
                "best_model": response.get("recommended_model"),
                "model_scores": response.get("model_scores", {}),
                "cost_comparison": response.get("cost_comparison", {}),
                "performance_comparison": response.get("performance_comparison", {}),
                "quality_comparison": response.get("quality_comparison", {})
            }
        return {}
    
    def get_model_comparison_sync(self, models: List[str], **kwargs) -> Dict[str, Any]:
        """Compare models synchronously."""
        return asyncio.run(self.get_model_comparison(models, **kwargs))
    
    async def get_provider_insights(
        self,
        *,
        period: str = "7d",
        **kwargs
    ) -> Dict[str, Any]:
        """Get insights about AI provider performance."""
        response = await self.brokle_client._make_request(
            "GET",
            "/api/v1/providers/analytics",
            {
                "time_range": period,
                "include_health": True,
                "include_costs": True
            }
        )
        
        if isinstance(response, dict):
            return {
                "period": period,
                "provider_rankings": response.get("rankings", []),
                "cost_leaders": response.get("cost_leaders", []),
                "performance_leaders": response.get("performance_leaders", []),
                "quality_leaders": response.get("quality_leaders", []),
                "provider_health_scores": response.get("health_scores", {}),
                "routing_recommendations": response.get("recommendations", [])
            }
        return {}
    
    def get_provider_insights_sync(self, **kwargs) -> Dict[str, Any]:
        """Get provider insights synchronously."""
        return asyncio.run(self.get_provider_insights(**kwargs))
    
    # Quality and Evaluation Insights
    async def get_quality_trends(
        self,
        *,
        period: str = "7d",
        **kwargs
    ) -> Dict[str, Any]:
        """Get AI response quality trends."""
        response = await self.brokle_client._make_request(
            "GET",
            "/api/v1/quality-metrics",
            {
                "time_range": period,
                "include_trends": True
            }
        )
        
        if isinstance(response, dict):
            return {
                "period": period,
                "average_quality_score": response.get("avg_quality", 0.0),
                "quality_trend": response.get("trend", "stable"),
                "quality_by_model": response.get("model_quality", {}),
                "quality_by_provider": response.get("provider_quality", {}),
                "improvement_suggestions": response.get("suggestions", []),
                "quality_alerts": response.get("alerts", [])
            }
        return {}
    
    def get_quality_trends_sync(self, **kwargs) -> Dict[str, Any]:
        """Get quality trends synchronously."""
        return asyncio.run(self.get_quality_trends(**kwargs))
    
    # Simple Dashboard Data
    async def get_dashboard_data(
        self,
        *,
        period: str = "24h",
        **kwargs
    ) -> Dict[str, Any]:
        """Get all key metrics for a simple AI dashboard."""
        # Fetch multiple metrics in parallel for dashboard
        import asyncio
        
        usage_task = asyncio.create_task(self.get_usage_summary(period=period))
        performance_task = asyncio.create_task(self.get_performance_metrics(period=period))
        realtime_task = asyncio.create_task(self.get_real_time_metrics())
        
        usage, performance, realtime = await asyncio.gather(
            usage_task, performance_task, realtime_task, return_exceptions=True
        )
        
        return {
            "period": period,
            "updated_at": datetime.now().isoformat(),
            "usage_summary": usage if not isinstance(usage, Exception) else {},
            "performance_metrics": performance if not isinstance(performance, Exception) else {},
            "real_time_metrics": realtime if not isinstance(realtime, Exception) else {},
            "status": "healthy" if all(not isinstance(r, Exception) for r in [usage, performance, realtime]) else "partial"
        }
    
    def get_dashboard_data_sync(self, **kwargs) -> Dict[str, Any]:
        """Get dashboard data synchronously."""
        return asyncio.run(self.get_dashboard_data(**kwargs))
    
    # Cost Optimization Helpers
    async def get_cost_optimization_tips(self, **kwargs) -> Dict[str, Any]:
        """Get simple cost optimization recommendations."""
        usage = await self.get_usage_summary()
        performance = await self.get_performance_metrics()
        
        tips = []
        savings_potential = 0.0
        
        # Simple optimization logic based on usage patterns
        if usage.get("average_cost_per_request", 0) > 0.01:
            tips.append("Consider using more cost-effective models for simple tasks")
            savings_potential += usage.get("total_cost_usd", 0) * 0.2
        
        if performance.get("cache_hit_rate", 0) < 0.5:
            tips.append("Enable semantic caching to reduce duplicate requests")
            savings_potential += usage.get("total_cost_usd", 0) * 0.15
        
        if len(usage.get("most_used_models", [])) > 5:
            tips.append("Consolidate to fewer models to optimize routing")
            savings_potential += usage.get("total_cost_usd", 0) * 0.1
        
        return {
            "potential_monthly_savings_usd": min(savings_potential, usage.get("total_cost_usd", 0) * 0.4),
            "optimization_tips": tips,
            "current_efficiency_score": min(100, max(0, 100 - len(tips) * 20)),
            "top_cost_drivers": usage.get("most_used_models", [])[:3]
        }
    
    def get_cost_optimization_tips_sync(self, **kwargs) -> Dict[str, Any]:
        """Get cost optimization tips synchronously."""
        return asyncio.run(self.get_cost_optimization_tips(**kwargs))