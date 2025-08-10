"""
Enhanced Analytics client for comprehensive AI observability via analytics-service.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from ..types.requests import AnalyticsRequest
from ..types.responses import AnalyticsResponse, AnalyticsMetric

logger = logging.getLogger(__name__)


class EnhancedAnalyticsClient:
    """Enhanced client for comprehensive AI analytics via analytics-service."""
    
    def __init__(self, brokle_client: 'Brokle'):
        self.brokle_client = brokle_client
    
    # Core Metrics Methods
    async def get_metrics(self, **kwargs) -> AnalyticsResponse:
        """Get analytics metrics."""
        request = AnalyticsRequest(**kwargs)
        return await self.brokle_client._make_request(
            "GET",
            "/api/v1/metrics",
            request.model_dump(exclude_none=True),
            response_model=AnalyticsResponse
        )
    
    def get_metrics_sync(self, **kwargs) -> AnalyticsResponse:
        """Get analytics metrics synchronously."""
        return asyncio.run(self.get_metrics(**kwargs))
    
    async def record_metric(
        self,
        name: str,
        value: Union[int, float, str],
        *,
        timestamp: Optional[str] = None,
        dimensions: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Record a single metric."""
        metric_data = {
            "name": name,
            "value": value,
            "timestamp": timestamp or datetime.now().isoformat(),
            "dimensions": dimensions or {},
            **kwargs
        }
        
        return await self.brokle_client._make_request(
            "POST",
            "/api/v1/metrics/record",
            metric_data
        )
    
    def record_metric_sync(self, name: str, value: Union[int, float, str], **kwargs) -> Dict[str, Any]:
        """Record metric synchronously."""
        return asyncio.run(self.record_metric(name, value, **kwargs))
    
    async def record_metrics_bulk(
        self,
        metrics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Record multiple metrics in bulk."""
        return await self.brokle_client._make_request(
            "POST",
            "/api/v1/metrics/record/bulk",
            {"metrics": metrics}
        )
    
    def record_metrics_bulk_sync(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Record bulk metrics synchronously."""
        return asyncio.run(self.record_metrics_bulk(metrics))
    
    async def get_time_series(
        self,
        *,
        metric_name: str,
        start_time: str,
        end_time: str,
        granularity: str = "1h",
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Get time series data for a metric."""
        params = {
            "metric_name": metric_name,
            "start_time": start_time,
            "end_time": end_time,
            "granularity": granularity,
            "filters": filters or {},
            **kwargs
        }
        
        return await self.brokle_client._make_request(
            "GET",
            "/api/v1/metrics/time-series",
            params
        )
    
    def get_time_series_sync(self, **kwargs) -> Dict[str, Any]:
        """Get time series synchronously."""
        return asyncio.run(self.get_time_series(**kwargs))
    
    async def get_real_time_metrics(self) -> AnalyticsResponse:
        """Get real-time metrics."""
        return await self.brokle_client._make_request(
            "GET",
            "/api/v1/metrics/real-time",
            response_model=AnalyticsResponse
        )
    
    def get_real_time_metrics_sync(self) -> AnalyticsResponse:
        """Get real-time metrics synchronously."""
        return asyncio.run(self.get_real_time_metrics())
    
    async def get_aggregated_metrics(
        self,
        *,
        metrics: List[str],
        aggregation: str = "sum",
        group_by: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Get aggregated metrics."""
        params = {
            "metrics": metrics,
            "aggregation": aggregation,
            "group_by": group_by or [],
            "filters": filters or {},
            **kwargs
        }
        
        return await self.brokle_client._make_request(
            "GET",
            "/api/v1/metrics/aggregated",
            params
        )
    
    def get_aggregated_metrics_sync(self, **kwargs) -> Dict[str, Any]:
        """Get aggregated metrics synchronously."""
        return asyncio.run(self.get_aggregated_metrics(**kwargs))
    
    # AI-Specific Analytics
    async def get_usage_analytics(
        self,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        organization_id: Optional[str] = None,
        project_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Get AI usage analytics."""
        params = {
            "start_date": start_date,
            "end_date": end_date,
            "organization_id": organization_id,
            "project_id": project_id,
            **kwargs
        }
        
        return await self.brokle_client._make_request(
            "GET",
            "/api/v1/metrics/usage",
            {k: v for k, v in params.items() if v is not None}
        )
    
    def get_usage_analytics_sync(self, **kwargs) -> Dict[str, Any]:
        """Get usage analytics synchronously."""
        return asyncio.run(self.get_usage_analytics(**kwargs))
    
    async def get_performance_analytics(
        self,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        time_range: str = "24h",
        **kwargs
    ) -> Dict[str, Any]:
        """Get AI performance analytics."""
        params = {
            "provider": provider,
            "model": model,
            "time_range": time_range,
            **kwargs
        }
        
        return await self.brokle_client._make_request(
            "GET",
            "/api/v1/metrics/performance",
            {k: v for k, v in params.items() if v is not None}
        )
    
    def get_performance_analytics_sync(self, **kwargs) -> Dict[str, Any]:
        """Get performance analytics synchronously."""
        return asyncio.run(self.get_performance_analytics(**kwargs))
    
    async def get_cost_analytics(
        self,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        breakdown_by: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Get cost analytics."""
        params = {
            "start_date": start_date,
            "end_date": end_date,
            "breakdown_by": breakdown_by,
            **kwargs
        }
        
        return await self.brokle_client._make_request(
            "GET",
            "/api/v1/metrics/cost",
            {k: v for k, v in params.items() if v is not None}
        )
    
    def get_cost_analytics_sync(self, **kwargs) -> Dict[str, Any]:
        """Get cost analytics synchronously."""
        return asyncio.run(self.get_cost_analytics(**kwargs))
    
    # Dashboard Management
    async def create_dashboard(
        self,
        *,
        name: str,
        description: Optional[str] = None,
        organization_id: str,
        project_id: Optional[str] = None,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Create a new dashboard."""
        dashboard_data = {
            "name": name,
            "description": description,
            "organization_id": organization_id,
            "project_id": project_id,
            "config": config,
            **kwargs
        }
        
        return await self.brokle_client._make_request(
            "POST",
            "/api/v1/dashboards",
            dashboard_data
        )
    
    def create_dashboard_sync(self, **kwargs) -> Dict[str, Any]:
        """Create dashboard synchronously."""
        return asyncio.run(self.create_dashboard(**kwargs))
    
    async def get_dashboard(self, dashboard_id: str) -> Dict[str, Any]:
        """Get dashboard by ID."""
        return await self.brokle_client._make_request(
            "GET",
            f"/api/v1/dashboards/{dashboard_id}"
        )
    
    def get_dashboard_sync(self, dashboard_id: str) -> Dict[str, Any]:
        """Get dashboard synchronously."""
        return asyncio.run(self.get_dashboard(dashboard_id))
    
    async def update_dashboard(
        self,
        dashboard_id: str,
        **updates
    ) -> Dict[str, Any]:
        """Update dashboard."""
        return await self.brokle_client._make_request(
            "PUT",
            f"/api/v1/dashboards/{dashboard_id}",
            updates
        )
    
    def update_dashboard_sync(self, dashboard_id: str, **updates) -> Dict[str, Any]:
        """Update dashboard synchronously."""
        return asyncio.run(self.update_dashboard(dashboard_id, **updates))
    
    async def delete_dashboard(self, dashboard_id: str) -> Dict[str, Any]:
        """Delete dashboard."""
        return await self.brokle_client._make_request(
            "DELETE",
            f"/api/v1/dashboards/{dashboard_id}"
        )
    
    def delete_dashboard_sync(self, dashboard_id: str) -> Dict[str, Any]:
        """Delete dashboard synchronously."""
        return asyncio.run(self.delete_dashboard(dashboard_id))
    
    async def get_organization_dashboards(self, org_id: str) -> List[Dict[str, Any]]:
        """Get dashboards for organization."""
        response = await self.brokle_client._make_request(
            "GET",
            f"/api/v1/dashboards/organization/{org_id}"
        )
        
        if isinstance(response, dict) and "dashboards" in response:
            return response["dashboards"]
        return []
    
    def get_organization_dashboards_sync(self, org_id: str) -> List[Dict[str, Any]]:
        """Get organization dashboards synchronously."""
        return asyncio.run(self.get_organization_dashboards(org_id))
    
    async def get_project_dashboards(self, project_id: str) -> List[Dict[str, Any]]:
        """Get dashboards for project."""
        response = await self.brokle_client._make_request(
            "GET",
            f"/api/v1/dashboards/project/{project_id}"
        )
        
        if isinstance(response, dict) and "dashboards" in response:
            return response["dashboards"]
        return []
    
    def get_project_dashboards_sync(self, project_id: str) -> List[Dict[str, Any]]:
        """Get project dashboards synchronously."""
        return asyncio.run(self.get_project_dashboards(project_id))
    
    async def get_dashboard_data(
        self,
        dashboard_id: str,
        *,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Get dashboard data."""
        params = {
            "start_time": start_time,
            "end_time": end_time,
            **kwargs
        }
        
        return await self.brokle_client._make_request(
            "GET",
            f"/api/v1/dashboards/{dashboard_id}/data",
            {k: v for k, v in params.items() if v is not None}
        )
    
    def get_dashboard_data_sync(self, dashboard_id: str, **kwargs) -> Dict[str, Any]:
        """Get dashboard data synchronously."""
        return asyncio.run(self.get_dashboard_data(dashboard_id, **kwargs))
    
    async def get_realtime_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        """Get realtime dashboard data."""
        return await self.brokle_client._make_request(
            "GET",
            f"/api/v1/dashboards/{dashboard_id}/realtime"
        )
    
    def get_realtime_dashboard_data_sync(self, dashboard_id: str) -> Dict[str, Any]:
        """Get realtime dashboard data synchronously."""
        return asyncio.run(self.get_realtime_dashboard_data(dashboard_id))
    
    async def get_widget_data(
        self,
        *,
        widget_config: Dict[str, Any],
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Get widget data."""
        params = {
            "widget_config": widget_config,
            "start_time": start_time,
            "end_time": end_time,
            **kwargs
        }
        
        return await self.brokle_client._make_request(
            "POST",
            "/api/v1/dashboards/widget-data",
            params
        )
    
    def get_widget_data_sync(self, **kwargs) -> Dict[str, Any]:
        """Get widget data synchronously."""
        return asyncio.run(self.get_widget_data(**kwargs))
    
    async def execute_custom_query(
        self,
        *,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute custom query."""
        params = {
            "query": query,
            "parameters": parameters or {},
            **kwargs
        }
        
        return await self.brokle_client._make_request(
            "POST",
            "/api/v1/dashboards/query",
            params
        )
    
    def execute_custom_query_sync(self, **kwargs) -> Dict[str, Any]:
        """Execute custom query synchronously."""
        return asyncio.run(self.execute_custom_query(**kwargs))
    
    # Provider Analytics
    async def create_provider_analytics(
        self,
        *,
        provider: str,
        metrics: Dict[str, Any],
        timestamp: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create provider analytics."""
        data = {
            "provider": provider,
            "metrics": metrics,
            "timestamp": timestamp or datetime.now().isoformat(),
            **kwargs
        }
        
        return await self.brokle_client._make_request(
            "POST",
            "/api/v1/providers/analytics",
            data
        )
    
    def create_provider_analytics_sync(self, **kwargs) -> Dict[str, Any]:
        """Create provider analytics synchronously."""
        return asyncio.run(self.create_provider_analytics(**kwargs))
    
    async def get_provider_analytics(
        self,
        *,
        provider: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Get provider analytics."""
        params = {
            "provider": provider,
            "start_time": start_time,
            "end_time": end_time,
            **kwargs
        }
        
        return await self.brokle_client._make_request(
            "GET",
            "/api/v1/providers/analytics",
            {k: v for k, v in params.items() if v is not None}
        )
    
    def get_provider_analytics_sync(self, **kwargs) -> Dict[str, Any]:
        """Get provider analytics synchronously."""
        return asyncio.run(self.get_provider_analytics(**kwargs))
    
    async def get_provider_performance_comparison(
        self,
        *,
        providers: Optional[List[str]] = None,
        metric: str = "latency",
        time_range: str = "24h",
        **kwargs
    ) -> Dict[str, Any]:
        """Get provider performance comparison."""
        params = {
            "providers": providers,
            "metric": metric,
            "time_range": time_range,
            **kwargs
        }
        
        return await self.brokle_client._make_request(
            "GET",
            "/api/v1/providers/performance/comparison",
            {k: v for k, v in params.items() if v is not None}
        )
    
    def get_provider_performance_comparison_sync(self, **kwargs) -> Dict[str, Any]:
        """Get provider performance comparison synchronously."""
        return asyncio.run(self.get_provider_performance_comparison(**kwargs))
    
    async def get_provider_rankings(
        self,
        *,
        criteria: str = "overall",
        time_range: str = "7d",
        **kwargs
    ) -> Dict[str, Any]:
        """Get provider rankings."""
        params = {
            "criteria": criteria,
            "time_range": time_range,
            **kwargs
        }
        
        return await self.brokle_client._make_request(
            "GET",
            "/api/v1/providers/rankings",
            params
        )
    
    def get_provider_rankings_sync(self, **kwargs) -> Dict[str, Any]:
        """Get provider rankings synchronously."""
        return asyncio.run(self.get_provider_rankings(**kwargs))
    
    async def get_provider_recommendations(
        self,
        *,
        use_case: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Get provider recommendations."""
        params = {
            "use_case": use_case,
            "constraints": constraints or {},
            **kwargs
        }
        
        return await self.brokle_client._make_request(
            "GET",
            "/api/v1/providers/recommendations",
            {k: v for k, v in params.items() if v is not None}
        )
    
    def get_provider_recommendations_sync(self, **kwargs) -> Dict[str, Any]:
        """Get provider recommendations synchronously."""
        return asyncio.run(self.get_provider_recommendations(**kwargs))