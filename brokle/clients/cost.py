"""
Cost tracking client for real-time cost optimization via cost-tracking-service.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union

from ..types.requests import (
    CostCalculationRequest,
    CostTrackingRequest,
    BudgetRequest,
    CostComparisonRequest,
)
from ..types.responses import (
    CostCalculationResponse,
    CostTrackingResponse,
    BudgetResponse,
    CostComparisonResponse,
    CostTrendResponse,
)

logger = logging.getLogger(__name__)


class CostClient:
    """Client for real-time cost optimization via cost-tracking-service."""
    
    def __init__(self, brokle_client: 'Brokle'):
        self.brokle_client = brokle_client
    
    # Core Cost Operations
    async def calculate_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        request_type: str,
        *,
        additional_costs: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> CostCalculationResponse:
        """Calculate cost for a request."""
        request = CostCalculationRequest(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            request_type=request_type,
            additional_costs=additional_costs
        )
        
        return await self.brokle_client._make_request(
            "POST",
            "/api/v1/cost/calculate",
            request.model_dump(exclude_none=True),
            response_model=CostCalculationResponse
        )
    
    def calculate_cost_sync(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        request_type: str,
        **kwargs
    ) -> CostCalculationResponse:
        """Calculate cost synchronously."""
        return asyncio.run(self.calculate_cost(
            provider, model, input_tokens, output_tokens, request_type, **kwargs
        ))
    
    async def track_cost(
        self,
        request_id: str,
        organization_id: str,
        project_id: str,
        environment_id: str,
        provider: str,
        model: str,
        calculated_cost: float,
        input_tokens: int,
        output_tokens: int,
        *,
        actual_cost: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> CostTrackingResponse:
        """Track actual cost for a request."""
        request = CostTrackingRequest(
            request_id=request_id,
            organization_id=organization_id,
            project_id=project_id,
            environment_id=environment_id,
            provider=provider,
            model=model,
            calculated_cost=calculated_cost,
            actual_cost=actual_cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata=metadata
        )
        
        return await self.brokle_client._make_request(
            "POST",
            "/api/v1/cost/track",
            request.model_dump(exclude_none=True),
            response_model=CostTrackingResponse
        )
    
    def track_cost_sync(
        self,
        request_id: str,
        organization_id: str,
        project_id: str,
        environment_id: str,
        provider: str,
        model: str,
        calculated_cost: float,
        input_tokens: int,
        output_tokens: int,
        **kwargs
    ) -> CostTrackingResponse:
        """Track cost synchronously."""
        return asyncio.run(self.track_cost(
            request_id, organization_id, project_id, environment_id,
            provider, model, calculated_cost, input_tokens, output_tokens, **kwargs
        ))
    
    async def compare_costs(
        self,
        providers: List[str],
        model_mappings: Dict[str, str],
        input_tokens: int,
        output_tokens: int,
        request_type: str,
        **kwargs
    ) -> CostComparisonResponse:
        """Compare costs across providers."""
        request = CostComparisonRequest(
            providers=providers,
            model_mappings=model_mappings,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            request_type=request_type
        )
        
        return await self.brokle_client._make_request(
            "POST",
            "/api/v1/cost/compare",
            request.model_dump(exclude_none=True),
            response_model=CostComparisonResponse
        )
    
    def compare_costs_sync(
        self,
        providers: List[str],
        model_mappings: Dict[str, str],
        input_tokens: int,
        output_tokens: int,
        request_type: str,
        **kwargs
    ) -> CostComparisonResponse:
        """Compare costs synchronously."""
        return asyncio.run(self.compare_costs(
            providers, model_mappings, input_tokens, output_tokens, request_type, **kwargs
        ))
    
    async def get_cost_trends(
        self,
        organization_id: str,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        granularity: str = "daily",
        **kwargs
    ) -> CostTrendResponse:
        """Get cost trends for organization."""
        params = {
            "start_date": start_date,
            "end_date": end_date,
            "granularity": granularity,
            **kwargs
        }
        
        return await self.brokle_client._make_request(
            "GET",
            f"/api/v1/cost/trends/{organization_id}",
            {k: v for k, v in params.items() if v is not None},
            response_model=CostTrendResponse
        )
    
    def get_cost_trends_sync(self, organization_id: str, **kwargs) -> CostTrendResponse:
        """Get cost trends synchronously."""
        return asyncio.run(self.get_cost_trends(organization_id, **kwargs))
    
    # Budget Management
    async def create_budget(
        self,
        organization_id: str,
        budget_type: str,
        amount: float,
        *,
        project_id: Optional[str] = None,
        environment_id: Optional[str] = None,
        alert_thresholds: Optional[List[float]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> BudgetResponse:
        """Create a new budget."""
        request = BudgetRequest(
            organization_id=organization_id,
            project_id=project_id,
            environment_id=environment_id,
            budget_type=budget_type,
            amount=amount,
            alert_thresholds=alert_thresholds,
            start_date=start_date,
            end_date=end_date
        )
        
        return await self.brokle_client._make_request(
            "POST",
            "/api/v1/budget",
            request.model_dump(exclude_none=True),
            response_model=BudgetResponse
        )
    
    def create_budget_sync(
        self,
        organization_id: str,
        budget_type: str,
        amount: float,
        **kwargs
    ) -> BudgetResponse:
        """Create budget synchronously."""
        return asyncio.run(self.create_budget(organization_id, budget_type, amount, **kwargs))
    
    async def update_budget(
        self,
        budget_id: str,
        **updates
    ) -> BudgetResponse:
        """Update budget."""
        return await self.brokle_client._make_request(
            "PUT",
            f"/api/v1/budget/{budget_id}",
            updates,
            response_model=BudgetResponse
        )
    
    def update_budget_sync(self, budget_id: str, **updates) -> BudgetResponse:
        """Update budget synchronously."""
        return asyncio.run(self.update_budget(budget_id, **updates))
    
    async def get_budget(self, budget_id: str) -> BudgetResponse:
        """Get budget by ID."""
        return await self.brokle_client._make_request(
            "GET",
            f"/api/v1/budget/{budget_id}",
            response_model=BudgetResponse
        )
    
    def get_budget_sync(self, budget_id: str) -> BudgetResponse:
        """Get budget synchronously."""
        return asyncio.run(self.get_budget(budget_id))
    
    async def delete_budget(self, budget_id: str) -> Dict[str, Any]:
        """Delete budget."""
        return await self.brokle_client._make_request(
            "DELETE",
            f"/api/v1/budget/{budget_id}"
        )
    
    def delete_budget_sync(self, budget_id: str) -> Dict[str, Any]:
        """Delete budget synchronously."""
        return asyncio.run(self.delete_budget(budget_id))
    
    async def check_budget(
        self,
        organization_id: str,
        project_id: str,
        environment_id: str,
        amount: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Check budget status before spending."""
        params = {
            "organization_id": organization_id,
            "project_id": project_id,
            "environment_id": environment_id,
            "amount": amount,
            **kwargs
        }
        
        return await self.brokle_client._make_request(
            "POST",
            "/api/v1/budget/check",
            params
        )
    
    def check_budget_sync(
        self,
        organization_id: str,
        project_id: str,
        environment_id: str,
        amount: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Check budget synchronously."""
        return asyncio.run(self.check_budget(organization_id, project_id, environment_id, amount, **kwargs))
    
    async def get_budget_status(
        self,
        organization_id: str,
        project_id: str,
        environment_id: str,
        **kwargs
    ) -> BudgetResponse:
        """Get budget status."""
        return await self.brokle_client._make_request(
            "GET",
            f"/api/v1/budget/status/{organization_id}/{project_id}/{environment_id}",
            {k: v for k, v in kwargs.items() if v is not None},
            response_model=BudgetResponse
        )
    
    def get_budget_status_sync(
        self,
        organization_id: str,
        project_id: str,
        environment_id: str,
        **kwargs
    ) -> BudgetResponse:
        """Get budget status synchronously."""
        return asyncio.run(self.get_budget_status(organization_id, project_id, environment_id, **kwargs))
    
    async def get_budget_alerts(self, budget_id: str) -> List[Dict[str, Any]]:
        """Get budget alerts."""
        response = await self.brokle_client._make_request(
            "GET",
            f"/api/v1/budget/alerts/{budget_id}"
        )
        
        if isinstance(response, dict) and "alerts" in response:
            return response["alerts"]
        return []
    
    def get_budget_alerts_sync(self, budget_id: str) -> List[Dict[str, Any]]:
        """Get budget alerts synchronously."""
        return asyncio.run(self.get_budget_alerts(budget_id))
    
    async def get_budget_utilization(
        self,
        organization_id: str,
        *,
        period: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Get budget utilization for organization."""
        params = {
            "period": period,
            **kwargs
        }
        
        return await self.brokle_client._make_request(
            "GET",
            f"/api/v1/budget/utilization/{organization_id}",
            {k: v for k, v in params.items() if v is not None}
        )
    
    def get_budget_utilization_sync(self, organization_id: str, **kwargs) -> Dict[str, Any]:
        """Get budget utilization synchronously."""
        return asyncio.run(self.get_budget_utilization(organization_id, **kwargs))
    
    # Cost Optimization Helpers
    async def get_cost_optimization_recommendations(
        self,
        organization_id: str,
        *,
        time_period: str = "30d",
        min_savings_threshold: float = 0.1,
        **kwargs
    ) -> Dict[str, Any]:
        """Get cost optimization recommendations."""
        # This would combine multiple API calls to provide recommendations
        
        # Get cost trends
        trends = await self.get_cost_trends(organization_id, **kwargs)
        
        # Get provider comparisons for recent requests
        # This is a simplified implementation - in practice would analyze actual usage patterns
        recommendations = {
            "organization_id": organization_id,
            "time_period": time_period,
            "total_potential_savings": 0.0,
            "recommendations": [],
            "trends": trends.model_dump() if hasattr(trends, 'model_dump') else trends
        }
        
        return recommendations
    
    def get_cost_optimization_recommendations_sync(self, organization_id: str, **kwargs) -> Dict[str, Any]:
        """Get cost optimization recommendations synchronously."""
        return asyncio.run(self.get_cost_optimization_recommendations(organization_id, **kwargs))
    
    async def estimate_monthly_cost(
        self,
        organization_id: str,
        *,
        usage_projection: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Estimate monthly cost based on current usage patterns."""
        # Get recent cost trends
        trends = await self.get_cost_trends(organization_id, **kwargs)
        
        # Calculate projection based on trends and usage patterns
        # This is simplified - would use actual statistical modeling
        projection = {
            "organization_id": organization_id,
            "estimated_monthly_cost": 0.0,
            "confidence_level": 0.0,
            "based_on_days": 0,
            "cost_breakdown": {},
            "growth_rate": 0.0,
            "trends": trends.model_dump() if hasattr(trends, 'model_dump') else trends
        }
        
        return projection
    
    def estimate_monthly_cost_sync(self, organization_id: str, **kwargs) -> Dict[str, Any]:
        """Estimate monthly cost synchronously."""
        return asyncio.run(self.estimate_monthly_cost(organization_id, **kwargs))
    
    async def analyze_cost_variance(
        self,
        organization_id: str,
        *,
        period: str = "7d",
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze variance between calculated and actual costs."""
        params = {
            "organization_id": organization_id,
            "period": period,
            **kwargs
        }
        
        # This would analyze the difference between calculated and actual costs
        # to improve cost calculation accuracy
        variance_analysis = {
            "organization_id": organization_id,
            "period": period,
            "average_variance": 0.0,
            "variance_by_provider": {},
            "variance_by_model": {},
            "accuracy_score": 0.0,
            "recommendations": []
        }
        
        return variance_analysis
    
    def analyze_cost_variance_sync(self, organization_id: str, **kwargs) -> Dict[str, Any]:
        """Analyze cost variance synchronously."""
        return asyncio.run(self.analyze_cost_variance(organization_id, **kwargs))