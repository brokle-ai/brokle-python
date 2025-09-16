"""
Cost efficiency evaluator for measuring AI generation cost effectiveness.

Evaluates the balance between response quality and computational cost,
providing insights for cost optimization in AI applications.
"""

import logging
from typing import Any, Dict, Optional, Union
from datetime import datetime, timedelta

from ..base import BaseEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


class CostEfficiencyEvaluator(BaseEvaluator):
    """
    Evaluates cost efficiency of AI responses.

    Considers multiple factors:
    - Token usage vs response quality
    - Latency vs cost trade-offs
    - Provider cost differences
    - Model size vs performance
    """

    def __init__(
        self,
        quality_evaluator: Optional[BaseEvaluator] = None,
        cost_per_token: Optional[float] = None,
        target_cost_per_quality: Optional[float] = None,
        include_latency_cost: bool = True,
        latency_cost_factor: float = 0.1,
        name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize cost efficiency evaluator.

        Args:
            quality_evaluator: Evaluator to measure response quality
            cost_per_token: Fixed cost per token (if not provided dynamically)
            target_cost_per_quality: Target cost-to-quality ratio
            include_latency_cost: Whether to include latency in cost calculation
            latency_cost_factor: Weight for latency in cost calculation
            name: Custom name for the evaluator
            **kwargs: Additional parameters passed to BaseEvaluator
        """
        super().__init__(
            name=name or "cost_efficiency",
            description="Cost efficiency evaluator for AI responses",
            **kwargs
        )

        self.quality_evaluator = quality_evaluator
        self.cost_per_token = cost_per_token
        self.target_cost_per_quality = target_cost_per_quality or 0.001  # $0.001 per quality point
        self.include_latency_cost = include_latency_cost
        self.latency_cost_factor = latency_cost_factor

        # Default cost rates for common providers (per 1K tokens)
        self.default_costs = {
            "openai": {
                "gpt-4": {"input": 0.03, "output": 0.06},
                "gpt-4-turbo": {"input": 0.01, "output": 0.03},
                "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            },
            "anthropic": {
                "claude-3-opus": {"input": 0.015, "output": 0.075},
                "claude-3-sonnet": {"input": 0.003, "output": 0.015},
                "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            },
            "google": {
                "gemini-pro": {"input": 0.0005, "output": 0.0015},
                "gemini-ultra": {"input": 0.001, "output": 0.002},
            }
        }

    def evaluate(
        self,
        prediction: Any,
        reference: Any = None,
        input_data: Any = None,
        cost_usd: Optional[float] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        latency_ms: Optional[float] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate cost efficiency of AI response.

        Args:
            prediction: The model's response
            reference: Expected response (optional)
            input_data: The input query/prompt (optional)
            cost_usd: Actual cost in USD
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            total_tokens: Total tokens (input + output)
            latency_ms: Response latency in milliseconds
            model: Model name used
            provider: Provider name
            **kwargs: Additional evaluation parameters

        Returns:
            EvaluationResult with cost efficiency score (0.0-1.0)
        """
        try:
            # Calculate actual cost
            actual_cost = self._calculate_cost(
                cost_usd=cost_usd,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                model=model,
                provider=provider
            )

            # Evaluate quality if evaluator is provided
            quality_score = 1.0
            if self.quality_evaluator:
                quality_result = self.quality_evaluator.evaluate(
                    prediction=prediction,
                    reference=reference,
                    input_data=input_data,
                    **kwargs
                )
                quality_score = quality_result.score

            # Calculate efficiency score
            efficiency_score = self._calculate_efficiency(actual_cost, quality_score)

            # Prepare metadata
            metadata = {
                "actual_cost_usd": actual_cost,
                "quality_score": quality_score,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "latency_ms": latency_ms,
                "model": model,
                "provider": provider,
                "target_cost_per_quality": self.target_cost_per_quality,
                "include_latency_cost": self.include_latency_cost
            }

            # Add cost breakdown
            if input_tokens and output_tokens:
                metadata["cost_per_input_token"] = actual_cost / (input_tokens + output_tokens) if (input_tokens + output_tokens) > 0 else 0
                metadata["cost_per_output_token"] = actual_cost / output_tokens if output_tokens > 0 else 0

            # Add efficiency metrics
            if quality_score > 0:
                metadata["cost_per_quality_point"] = actual_cost / quality_score
                metadata["quality_per_dollar"] = quality_score / actual_cost if actual_cost > 0 else float('inf')

            comment = self._generate_comment(efficiency_score, actual_cost, quality_score)

            return EvaluationResult(
                key=self.name,
                score=efficiency_score,
                value=efficiency_score,
                comment=comment,
                metadata=metadata,
                cost_impact=actual_cost,
                latency_impact=latency_ms,
                quality_score=quality_score
            )

        except Exception as e:
            logger.error(f"Cost efficiency evaluation failed: {e}")
            return EvaluationResult(
                key=self.name,
                score=0.0,
                comment=f"Evaluation error: {str(e)}",
                metadata={"error": str(e)}
            )

    def _calculate_cost(
        self,
        cost_usd: Optional[float] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        latency_ms: Optional[float] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None
    ) -> float:
        """Calculate the actual cost of the AI response."""

        # Use provided cost if available
        if cost_usd is not None:
            base_cost = cost_usd
        elif self.cost_per_token and total_tokens:
            base_cost = self.cost_per_token * total_tokens
        elif input_tokens is not None and output_tokens is not None:
            # Estimate cost based on provider and model
            base_cost = self._estimate_cost_from_tokens(
                input_tokens, output_tokens, model, provider
            )
        else:
            # Fallback to default minimal cost
            base_cost = 0.001  # $0.001 default

        # Add latency cost if enabled
        if self.include_latency_cost and latency_ms:
            # Convert latency to cost (higher latency = higher cost)
            # Assume baseline of 1 second = $0.0001 additional cost
            latency_seconds = latency_ms / 1000.0
            latency_cost = latency_seconds * 0.0001 * self.latency_cost_factor
            base_cost += latency_cost

        return base_cost

    def _estimate_cost_from_tokens(
        self,
        input_tokens: int,
        output_tokens: int,
        model: Optional[str] = None,
        provider: Optional[str] = None
    ) -> float:
        """Estimate cost based on token counts and model info."""

        # Default fallback rates
        input_rate = 0.001  # $0.001 per 1K input tokens
        output_rate = 0.002  # $0.002 per 1K output tokens

        # Try to get specific rates
        if provider and model:
            provider_costs = self.default_costs.get(provider.lower(), {})
            model_costs = provider_costs.get(model.lower(), {})

            if model_costs:
                input_rate = model_costs.get("input", input_rate)
                output_rate = model_costs.get("output", output_rate)

        # Calculate cost (rates are per 1K tokens)
        input_cost = (input_tokens / 1000.0) * input_rate
        output_cost = (output_tokens / 1000.0) * output_rate

        return input_cost + output_cost

    def _calculate_efficiency(self, cost: float, quality: float) -> float:
        """Calculate efficiency score based on cost and quality."""

        if cost <= 0:
            return 1.0 if quality > 0 else 0.0

        if quality <= 0:
            return 0.0

        # Calculate cost per quality point
        cost_per_quality = cost / quality

        # Compare against target
        if cost_per_quality <= self.target_cost_per_quality:
            # Excellent efficiency
            efficiency = 1.0
        else:
            # Scaled efficiency based on how much over target
            ratio = self.target_cost_per_quality / cost_per_quality
            efficiency = min(1.0, ratio)

        # Bonus for high quality at any cost
        if quality >= 0.9:
            efficiency = min(1.0, efficiency * 1.1)

        return efficiency

    def _generate_comment(self, efficiency: float, cost: float, quality: float) -> str:
        """Generate human-readable comment about cost efficiency."""
        cost_per_quality = cost / quality if quality > 0 else float('inf')

        if efficiency >= 0.9:
            return f"Excellent efficiency: ${cost:.4f} for {quality:.2f} quality"
        elif efficiency >= 0.7:
            return f"Good efficiency: ${cost:.4f} for {quality:.2f} quality"
        elif efficiency >= 0.5:
            return f"Moderate efficiency: ${cost:.4f} for {quality:.2f} quality"
        elif efficiency >= 0.3:
            return f"Poor efficiency: ${cost:.4f} for {quality:.2f} quality"
        else:
            return f"Very poor efficiency: ${cost:.4f} for {quality:.2f} quality"


class BudgetConstraintEvaluator(CostEfficiencyEvaluator):
    """
    Specialized cost efficiency evaluator with budget constraints.

    Evaluates whether responses stay within specified budget limits
    while maintaining acceptable quality thresholds.
    """

    def __init__(
        self,
        max_cost_per_request: float,
        min_quality_threshold: float = 0.7,
        budget_penalty_factor: float = 2.0,
        name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize budget constraint evaluator.

        Args:
            max_cost_per_request: Maximum allowed cost per request
            min_quality_threshold: Minimum acceptable quality score
            budget_penalty_factor: Penalty factor for exceeding budget
            name: Custom name for the evaluator
            **kwargs: Additional parameters passed to CostEfficiencyEvaluator
        """
        super().__init__(
            name=name or "cost_efficiency_budget",
            **kwargs
        )

        self.max_cost_per_request = max_cost_per_request
        self.min_quality_threshold = min_quality_threshold
        self.budget_penalty_factor = budget_penalty_factor

    def evaluate(self, **kwargs) -> EvaluationResult:
        """Evaluate cost efficiency with budget constraints."""

        # Get base efficiency evaluation
        base_result = super().evaluate(**kwargs)

        # Apply budget constraints
        actual_cost = base_result.metadata.get("actual_cost_usd", 0)
        quality_score = base_result.metadata.get("quality_score", 0)

        # Check budget violation
        budget_violation = max(0, actual_cost - self.max_cost_per_request)
        budget_compliance = budget_violation == 0

        # Check quality threshold
        quality_compliance = quality_score >= self.min_quality_threshold

        # Calculate final score with penalties
        final_score = base_result.score

        if not budget_compliance:
            # Apply budget penalty
            penalty = min(0.8, budget_violation * self.budget_penalty_factor)
            final_score *= (1.0 - penalty)

        if not quality_compliance:
            # Apply quality penalty
            quality_penalty = (self.min_quality_threshold - quality_score) * 0.5
            final_score *= (1.0 - quality_penalty)

        # Update metadata
        metadata = base_result.metadata.copy()
        metadata.update({
            "max_cost_per_request": self.max_cost_per_request,
            "min_quality_threshold": self.min_quality_threshold,
            "budget_violation": budget_violation,
            "budget_compliance": budget_compliance,
            "quality_compliance": quality_compliance,
            "budget_penalty_factor": self.budget_penalty_factor,
            "base_efficiency_score": base_result.score
        })

        # Generate comment
        comment_parts = []
        if budget_compliance and quality_compliance:
            comment_parts.append("Within budget and quality constraints")
        else:
            if not budget_compliance:
                comment_parts.append(f"Budget exceeded by ${budget_violation:.4f}")
            if not quality_compliance:
                comment_parts.append(f"Quality below threshold ({quality_score:.2f} < {self.min_quality_threshold})")

        comment = "; ".join(comment_parts) if comment_parts else base_result.comment

        return EvaluationResult(
            key=self.name,
            score=final_score,
            value=final_score,
            comment=comment,
            metadata=metadata,
            cost_impact=actual_cost,
            latency_impact=base_result.latency_impact,
            quality_score=quality_score
        )


class ROIEvaluator(CostEfficiencyEvaluator):
    """
    Return on Investment evaluator for AI responses.

    Evaluates the business value generated relative to cost,
    considering factors like user satisfaction and task completion.
    """

    def __init__(
        self,
        value_per_quality_point: float = 1.0,
        time_value_factor: float = 0.01,
        user_satisfaction_weight: float = 0.3,
        task_completion_weight: float = 0.4,
        quality_weight: float = 0.3,
        name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize ROI evaluator.

        Args:
            value_per_quality_point: Business value per quality point
            time_value_factor: Value factor for time savings
            user_satisfaction_weight: Weight for user satisfaction in ROI
            task_completion_weight: Weight for task completion in ROI
            quality_weight: Weight for quality in ROI
            name: Custom name for the evaluator
            **kwargs: Additional parameters passed to CostEfficiencyEvaluator
        """
        super().__init__(
            name=name or "roi_evaluator",
            **kwargs
        )

        self.value_per_quality_point = value_per_quality_point
        self.time_value_factor = time_value_factor
        self.user_satisfaction_weight = user_satisfaction_weight
        self.task_completion_weight = task_completion_weight
        self.quality_weight = quality_weight

    def evaluate(
        self,
        user_satisfaction: Optional[float] = None,
        task_completed: Optional[bool] = None,
        time_saved_seconds: Optional[float] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate ROI of AI response.

        Args:
            user_satisfaction: User satisfaction score (0.0-1.0)
            task_completed: Whether the task was completed successfully
            time_saved_seconds: Time saved compared to manual completion
            **kwargs: Additional parameters passed to base evaluator

        Returns:
            EvaluationResult with ROI score
        """

        # Get base cost efficiency evaluation
        base_result = super().evaluate(**kwargs)

        # Calculate business value
        quality_score = base_result.metadata.get("quality_score", 0)
        actual_cost = base_result.metadata.get("actual_cost_usd", 0)

        # Calculate value components
        quality_value = quality_score * self.value_per_quality_point

        satisfaction_value = (user_satisfaction or 0.5) * self.user_satisfaction_weight

        completion_value = (1.0 if task_completed else 0.0) * self.task_completion_weight

        time_value = 0.0
        if time_saved_seconds:
            time_value = time_saved_seconds * self.time_value_factor

        # Total business value
        total_value = (
            quality_value * self.quality_weight +
            satisfaction_value +
            completion_value +
            time_value
        )

        # Calculate ROI
        if actual_cost > 0:
            roi = (total_value - actual_cost) / actual_cost
        else:
            roi = total_value  # Infinite ROI if no cost

        # Normalize ROI to 0-1 scale for scoring
        # ROI of 100% (1.0) maps to score of 1.0
        roi_score = min(1.0, max(0.0, (roi + 1.0) / 2.0))

        # Update metadata
        metadata = base_result.metadata.copy()
        metadata.update({
            "quality_value": quality_value,
            "satisfaction_value": satisfaction_value,
            "completion_value": completion_value,
            "time_value": time_value,
            "total_business_value": total_value,
            "roi": roi,
            "roi_percentage": roi * 100,
            "user_satisfaction": user_satisfaction,
            "task_completed": task_completed,
            "time_saved_seconds": time_saved_seconds
        })

        comment = f"ROI: {roi*100:.1f}% (Value: ${total_value:.4f}, Cost: ${actual_cost:.4f})"

        return EvaluationResult(
            key=self.name,
            score=roi_score,
            value=roi,
            comment=comment,
            metadata=metadata,
            cost_impact=actual_cost,
            quality_score=quality_score
        )