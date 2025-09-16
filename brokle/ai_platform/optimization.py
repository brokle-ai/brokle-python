"""
Cost optimization configuration and budget management.

This module provides client-side cost configuration and budget enforcement
that integrates with Brokle's backend optimization intelligence. The SDK
handles local budget limits and usage tracking, while complex optimization
decisions are handled by the backend.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Cost optimization strategies."""
    AGGRESSIVE = "aggressive"     # Maximum cost savings
    BALANCED = "balanced"         # Balance cost and quality
    CONSERVATIVE = "conservative" # Minimal quality impact
    CUSTOM = "custom"            # User-defined rules


class BudgetPeriod(Enum):
    """Budget period types."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"
    CUSTOM = "custom"


class CostUnit(Enum):
    """Cost units for tracking."""
    USD = "usd"
    TOKENS = "tokens"
    REQUESTS = "requests"
    CUSTOM = "custom"


class BudgetAlert(Enum):
    """Budget alert levels."""
    WARNING = "warning"    # 80% of budget
    CRITICAL = "critical"  # 95% of budget
    EXCEEDED = "exceeded"  # 100% of budget


@dataclass
class CostBreakdown:
    """Detailed cost breakdown."""
    total_cost: float
    provider_costs: Dict[str, float] = field(default_factory=dict)
    model_costs: Dict[str, float] = field(default_factory=dict)

    # Token breakdown
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Request breakdown
    total_requests: int = 0
    cached_requests: int = 0
    direct_requests: int = 0

    # Savings
    cache_savings: float = 0.0
    routing_savings: float = 0.0
    total_savings: float = 0.0

    # Metadata
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    currency: str = "USD"


@dataclass
class BudgetConfig:
    """Budget configuration and limits."""
    enabled: bool = True

    # Budget limits
    max_cost: float = 100.0
    period: BudgetPeriod = BudgetPeriod.MONTHLY
    currency: str = "USD"

    # Alert thresholds
    warning_threshold: float = 0.8   # 80%
    critical_threshold: float = 0.95 # 95%

    # Enforcement
    enforce_hard_limit: bool = True
    block_on_exceeded: bool = True

    # Reset behavior
    auto_reset: bool = True
    rollover_unused: bool = False

    # Notifications
    email_alerts: bool = True
    webhook_alerts: bool = False
    alert_endpoints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "enabled": self.enabled,
            "max_cost": self.max_cost,
            "period": self.period.value,
            "currency": self.currency,
            "warning_threshold": self.warning_threshold,
            "critical_threshold": self.critical_threshold,
            "enforce_hard_limit": self.enforce_hard_limit,
            "block_on_exceeded": self.block_on_exceeded,
            "auto_reset": self.auto_reset,
            "rollover_unused": self.rollover_unused,
            "email_alerts": self.email_alerts,
            "webhook_alerts": self.webhook_alerts,
            "alert_endpoints": self.alert_endpoints
        }


@dataclass
class UsageConfig:
    """Usage tracking and limits configuration."""
    enabled: bool = True

    # Usage limits
    max_requests_per_minute: Optional[int] = None
    max_requests_per_hour: Optional[int] = None
    max_requests_per_day: Optional[int] = None

    # Token limits
    max_tokens_per_request: Optional[int] = None
    max_tokens_per_day: Optional[int] = None

    # Provider limits
    provider_limits: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Model limits
    model_limits: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Tracking granularity
    track_by_user: bool = True
    track_by_project: bool = True
    track_by_environment: bool = True

    def add_provider_limit(
        self,
        provider: str,
        max_cost: Optional[float] = None,
        max_requests: Optional[int] = None,
        max_tokens: Optional[int] = None
    ) -> None:
        """Add provider-specific limits."""
        self.provider_limits[provider] = {
            "max_cost": max_cost,
            "max_requests": max_requests,
            "max_tokens": max_tokens
        }

    def add_model_limit(
        self,
        model: str,
        max_cost: Optional[float] = None,
        max_requests: Optional[int] = None,
        max_tokens: Optional[int] = None
    ) -> None:
        """Add model-specific limits."""
        self.model_limits[model] = {
            "max_cost": max_cost,
            "max_requests": max_requests,
            "max_tokens": max_tokens
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "enabled": self.enabled,
            "max_requests_per_minute": self.max_requests_per_minute,
            "max_requests_per_hour": self.max_requests_per_hour,
            "max_requests_per_day": self.max_requests_per_day,
            "max_tokens_per_request": self.max_tokens_per_request,
            "max_tokens_per_day": self.max_tokens_per_day,
            "provider_limits": self.provider_limits,
            "model_limits": self.model_limits,
            "track_by_user": self.track_by_user,
            "track_by_project": self.track_by_project,
            "track_by_environment": self.track_by_environment
        }


@dataclass
class CostOptimizationConfig:
    """Complete cost optimization configuration."""
    enabled: bool = True
    strategy: OptimizationStrategy = OptimizationStrategy.BALANCED

    # Budget management
    budget: BudgetConfig = field(default_factory=BudgetConfig)

    # Usage limits
    usage: UsageConfig = field(default_factory=UsageConfig)

    # Optimization settings
    enable_smart_routing: bool = True
    enable_caching: bool = True
    enable_model_fallback: bool = True

    # Cost preferences
    max_cost_per_request: Optional[float] = None
    preferred_providers: List[str] = field(default_factory=list)
    blocked_providers: List[str] = field(default_factory=list)

    # Quality trade-offs
    allow_quality_degradation: bool = False
    min_quality_score: float = 0.7

    # Advanced optimization
    enable_batch_processing: bool = False
    batch_size: int = 10
    batch_timeout_seconds: float = 5.0

    # Analytics
    track_cost_savings: bool = True
    track_provider_performance: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "enabled": self.enabled,
            "strategy": self.strategy.value,
            "budget": self.budget.to_dict(),
            "usage": self.usage.to_dict(),
            "enable_smart_routing": self.enable_smart_routing,
            "enable_caching": self.enable_caching,
            "enable_model_fallback": self.enable_model_fallback,
            "max_cost_per_request": self.max_cost_per_request,
            "preferred_providers": self.preferred_providers,
            "blocked_providers": self.blocked_providers,
            "allow_quality_degradation": self.allow_quality_degradation,
            "min_quality_score": self.min_quality_score,
            "enable_batch_processing": self.enable_batch_processing,
            "batch_size": self.batch_size,
            "batch_timeout_seconds": self.batch_timeout_seconds,
            "track_cost_savings": self.track_cost_savings,
            "track_provider_performance": self.track_provider_performance
        }


class CostTracker:
    """
    Local cost tracking with budget enforcement.

    Provides real-time cost monitoring and budget enforcement
    while integrating with backend optimization intelligence.
    """

    def __init__(self):
        self.config = CostOptimizationConfig()
        self._current_costs: Dict[str, float] = {}
        self._current_usage: Dict[str, int] = {}
        self._cost_history: List[Dict[str, Any]] = []
        self._alerts_sent: Dict[str, datetime] = {}

    def configure(self, config: CostOptimizationConfig) -> None:
        """Configure cost optimization settings."""
        self.config = config
        logger.info(f"Cost optimization configured with strategy: {config.strategy.value}")

    def get_config(self) -> CostOptimizationConfig:
        """Get current cost optimization configuration."""
        return self.config

    def track_request_cost(
        self,
        cost: float,
        provider: str,
        model: str,
        tokens: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Track cost for a single request."""
        if not self.config.enabled:
            return {"allowed": True, "cost": cost}

        # Update current costs
        period_key = self._get_current_period_key()
        if period_key not in self._current_costs:
            self._current_costs[period_key] = 0.0

        self._current_costs[period_key] += cost

        # Update usage counters
        self._update_usage_counters(provider, model, tokens)

        # Check budget limits
        budget_status = self._check_budget_limits(cost)

        # Check usage limits
        usage_status = self._check_usage_limits(provider, model, tokens)

        # Record cost entry
        self._record_cost_entry(cost, provider, model, tokens, metadata)

        return {
            "allowed": budget_status["allowed"] and usage_status["allowed"],
            "cost": cost,
            "budget_status": budget_status,
            "usage_status": usage_status,
            "total_cost_this_period": self._current_costs[period_key]
        }

    def _get_current_period_key(self) -> str:
        """Get current budget period key."""
        now = datetime.utcnow()

        if self.config.budget.period == BudgetPeriod.DAILY:
            return now.strftime("%Y-%m-%d")
        elif self.config.budget.period == BudgetPeriod.WEEKLY:
            # ISO week
            year, week, _ = now.isocalendar()
            return f"{year}-W{week:02d}"
        elif self.config.budget.period == BudgetPeriod.MONTHLY:
            return now.strftime("%Y-%m")
        elif self.config.budget.period == BudgetPeriod.YEARLY:
            return now.strftime("%Y")
        else:
            # Default to daily for custom periods
            return now.strftime("%Y-%m-%d")

    def _update_usage_counters(self, provider: str, model: str, tokens: int) -> None:
        """Update usage counters."""
        period_key = self._get_current_period_key()

        # Initialize counters if needed
        if period_key not in self._current_usage:
            self._current_usage[period_key] = {
                "requests": 0,
                "tokens": 0,
                "providers": {},
                "models": {}
            }

        usage = self._current_usage[period_key]

        # Update request and token counters
        usage["requests"] += 1
        usage["tokens"] += tokens

        # Update provider counters
        if provider not in usage["providers"]:
            usage["providers"][provider] = {"requests": 0, "tokens": 0}
        usage["providers"][provider]["requests"] += 1
        usage["providers"][provider]["tokens"] += tokens

        # Update model counters
        if model not in usage["models"]:
            usage["models"][model] = {"requests": 0, "tokens": 0}
        usage["models"][model]["requests"] += 1
        usage["models"][model]["tokens"] += tokens

    def _check_budget_limits(self, cost: float) -> Dict[str, Any]:
        """Check if request would exceed budget limits."""
        if not self.config.budget.enabled:
            return {"allowed": True, "reason": "budget_disabled"}

        period_key = self._get_current_period_key()
        current_cost = self._current_costs.get(period_key, 0.0)
        projected_cost = current_cost + cost

        max_cost = self.config.budget.max_cost
        usage_ratio = projected_cost / max_cost if max_cost > 0 else 0

        # Check hard limit
        if self.config.budget.enforce_hard_limit and projected_cost > max_cost:
            self._send_budget_alert(BudgetAlert.EXCEEDED, usage_ratio, projected_cost)
            return {
                "allowed": False,
                "reason": "budget_exceeded",
                "current_cost": current_cost,
                "projected_cost": projected_cost,
                "max_cost": max_cost,
                "usage_ratio": usage_ratio
            }

        # Check alert thresholds
        if usage_ratio >= self.config.budget.critical_threshold:
            self._send_budget_alert(BudgetAlert.CRITICAL, usage_ratio, projected_cost)
        elif usage_ratio >= self.config.budget.warning_threshold:
            self._send_budget_alert(BudgetAlert.WARNING, usage_ratio, projected_cost)

        return {
            "allowed": True,
            "current_cost": current_cost,
            "projected_cost": projected_cost,
            "max_cost": max_cost,
            "usage_ratio": usage_ratio
        }

    def _check_usage_limits(self, provider: str, model: str, tokens: int) -> Dict[str, Any]:
        """Check if request would exceed usage limits."""
        if not self.config.usage.enabled:
            return {"allowed": True, "reason": "usage_limits_disabled"}

        period_key = self._get_current_period_key()
        usage = self._current_usage.get(period_key, {})

        # Check provider limits
        if provider in self.config.usage.provider_limits:
            provider_limits = self.config.usage.provider_limits[provider]
            provider_usage = usage.get("providers", {}).get(provider, {"requests": 0, "tokens": 0})

            if provider_limits.get("max_requests") and provider_usage["requests"] >= provider_limits["max_requests"]:
                return {"allowed": False, "reason": f"provider_{provider}_request_limit"}

            if provider_limits.get("max_tokens") and provider_usage["tokens"] + tokens > provider_limits["max_tokens"]:
                return {"allowed": False, "reason": f"provider_{provider}_token_limit"}

        # Check model limits
        if model in self.config.usage.model_limits:
            model_limits = self.config.usage.model_limits[model]
            model_usage = usage.get("models", {}).get(model, {"requests": 0, "tokens": 0})

            if model_limits.get("max_requests") and model_usage["requests"] >= model_limits["max_requests"]:
                return {"allowed": False, "reason": f"model_{model}_request_limit"}

            if model_limits.get("max_tokens") and model_usage["tokens"] + tokens > model_limits["max_tokens"]:
                return {"allowed": False, "reason": f"model_{model}_token_limit"}

        return {"allowed": True}

    def _send_budget_alert(self, alert_type: BudgetAlert, usage_ratio: float, current_cost: float) -> None:
        """Send budget alert if not recently sent."""
        alert_key = f"{alert_type.value}_{self._get_current_period_key()}"

        # Check if alert was recently sent (avoid spam)
        if alert_key in self._alerts_sent:
            last_sent = self._alerts_sent[alert_key]
            if datetime.utcnow() - last_sent < timedelta(hours=1):
                return

        # Record alert
        self._alerts_sent[alert_key] = datetime.utcnow()

        # Log alert
        logger.warning(
            f"Budget alert: {alert_type.value} - "
            f"Usage: {usage_ratio:.1%}, Cost: ${current_cost:.2f}, "
            f"Limit: ${self.config.budget.max_cost:.2f}"
        )

        # Send notifications (placeholder for actual implementation)
        if self.config.budget.email_alerts:
            logger.info(f"Would send email alert for {alert_type.value}")

        if self.config.budget.webhook_alerts and self.config.budget.alert_endpoints:
            logger.info(f"Would send webhook alert to {len(self.config.budget.alert_endpoints)} endpoints")

    def _record_cost_entry(
        self,
        cost: float,
        provider: str,
        model: str,
        tokens: int,
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Record cost entry in history."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "cost": cost,
            "provider": provider,
            "model": model,
            "tokens": tokens,
            "metadata": metadata or {}
        }

        self._cost_history.append(entry)

        # Keep history size manageable
        if len(self._cost_history) > 10000:
            self._cost_history = self._cost_history[-5000:]  # Keep last 5000 entries

    def get_cost_breakdown(
        self,
        period: Optional[str] = None
    ) -> CostBreakdown:
        """Get detailed cost breakdown for period."""
        target_period = period or self._get_current_period_key()

        # Calculate costs from history
        period_entries = [
            entry for entry in self._cost_history
            if self._get_period_key_for_timestamp(entry["timestamp"]) == target_period
        ]

        total_cost = sum(entry["cost"] for entry in period_entries)
        total_tokens = sum(entry["tokens"] for entry in period_entries)
        total_requests = len(period_entries)

        # Provider breakdown
        provider_costs = {}
        for entry in period_entries:
            provider = entry["provider"]
            provider_costs[provider] = provider_costs.get(provider, 0) + entry["cost"]

        # Model breakdown
        model_costs = {}
        for entry in period_entries:
            model = entry["model"]
            model_costs[model] = model_costs.get(model, 0) + entry["cost"]

        return CostBreakdown(
            total_cost=total_cost,
            provider_costs=provider_costs,
            model_costs=model_costs,
            total_tokens=total_tokens,
            total_requests=total_requests
        )

    def _get_period_key_for_timestamp(self, timestamp_iso: str) -> str:
        """Get period key for a timestamp."""
        timestamp = datetime.fromisoformat(timestamp_iso.replace('Z', '+00:00'))

        if self.config.budget.period == BudgetPeriod.DAILY:
            return timestamp.strftime("%Y-%m-%d")
        elif self.config.budget.period == BudgetPeriod.WEEKLY:
            year, week, _ = timestamp.isocalendar()
            return f"{year}-W{week:02d}"
        elif self.config.budget.period == BudgetPeriod.MONTHLY:
            return timestamp.strftime("%Y-%m")
        elif self.config.budget.period == BudgetPeriod.YEARLY:
            return timestamp.strftime("%Y")
        else:
            return timestamp.strftime("%Y-%m-%d")

    def create_aggressive_config(self) -> CostOptimizationConfig:
        """Create aggressive cost optimization configuration template."""
        config = CostOptimizationConfig(strategy=OptimizationStrategy.AGGRESSIVE)

        # Basic budget configuration - user should customize
        config.budget.max_cost = 100.0  # Default, user should set appropriate value
        config.budget.warning_threshold = 0.7
        config.budget.critical_threshold = 0.85

        # Enable all optimization features - backend will handle specifics
        config.enable_smart_routing = True
        config.enable_caching = True
        config.enable_model_fallback = True

        return config

    def create_balanced_config(self) -> CostOptimizationConfig:
        """Create balanced cost optimization configuration template."""
        config = CostOptimizationConfig(strategy=OptimizationStrategy.BALANCED)

        # Moderate budget configuration
        config.budget.max_cost = 200.0  # Default, user should set appropriate value
        config.budget.warning_threshold = 0.8
        config.budget.critical_threshold = 0.95

        # Balanced feature enablement
        config.enable_smart_routing = True
        config.enable_caching = True
        config.enable_model_fallback = True

        return config

    def create_conservative_config(self) -> CostOptimizationConfig:
        """Create conservative cost optimization configuration template."""
        config = CostOptimizationConfig(strategy=OptimizationStrategy.CONSERVATIVE)

        # Higher budget allowance
        config.budget.max_cost = 500.0  # Default, user should set appropriate value
        config.budget.warning_threshold = 0.85
        config.budget.critical_threshold = 0.95

        # Conservative feature set - let backend handle optimization details
        config.enable_smart_routing = True
        config.enable_caching = True
        config.enable_model_fallback = False  # More conservative

        return config

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get cost optimization statistics."""
        current_period = self._get_current_period_key()
        current_cost = self._current_costs.get(current_period, 0.0)
        current_usage = self._current_usage.get(current_period, {})

        return {
            "config": self.config.to_dict(),
            "current_period": current_period,
            "current_cost": current_cost,
            "budget_usage_ratio": current_cost / self.config.budget.max_cost if self.config.budget.max_cost > 0 else 0,
            "current_requests": current_usage.get("requests", 0),
            "current_tokens": current_usage.get("tokens", 0),
            "cost_history_size": len(self._cost_history),
            "alerts_sent": len(self._alerts_sent)
        }


# Global cost tracker instance
_cost_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Get global cost tracker instance."""
    global _cost_tracker

    if _cost_tracker is None:
        _cost_tracker = CostTracker()

    return _cost_tracker


def configure_optimization(config: CostOptimizationConfig) -> None:
    """Configure global cost optimization settings."""
    tracker = get_cost_tracker()
    tracker.configure(config)


def get_optimization_config() -> CostOptimizationConfig:
    """Get current cost optimization configuration."""
    tracker = get_cost_tracker()
    return tracker.get_config()


def track_request_cost(
    cost: float,
    provider: str,
    model: str,
    tokens: int = 0,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Track cost for a single request."""
    tracker = get_cost_tracker()
    return tracker.track_request_cost(cost, provider, model, tokens, metadata)


def get_cost_breakdown(period: Optional[str] = None) -> CostBreakdown:
    """Get detailed cost breakdown for period."""
    tracker = get_cost_tracker()
    return tracker.get_cost_breakdown(period)


def create_aggressive_optimization() -> CostOptimizationConfig:
    """Create aggressive cost optimization configuration."""
    tracker = get_cost_tracker()
    return tracker.create_aggressive_config()


def create_balanced_optimization() -> CostOptimizationConfig:
    """Create balanced cost optimization configuration."""
    tracker = get_cost_tracker()
    return tracker.create_balanced_config()


def create_conservative_optimization() -> CostOptimizationConfig:
    """Create conservative cost optimization configuration."""
    tracker = get_cost_tracker()
    return tracker.create_conservative_config()


def get_optimization_stats() -> Dict[str, Any]:
    """Get cost optimization statistics."""
    tracker = get_cost_tracker()
    return tracker.get_optimization_stats()