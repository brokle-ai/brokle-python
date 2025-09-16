"""
Provider health configuration and basic monitoring.

This module provides client-side provider configuration and basic health
monitoring that integrates with Brokle's backend provider intelligence.
Complex health scoring and ranking is handled by the backend.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ProviderStatus(Enum):
    """Provider status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class HealthCheckType(Enum):
    """Types of health checks."""
    PING = "ping"               # Simple connectivity check
    AUTH = "auth"               # Authentication validation
    QUOTA = "quota"             # Rate limit and quota check
    LATENCY = "latency"         # Performance check
    QUALITY = "quality"         # Response quality check
    COST = "cost"               # Cost efficiency check
    FULL = "full"               # Comprehensive check


@dataclass
class HealthMetric:
    """Individual health metric measurement."""
    name: str
    value: float
    unit: str
    threshold: Optional[float] = None
    status: ProviderStatus = ProviderStatus.UNKNOWN

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now())
    source: str = "local"
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderHealth:
    """Complete provider health assessment."""
    provider: str
    status: ProviderStatus
    overall_score: float  # 0.0 - 1.0

    # Individual metrics
    metrics: Dict[str, HealthMetric] = field(default_factory=dict)

    # Performance data
    latency_ms: Optional[float] = None
    success_rate: Optional[float] = None
    error_rate: Optional[float] = None

    # Availability data
    uptime_percentage: Optional[float] = None
    last_successful_request: Optional[datetime] = None
    last_failed_request: Optional[datetime] = None

    # Cost data
    average_cost_per_request: Optional[float] = None
    cost_efficiency_score: Optional[float] = None

    # Quality data
    average_quality_score: Optional[float] = None
    quality_consistency: Optional[float] = None

    # Metadata
    last_check: datetime = field(default_factory=lambda: datetime.now())
    next_check: Optional[datetime] = None
    check_interval_seconds: int = 300  # 5 minutes

    # History
    status_history: List[Dict[str, Any]] = field(default_factory=list)

    def add_metric(self, metric: HealthMetric) -> None:
        """Add health metric."""
        self.metrics[metric.name] = metric

    def get_metric_value(self, name: str) -> Optional[float]:
        """Get metric value by name."""
        metric = self.metrics.get(name)
        return metric.value if metric else None

    def is_healthy(self) -> bool:
        """Check if provider is healthy."""
        return self.status in [ProviderStatus.HEALTHY, ProviderStatus.DEGRADED]

    def needs_check(self) -> bool:
        """Check if health check is needed."""
        if not self.next_check:
            return True
        return datetime.now() >= self.next_check

    def update_next_check(self) -> None:
        """Update next check time."""
        self.next_check = datetime.now() + timedelta(seconds=self.check_interval_seconds)


@dataclass
class ProviderMetrics:
    """Comprehensive provider metrics."""
    provider: str

    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # Performance metrics
    total_latency_ms: float = 0.0
    min_latency_ms: Optional[float] = None
    max_latency_ms: Optional[float] = None

    # Cost metrics
    total_cost: float = 0.0
    total_tokens: int = 0

    # Quality metrics
    total_quality_score: float = 0.0
    quality_evaluations: int = 0

    # Time period
    period_start: datetime = field(default_factory=lambda: datetime.now())
    period_end: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    @property
    def average_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests

    @property
    def average_cost_per_request(self) -> float:
        """Calculate average cost per request."""
        if self.total_requests == 0:
            return 0.0
        return self.total_cost / self.total_requests

    @property
    def average_quality_score(self) -> float:
        """Calculate average quality score."""
        if self.quality_evaluations == 0:
            return 0.0
        return self.total_quality_score / self.quality_evaluations

    def record_request(
        self,
        success: bool,
        latency_ms: float,
        cost: float = 0.0,
        tokens: int = 0,
        quality_score: Optional[float] = None
    ) -> None:
        """Record request metrics."""
        self.total_requests += 1

        if success:
            self.successful_requests += 1
            self.total_latency_ms += latency_ms

            # Update min/max latency
            if self.min_latency_ms is None or latency_ms < self.min_latency_ms:
                self.min_latency_ms = latency_ms
            if self.max_latency_ms is None or latency_ms > self.max_latency_ms:
                self.max_latency_ms = latency_ms
        else:
            self.failed_requests += 1

        # Update cost and token metrics
        self.total_cost += cost
        self.total_tokens += tokens

        # Update quality metrics
        if quality_score is not None:
            self.total_quality_score += quality_score
            self.quality_evaluations += 1


class ProviderMonitor:
    """
    Client-side provider monitoring and configuration.

    Tracks basic provider metrics and coordinates with Brokle's backend
    provider intelligence service for health scoring and ranking.
    """

    def __init__(self):
        self._provider_health: Dict[str, ProviderHealth] = {}
        self._provider_metrics: Dict[str, ProviderMetrics] = {}
        self._backend_available = True

        # Health check configuration
        self._default_check_interval = 300  # 5 minutes
        self._health_check_timeout = 10.0   # 10 seconds

    def get_provider_health(self, provider: str) -> Optional[ProviderHealth]:
        """Get health status for a provider."""
        health = self._provider_health.get(provider)

        # Perform health check if needed
        if not health or health.needs_check():
            health = self._perform_health_check(provider)

        return health

    def get_provider_metrics(self, provider: str) -> Optional[ProviderMetrics]:
        """Get metrics for a provider."""
        return self._provider_metrics.get(provider)

    def record_provider_request(
        self,
        provider: str,
        success: bool,
        latency_ms: float,
        cost: float = 0.0,
        tokens: int = 0,
        quality_score: Optional[float] = None,
        error_details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record provider request for metrics."""
        # Initialize metrics if needed
        if provider not in self._provider_metrics:
            self._provider_metrics[provider] = ProviderMetrics(provider=provider)

        # Record request
        self._provider_metrics[provider].record_request(
            success, latency_ms, cost, tokens, quality_score
        )

        # Update health status based on request
        self._update_health_from_request(provider, success, latency_ms, error_details)

    def _perform_health_check(self, provider: str) -> ProviderHealth:
        """Perform comprehensive health check for provider."""
        start_time = datetime.now()

        # Get existing health or create new
        health = self._provider_health.get(provider) or ProviderHealth(
            provider=provider,
            status=ProviderStatus.UNKNOWN,
            overall_score=0.0
        )

        try:
            # Try backend health check first
            if self._backend_available:
                backend_health = self._get_backend_health(provider)
                if backend_health:
                    health = backend_health
                else:
                    self._backend_available = False

            # Fallback to simple health check
            if not self._backend_available:
                health = self._create_simple_health_fallback(provider, health)

        except Exception as e:
            logger.warning(f"Health check failed for {provider}: {e}")
            health.status = ProviderStatus.UNHEALTHY
            health.overall_score = 0.0

        # Update timing
        health.last_check = datetime.now()
        health.update_next_check()

        # Store updated health
        self._provider_health[provider] = health

        logger.debug(f"Health check for {provider}: {health.status.value} (score: {health.overall_score:.2f})")

        return health

    def _get_backend_health(self, provider: str) -> Optional[ProviderHealth]:
        """Get provider health from backend service."""
        try:
            # TODO: Implement actual backend API call
            # Example: GET /api/v1/providers/{provider}/health
            logger.debug(f"Would call backend API: GET /api/v1/providers/{provider}/health")

            # Return None to trigger fallback
            return None

        except Exception as e:
            logger.warning(f"Backend provider health check failed: {e}")
            return None

    def _create_simple_health_fallback(self, provider: str, existing_health: ProviderHealth) -> ProviderHealth:
        """Create simple health status when backend unavailable."""
        metrics = self._provider_metrics.get(provider)

        if not metrics or metrics.total_requests == 0:
            # No data available - assume unknown
            existing_health.status = ProviderStatus.UNKNOWN
            existing_health.overall_score = 0.0
            return existing_health

        # Very basic health assessment
        success_rate = metrics.success_rate

        if success_rate >= 0.95:
            status = ProviderStatus.HEALTHY
            score = 0.9
        elif success_rate >= 0.80:
            status = ProviderStatus.DEGRADED
            score = 0.7
        elif success_rate >= 0.50:
            status = ProviderStatus.UNHEALTHY
            score = 0.4
        else:
            status = ProviderStatus.OFFLINE
            score = 0.1

        # Update basic health info
        existing_health.status = status
        existing_health.overall_score = score
        existing_health.success_rate = success_rate
        existing_health.error_rate = metrics.error_rate
        existing_health.latency_ms = metrics.average_latency_ms
        existing_health.average_cost_per_request = metrics.average_cost_per_request

        return existing_health

    def _update_health_from_request(
        self,
        provider: str,
        success: bool,
        latency_ms: float,
        error_details: Optional[Dict[str, Any]]
    ) -> None:
        """Update health status based on request result."""
        health = self._provider_health.get(provider)
        if not health:
            return

        # Update last request timestamps
        now = datetime.now()
        if success:
            health.last_successful_request = now
        else:
            health.last_failed_request = now

        # Adjust health score based on recent performance
        if success and latency_ms < 2000:  # Good performance
            health.overall_score = min(1.0, health.overall_score + 0.01)
        elif not success:  # Failed request
            health.overall_score = max(0.0, health.overall_score - 0.05)

        # Update status based on new score
        health.status = self._score_to_status(health.overall_score)

    def get_all_provider_health(self) -> Dict[str, ProviderHealth]:
        """Get health status for all monitored providers."""
        # Refresh health for providers that need checks
        for provider in self._provider_health.keys():
            self.get_provider_health(provider)

        return self._provider_health.copy()

    def get_all_provider_metrics(self) -> Dict[str, ProviderMetrics]:
        """Get metrics for all monitored providers."""
        return self._provider_metrics.copy()

    def _score_to_status(self, score: float) -> ProviderStatus:
        """Convert health score to provider status."""
        if score >= 0.9:
            return ProviderStatus.HEALTHY
        elif score >= 0.7:
            return ProviderStatus.DEGRADED
        elif score >= 0.3:
            return ProviderStatus.UNHEALTHY
        else:
            return ProviderStatus.OFFLINE

    def get_healthy_providers(self) -> List[str]:
        """Get list of healthy providers."""
        healthy = []
        for provider, health in self._provider_health.items():
            if health.is_healthy():
                healthy.append(provider)
        return healthy

    def get_provider_rankings(self) -> List[Dict[str, Any]]:
        """Get providers ranked by health score."""
        rankings = []

        for provider, health in self._provider_health.items():
            metrics = self._provider_metrics.get(provider)

            rankings.append({
                "provider": provider,
                "health_score": health.overall_score,
                "status": health.status.value,
                "success_rate": metrics.success_rate if metrics else 0.0,
                "average_latency_ms": metrics.average_latency_ms if metrics else 0.0,
                "average_cost": metrics.average_cost_per_request if metrics else 0.0,
                "total_requests": metrics.total_requests if metrics else 0
            })

        # Sort by health score (descending)
        rankings.sort(key=lambda x: x["health_score"], reverse=True)

        return rankings

    def reset_provider_metrics(self, provider: Optional[str] = None) -> None:
        """Reset metrics for provider or all providers."""
        if provider:
            if provider in self._provider_metrics:
                del self._provider_metrics[provider]
        else:
            self._provider_metrics.clear()

        logger.info(f"Reset metrics for {provider or 'all providers'}")

    def get_monitor_stats(self) -> Dict[str, Any]:
        """Get monitoring system statistics."""
        total_providers = len(self._provider_health)
        healthy_providers = len(self.get_healthy_providers())

        return {
            "total_providers": total_providers,
            "healthy_providers": healthy_providers,
            "unhealthy_providers": total_providers - healthy_providers,
            "backend_available": self._backend_available,
            "default_check_interval": self._default_check_interval,
            "providers_with_metrics": len(self._provider_metrics)
        }


# Global provider monitor instance
_provider_monitor: Optional[ProviderMonitor] = None


def get_provider_monitor() -> ProviderMonitor:
    """Get global provider monitor instance."""
    global _provider_monitor

    if _provider_monitor is None:
        _provider_monitor = ProviderMonitor()

    return _provider_monitor


def get_provider_health(provider: str) -> Optional[ProviderHealth]:
    """Get health status for a provider."""
    monitor = get_provider_monitor()
    return monitor.get_provider_health(provider)


def get_provider_status(provider: str) -> ProviderStatus:
    """Get status for a provider."""
    health = get_provider_health(provider)
    return health.status if health else ProviderStatus.UNKNOWN


def record_provider_request(
    provider: str,
    success: bool,
    latency_ms: float,
    cost: float = 0.0,
    tokens: int = 0,
    quality_score: Optional[float] = None,
    error_details: Optional[Dict[str, Any]] = None
) -> None:
    """Record provider request for metrics and health tracking."""
    monitor = get_provider_monitor()
    monitor.record_provider_request(
        provider, success, latency_ms, cost, tokens, quality_score, error_details
    )


def get_all_provider_health() -> Dict[str, ProviderHealth]:
    """Get health status for all monitored providers."""
    monitor = get_provider_monitor()
    return monitor.get_all_provider_health()


def get_healthy_providers() -> List[str]:
    """Get list of healthy providers."""
    monitor = get_provider_monitor()
    return monitor.get_healthy_providers()


def get_provider_rankings() -> List[Dict[str, Any]]:
    """Get providers ranked by health score."""
    monitor = get_provider_monitor()
    return monitor.get_provider_rankings()


def reset_provider_metrics(provider: Optional[str] = None) -> None:
    """Reset metrics for provider or all providers."""
    monitor = get_provider_monitor()
    monitor.reset_provider_metrics(provider)