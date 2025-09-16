"""
AI routing configuration and preferences.

This module provides client-side routing configuration that integrates
with Brokle's backend routing intelligence. The SDK manages routing
preferences and fallback settings while routing decisions are made by the backend.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """AI routing strategies supported by Brokle platform."""
    COST_OPTIMIZED = "cost_optimized"
    QUALITY_OPTIMIZED = "quality_optimized"
    LATENCY_OPTIMIZED = "latency_optimized"
    BALANCED = "balanced"
    CUSTOM = "custom"


class ProviderTier(Enum):
    """Provider performance tiers."""
    PREMIUM = "premium"      # GPT-4, Claude-3-Opus
    STANDARD = "standard"    # GPT-3.5, Claude-3-Sonnet
    ECONOMY = "economy"      # Claude-3-Haiku, smaller models


@dataclass
class ProviderConfig:
    """Configuration for individual AI providers."""
    name: str
    model: str
    tier: ProviderTier = ProviderTier.STANDARD

    # Cost constraints
    max_cost_per_request: Optional[float] = None
    max_cost_per_day: Optional[float] = None

    # Quality constraints
    min_quality_score: Optional[float] = None

    # Performance constraints
    max_latency_ms: Optional[float] = None

    # Availability
    enabled: bool = True
    weight: float = 1.0  # For load balancing

    # Metadata
    region: Optional[str] = None
    api_key_env: Optional[str] = None
    custom_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "name": self.name,
            "model": self.model,
            "tier": self.tier.value,
            "max_cost_per_request": self.max_cost_per_request,
            "max_cost_per_day": self.max_cost_per_day,
            "min_quality_score": self.min_quality_score,
            "max_latency_ms": self.max_latency_ms,
            "enabled": self.enabled,
            "weight": self.weight,
            "region": self.region,
            "api_key_env": self.api_key_env,
            "custom_config": self.custom_config
        }


@dataclass
class FallbackConfig:
    """Fallback configuration for when backend routing fails."""
    enabled: bool = True

    # Fallback providers in priority order
    providers: List[ProviderConfig] = field(default_factory=list)

    # Fallback behavior
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    timeout_seconds: float = 30.0

    # Circuit breaker
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 300.0

    def add_fallback_provider(
        self,
        name: str,
        model: str,
        tier: ProviderTier = ProviderTier.STANDARD,
        **kwargs
    ) -> None:
        """Add a fallback provider."""
        provider = ProviderConfig(
            name=name,
            model=model,
            tier=tier,
            **kwargs
        )
        self.providers.append(provider)


@dataclass
class RoutingConfig:
    """Complete routing configuration for Brokle AI platform."""
    strategy: RoutingStrategy = RoutingStrategy.BALANCED

    # Primary providers (backend will choose from these)
    primary_providers: List[ProviderConfig] = field(default_factory=list)

    # Fallback configuration
    fallback: FallbackConfig = field(default_factory=FallbackConfig)

    # Routing preferences
    preferences: Dict[str, Any] = field(default_factory=dict)

    # Performance requirements
    max_latency_ms: Optional[float] = None
    max_cost_per_request: Optional[float] = None
    min_quality_score: Optional[float] = None

    # Load balancing
    enable_load_balancing: bool = True
    round_robin: bool = False

    # Experimentation
    enable_ab_testing: bool = False
    experiment_ratio: float = 0.1  # 10% of requests for experiments

    def add_primary_provider(
        self,
        name: str,
        model: str,
        tier: ProviderTier = ProviderTier.STANDARD,
        **kwargs
    ) -> None:
        """Add a primary provider."""
        provider = ProviderConfig(
            name=name,
            model=model,
            tier=tier,
            **kwargs
        )
        self.primary_providers.append(provider)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "strategy": self.strategy.value,
            "primary_providers": [p.to_dict() for p in self.primary_providers],
            "fallback": {
                "enabled": self.fallback.enabled,
                "providers": [p.to_dict() for p in self.fallback.providers],
                "max_retries": self.fallback.max_retries,
                "retry_delay_seconds": self.fallback.retry_delay_seconds,
                "timeout_seconds": self.fallback.timeout_seconds,
                "failure_threshold": self.fallback.failure_threshold,
                "recovery_timeout_seconds": self.fallback.recovery_timeout_seconds
            },
            "preferences": self.preferences,
            "max_latency_ms": self.max_latency_ms,
            "max_cost_per_request": self.max_cost_per_request,
            "min_quality_score": self.min_quality_score,
            "enable_load_balancing": self.enable_load_balancing,
            "round_robin": self.round_robin,
            "enable_ab_testing": self.enable_ab_testing,
            "experiment_ratio": self.experiment_ratio
        }


class RoutingManager:
    """
    Client-side routing manager.

    Manages routing configuration and provides fallback capabilities
    when the backend routing service is unavailable.
    """

    def __init__(self):
        self.config = RoutingConfig()
        self._circuit_breaker_state: Dict[str, Dict[str, Any]] = {}

    def configure(self, config: RoutingConfig) -> None:
        """Configure routing settings."""
        self.config = config
        logger.info(f"Routing configured with strategy: {config.strategy.value}")

    def get_config(self) -> RoutingConfig:
        """Get current routing configuration."""
        return self.config

    def add_provider(
        self,
        name: str,
        model: str,
        tier: ProviderTier = ProviderTier.STANDARD,
        as_fallback: bool = False,
        **kwargs
    ) -> None:
        """Add a provider to routing configuration."""
        if as_fallback:
            self.config.fallback.add_fallback_provider(name, model, tier, **kwargs)
        else:
            self.config.add_primary_provider(name, model, tier, **kwargs)

    def create_cost_optimized_config(self) -> RoutingConfig:
        """Create a cost-optimized routing configuration template."""
        config = RoutingConfig(strategy=RoutingStrategy.COST_OPTIMIZED)

        # Set strategy preferences - backend will determine actual providers
        config.max_cost_per_request = 0.02  # Example constraint
        config.preferences = {
            "prioritize": "cost",
            "tier_preference": ["economy", "standard"],
            "allow_premium": False
        }

        return config

    def create_quality_optimized_config(self) -> RoutingConfig:
        """Create a quality-optimized routing configuration template."""
        config = RoutingConfig(strategy=RoutingStrategy.QUALITY_OPTIMIZED)

        # Set quality preferences - backend will determine actual providers
        config.min_quality_score = 0.85
        config.preferences = {
            "prioritize": "quality",
            "tier_preference": ["premium", "standard"],
            "min_model_size": "large"
        }

        return config

    def create_latency_optimized_config(self) -> RoutingConfig:
        """Create a latency-optimized routing configuration template."""
        config = RoutingConfig(strategy=RoutingStrategy.LATENCY_OPTIMIZED)

        # Set latency preferences - backend will determine actual providers
        config.max_latency_ms = 2000.0
        config.preferences = {
            "prioritize": "latency",
            "tier_preference": ["economy", "standard"],
            "max_response_time": "2s"
        }

        return config

    def is_provider_available(self, provider_name: str) -> bool:
        """Check if provider is available (circuit breaker logic)."""
        if provider_name not in self._circuit_breaker_state:
            return True

        state = self._circuit_breaker_state[provider_name]

        # Check if circuit breaker is open
        if state.get("open", False):
            # Check if recovery timeout has passed
            recovery_time = state.get("recovery_time")
            if recovery_time and datetime.utcnow() > recovery_time:
                # Reset circuit breaker
                self._circuit_breaker_state[provider_name] = {
                    "failures": 0,
                    "open": False
                }
                return True
            return False

        return True

    def record_provider_failure(self, provider_name: str) -> None:
        """Record a provider failure for circuit breaker logic."""
        if provider_name not in self._circuit_breaker_state:
            self._circuit_breaker_state[provider_name] = {"failures": 0, "open": False}

        state = self._circuit_breaker_state[provider_name]
        state["failures"] += 1

        # Check if we should open circuit breaker
        if state["failures"] >= self.config.fallback.failure_threshold:
            state["open"] = True
            state["recovery_time"] = datetime.utcnow() + timedelta(
                seconds=self.config.fallback.recovery_timeout_seconds
            )
            logger.warning(f"Circuit breaker opened for provider {provider_name}")

    def record_provider_success(self, provider_name: str) -> None:
        """Record a provider success (reset failures)."""
        if provider_name in self._circuit_breaker_state:
            self._circuit_breaker_state[provider_name]["failures"] = 0

    def get_available_providers(self, include_fallback: bool = False) -> List[ProviderConfig]:
        """Get list of currently available providers."""
        available = []

        # Check primary providers
        for provider in self.config.primary_providers:
            if provider.enabled and self.is_provider_available(provider.name):
                available.append(provider)

        # Include fallback providers if requested
        if include_fallback:
            for provider in self.config.fallback.providers:
                if provider.enabled and self.is_provider_available(provider.name):
                    available.append(provider)

        return available

    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get circuit breaker status for all providers."""
        return self._circuit_breaker_state.copy()


# Global routing manager instance
_routing_manager: Optional[RoutingManager] = None


def get_routing_manager() -> RoutingManager:
    """Get global routing manager instance."""
    global _routing_manager

    if _routing_manager is None:
        _routing_manager = RoutingManager()

    return _routing_manager


def configure_routing(config: RoutingConfig) -> None:
    """Configure global routing settings."""
    manager = get_routing_manager()
    manager.configure(config)


def get_routing_config() -> RoutingConfig:
    """Get current routing configuration."""
    manager = get_routing_manager()
    return manager.get_config()


def create_cost_optimized_routing() -> RoutingConfig:
    """Create cost-optimized routing configuration."""
    manager = get_routing_manager()
    return manager.create_cost_optimized_config()


def create_quality_optimized_routing() -> RoutingConfig:
    """Create quality-optimized routing configuration."""
    manager = get_routing_manager()
    return manager.create_quality_optimized_config()


def create_latency_optimized_routing() -> RoutingConfig:
    """Create latency-optimized routing configuration."""
    manager = get_routing_manager()
    return manager.create_latency_optimized_config()


def add_provider(
    name: str,
    model: str,
    tier: ProviderTier = ProviderTier.STANDARD,
    as_fallback: bool = False,
    **kwargs
) -> None:
    """Add provider to routing configuration."""
    manager = get_routing_manager()
    manager.add_provider(name, model, tier, as_fallback, **kwargs)