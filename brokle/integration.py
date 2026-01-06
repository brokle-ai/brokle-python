"""
Base Integration Class and Interface

Provides a standardized contract for all Brokle integrations
following the patterns defined in the integration ecosystem design.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from brokle import Brokle

logger = logging.getLogger(__name__)


class IntegrationType(Enum):
    """Integration type classification."""
    WRAPPER = "wrapper"
    ADAPTER = "adapter"
    INSTRUMENTATION = "instrumentation"
    CALLBACK = "callback"


class IntegrationStatus(Enum):
    """Integration status."""
    REGISTERED = "registered"
    ACTIVE = "active"
    DISABLED = "disabled"
    ERROR = "error"


@dataclass
class IntegrationMetadata:
    """Integration metadata."""
    name: str
    version: str
    type: IntegrationType
    provider: str
    description: Optional[str] = None
    features: Optional[List[str]] = None
    docs_url: Optional[str] = None


@dataclass
class IntegrationConfig:
    """Integration configuration."""
    enabled: bool = True
    capture_input: bool = True
    capture_output: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrationStats:
    """Integration statistics."""
    operation_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_tokens: int = 0
    last_operation_at: Optional[datetime] = None
    last_error_at: Optional[datetime] = None
    avg_latency_ms: float = 0.0


class BrokleIntegration(ABC):
    """
    Base Integration Interface.

    Every integration implements this contract for consistent behavior
    across the Brokle ecosystem.

    Example:
        >>> class MyIntegration(BaseIntegration):
        ...     @property
        ...     def metadata(self) -> IntegrationMetadata:
        ...         return IntegrationMetadata(
        ...             name="my-integration",
        ...             version="1.0.0",
        ...             type=IntegrationType.WRAPPER,
        ...             provider="my-provider"
        ...         )
        ...     def _on_enable(self) -> None:
        ...         pass
        ...     def _on_disable(self) -> None:
        ...         pass
    """

    @property
    @abstractmethod
    def metadata(self) -> IntegrationMetadata:
        """Integration metadata."""
        ...

    @property
    @abstractmethod
    def status(self) -> IntegrationStatus:
        """Current status."""
        ...

    @property
    @abstractmethod
    def stats(self) -> IntegrationStats:
        """Integration statistics."""
        ...

    @abstractmethod
    def register(self, client: "Brokle") -> None:
        """Register the integration with a Brokle client."""
        ...

    @abstractmethod
    def unregister(self) -> None:
        """Unregister the integration."""
        ...

    @abstractmethod
    def is_enabled(self) -> bool:
        """Check if the integration is enabled."""
        ...

    @abstractmethod
    def enable(self) -> None:
        """Enable the integration."""
        ...

    @abstractmethod
    def disable(self) -> None:
        """Disable the integration."""
        ...

    @abstractmethod
    def reset_stats(self) -> None:
        """Reset statistics."""
        ...


class BaseIntegration(BrokleIntegration):
    """
    Abstract base class for integrations.

    Provides common functionality for all integrations.
    """

    def __init__(self, config: Optional[IntegrationConfig] = None) -> None:
        self._config = config or IntegrationConfig()
        self._status = IntegrationStatus.DISABLED
        self._client: Optional["Brokle"] = None
        self._stats = IntegrationStats()

    @property
    @abstractmethod
    def metadata(self) -> IntegrationMetadata:
        """Integration metadata - must be implemented by subclass."""
        ...

    @property
    def status(self) -> IntegrationStatus:
        """Current status."""
        return self._status

    @property
    def stats(self) -> IntegrationStats:
        """Integration statistics (returns a copy)."""
        return IntegrationStats(
            operation_count=self._stats.operation_count,
            success_count=self._stats.success_count,
            error_count=self._stats.error_count,
            total_tokens=self._stats.total_tokens,
            last_operation_at=self._stats.last_operation_at,
            last_error_at=self._stats.last_error_at,
            avg_latency_ms=self._stats.avg_latency_ms,
        )

    @property
    def config(self) -> IntegrationConfig:
        """Integration configuration."""
        return self._config

    def register(self, client: "Brokle") -> None:
        """Register the integration with a Brokle client."""
        if self._status not in (IntegrationStatus.DISABLED, IntegrationStatus.ERROR):
            logger.warning(
                f"Integration {self.metadata.name} is already registered"
            )
            return

        self._client = client
        self._status = IntegrationStatus.REGISTERED

        if self._config.enabled:
            self.enable()

        self._on_register()

        if self._client and self._client.config.debug:
            logger.debug(
                f"Integration {self.metadata.name} v{self.metadata.version} registered"
            )

    def unregister(self) -> None:
        """Unregister the integration."""
        if self._status == IntegrationStatus.DISABLED:
            return

        self.disable()
        self._on_unregister()

        if self._client and self._client.config.debug:
            logger.debug(f"Integration {self.metadata.name} unregistered")

        self._client = None
        self._status = IntegrationStatus.DISABLED

    def is_enabled(self) -> bool:
        """Check if the integration is enabled."""
        return self._status == IntegrationStatus.ACTIVE

    def enable(self) -> None:
        """Enable the integration."""
        if not self._client:
            raise RuntimeError(
                f"Integration {self.metadata.name} must be registered before enabling"
            )

        if self._status == IntegrationStatus.ACTIVE:
            return

        try:
            self._on_enable()
            self._status = IntegrationStatus.ACTIVE

            if self._client.config.debug:
                logger.debug(f"Integration {self.metadata.name} enabled")
        except Exception as e:
            self._status = IntegrationStatus.ERROR
            logger.error(f"Failed to enable integration {self.metadata.name}: {e}")
            raise

    def disable(self) -> None:
        """Disable the integration."""
        if self._status != IntegrationStatus.ACTIVE:
            return

        try:
            self._on_disable()
            self._status = IntegrationStatus.REGISTERED

            if self._client and self._client.config.debug:
                logger.debug(f"Integration {self.metadata.name} disabled")
        except Exception as e:
            logger.error(f"Error disabling integration {self.metadata.name}: {e}")

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = IntegrationStats()

    def _record_success(self, latency_ms: float, tokens: Optional[int] = None) -> None:
        """Record a successful operation."""
        self._stats.operation_count += 1
        self._stats.success_count += 1
        self._stats.last_operation_at = datetime.now()

        if tokens:
            self._stats.total_tokens += tokens

        # Update rolling average latency
        total_latency = (
            self._stats.avg_latency_ms * (self._stats.success_count - 1) + latency_ms
        )
        self._stats.avg_latency_ms = total_latency / self._stats.success_count

    def _record_error(self) -> None:
        """Record a failed operation."""
        self._stats.operation_count += 1
        self._stats.error_count += 1
        self._stats.last_error_at = datetime.now()

    def _get_client(self) -> "Brokle":
        """Get the Brokle client."""
        if not self._client:
            raise RuntimeError(f"Integration {self.metadata.name} is not registered")
        return self._client

    def _is_client_enabled(self) -> bool:
        """Check if the client is properly configured."""
        return self._client.config.enabled if self._client else False

    def _on_register(self) -> None:
        """Hook called when the integration is registered."""
        pass

    def _on_unregister(self) -> None:
        """Hook called when the integration is unregistered."""
        pass

    @abstractmethod
    def _on_enable(self) -> None:
        """Hook called when the integration is enabled."""
        ...

    @abstractmethod
    def _on_disable(self) -> None:
        """Hook called when the integration is disabled."""
        ...


class IntegrationRegistry:
    """
    Integration Registry.

    Manages registered integrations.

    Example:
        >>> registry = IntegrationRegistry()
        >>> registry.register(my_integration, brokle_client)
        >>> registry.get("my-integration")
    """

    def __init__(self) -> None:
        self._integrations: Dict[str, BrokleIntegration] = {}

    def register(
        self,
        integration: BrokleIntegration,
        client: "Brokle"
    ) -> None:
        """Register an integration."""
        name = integration.metadata.name

        if name in self._integrations:
            logger.warning(
                f"Integration {name} is already registered, replacing..."
            )
            self.unregister(name)

        integration.register(client)
        self._integrations[name] = integration

    def unregister(self, name: str) -> bool:
        """Unregister an integration by name."""
        integration = self._integrations.get(name)
        if integration:
            integration.unregister()
            del self._integrations[name]
            return True
        return False

    def get(self, name: str) -> Optional[BrokleIntegration]:
        """Get an integration by name."""
        return self._integrations.get(name)

    def get_all(self) -> List[BrokleIntegration]:
        """Get all registered integrations."""
        return list(self._integrations.values())

    def get_names(self) -> List[str]:
        """Get all integration names."""
        return list(self._integrations.keys())

    def has(self, name: str) -> bool:
        """Check if an integration is registered."""
        return name in self._integrations

    @property
    def count(self) -> int:
        """Get count of registered integrations."""
        return len(self._integrations)

    def enable_all(self) -> None:
        """Enable all integrations."""
        for integration in self._integrations.values():
            integration.enable()

    def disable_all(self) -> None:
        """Disable all integrations."""
        for integration in self._integrations.values():
            integration.disable()

    def clear(self) -> None:
        """Unregister all integrations."""
        for integration in list(self._integrations.values()):
            integration.unregister()
        self._integrations.clear()

    def get_stats(self) -> Dict[str, IntegrationStats]:
        """Get summary statistics across all integrations."""
        return {
            name: integration.stats
            for name, integration in self._integrations.items()
        }


# Global integration registry
_global_registry: Optional[IntegrationRegistry] = None


def get_integration_registry() -> IntegrationRegistry:
    """Get the global integration registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = IntegrationRegistry()
    return _global_registry


def reset_integration_registry() -> None:
    """Reset the global integration registry (mainly for testing)."""
    global _global_registry
    if _global_registry:
        _global_registry.clear()
    _global_registry = None
