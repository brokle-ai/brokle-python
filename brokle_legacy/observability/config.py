"""
Observability configuration management.

Provides configuration access for Pattern 1/2 compatibility.
"""

from typing import Optional

from pydantic import BaseModel

from ..config import Config as BrokleConfig


class ObservabilityConfig(BaseModel):
    """
    Observability-specific configuration.

    Maintains compatibility with existing Pattern 1/2 code.
    """

    telemetry_enabled: bool = True
    debug: bool = False
    sample_rate: float = 1.0
    batch_size: int = 100
    flush_interval: int = 10000  # ms


# Global configuration instance
_observability_config: Optional[ObservabilityConfig] = None


def get_config() -> ObservabilityConfig:
    """
    Get observability configuration.

    Returns:
        ObservabilityConfig instance
    """
    global _observability_config

    if _observability_config is None:
        # Create default config
        _observability_config = ObservabilityConfig()

    return _observability_config


def telemetry_enabled() -> bool:
    """
    Check if telemetry is enabled.

    Returns:
        True if telemetry is enabled
    """
    return get_config().telemetry_enabled


def configure_observability(
    telemetry_enabled: bool = True,
    debug: bool = False,
    sample_rate: float = 1.0,
    batch_size: int = 100,
    flush_interval: int = 10000,
) -> None:
    """
    Configure observability settings.

    Args:
        telemetry_enabled: Enable/disable telemetry
        debug: Enable debug logging
        sample_rate: Sampling rate (0.0 to 1.0)
        batch_size: Batch size for telemetry
        flush_interval: Flush interval in milliseconds
    """
    global _observability_config

    _observability_config = ObservabilityConfig(
        telemetry_enabled=telemetry_enabled,
        debug=debug,
        sample_rate=sample_rate,
        batch_size=batch_size,
        flush_interval=flush_interval,
    )


def reset_config() -> None:
    """Reset observability configuration to defaults."""
    global _observability_config
    _observability_config = None
