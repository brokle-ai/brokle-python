"""
Runtime Controls for Brokle Platform Instrumentation.

This module provides runtime control APIs for enabling and disabling
instrumentation across all providers. It integrates with the provider
registry and environment configuration.

Features:
- Global instrumentation controls
- Per-provider enable/disable
- Environment variable configuration
- Status reporting and debugging
- Graceful fallback handling

Usage:
    # Global controls
    disable_all_instrumentation()
    enable_all_instrumentation()

    # Provider-specific controls
    disable_provider("openai")
    enable_provider("anthropic")

    # Status reporting
    status = get_instrumentation_status()
"""

import logging
import os
from typing import Any, Dict, List, Optional

from ._registry import (
    disable_all_instrumentation as registry_disable_all,
    enable_all_instrumentation as registry_enable_all,
    disable_provider as registry_disable_provider,
    enable_provider as registry_enable_provider,
    get_all_provider_status,
    get_debug_info
)

logger = logging.getLogger(__name__)


class InstrumentationConfig:
    """Configuration manager for instrumentation settings."""

    def __init__(self):
        """Initialize configuration from environment variables."""
        self._load_config()

    def _load_config(self):
        """Load configuration from environment variables."""
        # Global instrumentation control
        self.auto_instrument = self._env_bool("BROKLE_AUTO_INSTRUMENT", True)

        # Per-provider controls
        self.providers = {
            "openai": self._env_bool("BROKLE_AUTO_INSTRUMENT_OPENAI", True),
            "anthropic": self._env_bool("BROKLE_AUTO_INSTRUMENT_ANTHROPIC", True),
            "gemini": self._env_bool("BROKLE_AUTO_INSTRUMENT_GEMINI", True),
            "bedrock": self._env_bool("BROKLE_AUTO_INSTRUMENT_BEDROCK", True),
            "cohere": self._env_bool("BROKLE_AUTO_INSTRUMENT_COHERE", True),
        }

        # Debug and development settings
        self.debug_mode = self._env_bool("BROKLE_INSTRUMENTATION_DEBUG", False)
        self.verbose_errors = self._env_bool("BROKLE_INSTRUMENTATION_VERBOSE_ERRORS", False)

        # Performance settings
        self.circuit_breaker_threshold = int(os.getenv("BROKLE_CIRCUIT_BREAKER_THRESHOLD", "5"))
        self.circuit_breaker_reset = int(os.getenv("BROKLE_CIRCUIT_BREAKER_RESET", "3"))

        logger.debug(f"Loaded instrumentation config: {self.to_dict()}")

    def _env_bool(self, key: str, default: bool) -> bool:
        """Parse boolean environment variable."""
        value = os.getenv(key, "").lower()
        if value in ("true", "1", "yes", "on"):
            return True
        elif value in ("false", "0", "no", "off"):
            return False
        else:
            return default

    def is_provider_enabled(self, provider: str) -> bool:
        """Check if a provider should be instrumented based on config."""
        if not self.auto_instrument:
            return False

        return self.providers.get(provider, True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "auto_instrument": self.auto_instrument,
            "providers": self.providers.copy(),
            "debug_mode": self.debug_mode,
            "verbose_errors": self.verbose_errors,
            "circuit_breaker_threshold": self.circuit_breaker_threshold,
            "circuit_breaker_reset": self.circuit_breaker_reset
        }


# Global configuration instance
_config = InstrumentationConfig()


def get_config() -> InstrumentationConfig:
    """Get global instrumentation configuration."""
    return _config


def reload_config() -> InstrumentationConfig:
    """Reload configuration from environment variables."""
    global _config
    _config = InstrumentationConfig()
    return _config


def disable_all_instrumentation() -> None:
    """
    Disable all instrumentation globally.

    This affects both auto-instrumentation and manual wrappers.
    Existing instrumentation will be disabled but wrappers remain in place.
    """
    try:
        registry_disable_all()
        logger.info("All instrumentation disabled")
    except Exception as e:
        logger.error(f"Failed to disable all instrumentation: {e}")


def enable_all_instrumentation() -> None:
    """
    Enable all instrumentation globally.

    This re-enables both auto-instrumentation and manual wrappers.
    """
    try:
        registry_enable_all()
        logger.info("All instrumentation enabled")
    except Exception as e:
        logger.error(f"Failed to enable all instrumentation: {e}")


def disable_provider(provider: str) -> None:
    """
    Disable instrumentation for a specific provider.

    Args:
        provider: Provider name (e.g., "openai", "anthropic")
    """
    try:
        registry_disable_provider(provider)
        logger.info(f"Disabled instrumentation for provider: {provider}")
    except Exception as e:
        logger.error(f"Failed to disable provider {provider}: {e}")


def enable_provider(provider: str) -> None:
    """
    Enable instrumentation for a specific provider.

    Args:
        provider: Provider name (e.g., "openai", "anthropic")
    """
    try:
        registry_enable_provider(provider)
        logger.info(f"Enabled instrumentation for provider: {provider}")
    except Exception as e:
        logger.error(f"Failed to enable provider {provider}: {e}")


def get_instrumentation_status() -> Dict[str, Any]:
    """
    Get comprehensive instrumentation status.

    Returns:
        Dictionary with status information across all providers
    """
    try:
        provider_status = get_all_provider_status()

        return {
            "global_enabled": _config.auto_instrument,
            "configuration": _config.to_dict(),
            "providers": {
                name: {
                    "available": status.available,
                    "instrumented": status.instrumented,
                    "enabled_in_config": _config.is_provider_enabled(name),
                    "version": status.version,
                    "methods_wrapped": status.methods_wrapped,
                    "errors": status.errors
                }
                for name, status in provider_status.items()
            },
            "environment_variables": {
                "BROKLE_AUTO_INSTRUMENT": os.getenv("BROKLE_AUTO_INSTRUMENT"),
                "BROKLE_AUTO_INSTRUMENT_OPENAI": os.getenv("BROKLE_AUTO_INSTRUMENT_OPENAI"),
                "BROKLE_AUTO_INSTRUMENT_ANTHROPIC": os.getenv("BROKLE_AUTO_INSTRUMENT_ANTHROPIC"),
                "BROKLE_INSTRUMENTATION_DEBUG": os.getenv("BROKLE_INSTRUMENTATION_DEBUG"),
            }
        }

    except Exception as e:
        logger.error(f"Failed to get instrumentation status: {e}")
        return {
            "error": f"Failed to get status: {e}",
            "global_enabled": _config.auto_instrument,
            "configuration": _config.to_dict()
        }


def get_enabled_providers() -> List[str]:
    """Get list of currently enabled providers."""
    try:
        status = get_instrumentation_status()
        return [
            name for name, info in status.get("providers", {}).items()
            if info.get("instrumented", False)
        ]
    except Exception as e:
        logger.error(f"Failed to get enabled providers: {e}")
        return []


def get_available_providers() -> List[str]:
    """Get list of available providers (libraries installed)."""
    try:
        status = get_instrumentation_status()
        return [
            name for name, info in status.get("providers", {}).items()
            if info.get("available", False)
        ]
    except Exception as e:
        logger.error(f"Failed to get available providers: {e}")
        return []


def debug_status() -> Dict[str, Any]:
    """
    Get comprehensive debug information for troubleshooting.

    Returns:
        Detailed debug information including errors, configuration, and registry state
    """
    try:
        debug_info = get_debug_info()
        instrumentation_status = get_instrumentation_status()

        return {
            "instrumentation_status": instrumentation_status,
            "registry_debug": debug_info,
            "configuration": _config.to_dict(),
            "environment": {
                key: os.getenv(key)
                for key in os.environ.keys()
                if key.startswith("BROKLE_")
            },
            "import_status": _check_import_status(),
            "recommendations": _get_recommendations(instrumentation_status)
        }

    except Exception as e:
        logger.error(f"Failed to get debug status: {e}")
        return {
            "error": f"Debug status failed: {e}",
            "configuration": _config.to_dict()
        }


def _check_import_status() -> Dict[str, bool]:
    """Check which provider libraries are importable."""
    providers = ["openai", "anthropic", "google.generativeai", "boto3", "cohere"]
    status = {}

    for provider in providers:
        try:
            __import__(provider)
            status[provider] = True
        except ImportError:
            status[provider] = False

    return status


def _get_recommendations(status: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on current status."""
    recommendations = []

    # Check for disabled instrumentation
    if not status.get("global_enabled", True):
        recommendations.append("Global instrumentation is disabled. Set BROKLE_AUTO_INSTRUMENT=true to enable.")

    # Check for providers with errors
    for name, info in status.get("providers", {}).items():
        if info.get("errors"):
            recommendations.append(f"Provider '{name}' has errors: {info['errors'][-1]}")

        if info.get("available") and not info.get("instrumented"):
            recommendations.append(f"Provider '{name}' is available but not instrumented. Check configuration.")

    # Check for missing dependencies
    if not recommendations:
        recommendations.append("All available providers are properly instrumented.")

    return recommendations


# Auto-apply environment configuration on import
try:
    # Apply environment-based disable settings
    if not _config.auto_instrument:
        disable_all_instrumentation()
        logger.info("Auto-instrumentation disabled via BROKLE_AUTO_INSTRUMENT=false")
    else:
        # Apply per-provider settings
        for provider, enabled in _config.providers.items():
            if not enabled:
                disable_provider(provider)
                logger.debug(f"Provider '{provider}' disabled via environment variable")

except Exception as e:
    logger.debug(f"Failed to apply environment configuration: {e}")


# Export public API
__all__ = [
    "disable_all_instrumentation",
    "enable_all_instrumentation",
    "disable_provider",
    "enable_provider",
    "get_instrumentation_status",
    "get_enabled_providers",
    "get_available_providers",
    "debug_status",
    "get_config",
    "reload_config"
]