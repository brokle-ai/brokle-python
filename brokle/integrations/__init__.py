"""
Brokle Platform Integrations.

This module provides the unified public API for all provider integrations,
manual wrappers, and runtime controls. It serves as the main entry point
for all instrumentation functionality.

Features:
- Auto-instrumentation for multiple providers
- Manual wrapper functions
- Runtime control APIs
- Status reporting and debugging
- Environment variable configuration

Usage:
    # Auto-instrumentation (import triggers instrumentation)
    import brokle.integrations.openai
    import brokle.integrations.anthropic

    # Manual wrappers
    from brokle.integrations import track_openai, track_anthropic
    client = track_openai(OpenAI())

    # Runtime controls
    from brokle.integrations import disable_provider, get_status
    disable_provider("openai")
    status = get_status()

    # Decorator pattern
    from brokle.integrations import observe
    @observe()
    def my_function():
        pass
"""

import logging
from typing import Any, Dict, List, Optional

# Import manual wrapper functions
from .manual import (
    track_openai,
    track_anthropic,
    track_client,
    observe,
    get_tracking_status
)

# Import runtime controls
from .controls import (
    disable_all_instrumentation,
    enable_all_instrumentation,
    disable_provider,
    enable_provider,
    get_instrumentation_status,
    get_enabled_providers,
    get_available_providers,
    debug_status,
    get_config,
    reload_config
)

# Import registry functions
from ._registry import (
    register_provider,
    get_provider_instrumentation,
    get_all_provider_instrumentations,
    is_provider_enabled,
    auto_discover_providers
)

# Legacy imports for backward compatibility
try:
    from .openai import (
        is_instrumented as openai_is_instrumented,
        get_instrumentation_status as openai_get_status,
        HAS_OPENAI,
        HAS_WRAPT
    )
    OPENAI_AUTO_AVAILABLE = bool(HAS_OPENAI and HAS_WRAPT)
except ImportError:
    OPENAI_AUTO_AVAILABLE = False
    openai_is_instrumented = lambda: False
    openai_get_status = lambda: {"instrumented": False, "error": "Auto-instrumentation not available"}
    HAS_OPENAI = False
    HAS_WRAPT = False

logger = logging.getLogger(__name__)

def get_status() -> Dict[str, Any]:
    """
    Get comprehensive status of all integrations.

    This is the main status function that combines information from
    all subsystems for easy debugging and monitoring.

    Returns:
        Dictionary with complete status information
    """
    try:
        instrumentation_status = get_instrumentation_status()
        tracking_status = get_tracking_status()

        return {
            "framework_version": "2.0",
            "framework_features": {
                "auto_instrumentation": True,
                "manual_wrappers": True,
                "runtime_controls": True,
                "multi_provider": True,
                "environment_config": True
            },
            "instrumentation": instrumentation_status,
            "manual_tracking": tracking_status,
            "summary": {
                "total_providers": len(instrumentation_status.get("providers", {})),
                "enabled_providers": len(get_enabled_providers()),
                "available_providers": len(get_available_providers()),
                "global_enabled": instrumentation_status.get("global_enabled", False)
            }
        }

    except Exception as e:
        logger.error(f"Failed to get comprehensive status: {e}")
        return {
            "error": f"Status collection failed: {e}",
            "framework_version": "2.0",
            "partial_status": True
        }


def list_providers() -> List[str]:
    """List all registered providers."""
    try:
        return list(get_all_provider_instrumentations().keys())
    except Exception as e:
        logger.error(f"Failed to list providers: {e}")
        return []


def is_provider_available(provider: str) -> bool:
    """Check if a provider is available (library installed)."""
    try:
        status = get_status()
        provider_info = status.get("instrumentation", {}).get("providers", {}).get(provider, {})
        return provider_info.get("available", False)
    except Exception as e:
        logger.error(f"Failed to check provider availability: {e}")
        return False


def is_provider_instrumented(provider: str) -> bool:
    """Check if a provider is currently instrumented."""
    try:
        status = get_status()
        provider_info = status.get("instrumentation", {}).get("providers", {}).get(provider, {})
        return provider_info.get("instrumented", False)
    except Exception as e:
        logger.error(f"Failed to check provider instrumentation: {e}")
        return False


def get_provider_info(provider: str) -> Dict[str, Any]:
    """Get detailed information about a specific provider."""
    try:
        status = get_status()
        return status.get("instrumentation", {}).get("providers", {}).get(provider, {})
    except Exception as e:
        logger.error(f"Failed to get provider info: {e}")
        return {"error": f"Failed to get info: {e}"}


def validate_setup() -> Dict[str, Any]:
    """
    Validate the current setup and provide recommendations.

    Returns:
        Dictionary with validation results and recommendations
    """
    try:
        status = get_status()
        validation = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }

        # Check global settings
        if not status.get("instrumentation", {}).get("global_enabled", True):
            validation["warnings"].append("Global instrumentation is disabled")
            validation["recommendations"].append("Set BROKLE_AUTO_INSTRUMENT=true to enable")

        # Check providers
        providers = status.get("instrumentation", {}).get("providers", {})
        available_count = 0
        instrumented_count = 0

        for name, info in providers.items():
            if info.get("available"):
                available_count += 1
                if info.get("instrumented"):
                    instrumented_count += 1
                else:
                    validation["warnings"].append(f"Provider '{name}' available but not instrumented")

            if info.get("errors"):
                validation["errors"].extend([f"{name}: {error}" for error in info["errors"]])

        # Overall assessment
        if available_count == 0:
            validation["errors"].append("No AI provider libraries found")
            validation["recommendations"].append("Install at least one provider: pip install openai anthropic")
        elif instrumented_count == 0:
            validation["errors"].append("No providers are instrumented")
            validation["valid"] = False
        elif instrumented_count < available_count:
            validation["recommendations"].append("Consider enabling all available providers")

        if not validation["errors"] and not validation["warnings"]:
            validation["recommendations"].append("Setup looks good! All available providers are instrumented.")

        return validation

    except Exception as e:
        logger.error(f"Setup validation failed: {e}")
        return {
            "valid": False,
            "errors": [f"Validation failed: {e}"],
            "warnings": [],
            "recommendations": ["Check logs for detailed error information"]
        }


# Convenience aliases for common functions
status = get_status
disable = disable_provider
enable = enable_provider
providers = list_providers


# Export public API
__all__ = [
    # Manual wrapper functions
    "track_openai",
    "track_anthropic",
    "track_client",
    "observe",

    # Runtime controls
    "disable_all_instrumentation",
    "enable_all_instrumentation",
    "disable_provider",
    "enable_provider",
    "disable",  # Alias
    "enable",   # Alias

    # Status and information
    "get_status",
    "status",  # Alias
    "get_instrumentation_status",
    "get_tracking_status",
    "debug_status",
    "validate_setup",

    # Provider management
    "list_providers",
    "providers",  # Alias
    "get_enabled_providers",
    "get_available_providers",
    "is_provider_available",
    "is_provider_instrumented",
    "get_provider_info",

    # Configuration
    "get_config",
    "reload_config",

    # Advanced
    "register_provider",
    "auto_discover_providers",

    # Legacy compatibility
    "openai_is_instrumented",
    "openai_get_status",
    "HAS_OPENAI",
    "HAS_WRAPT",
    "OPENAI_AUTO_AVAILABLE"
]
