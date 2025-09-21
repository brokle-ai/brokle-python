"""
OpenAI Auto-Instrumentation for Brokle Platform.

This module provides automatic instrumentation for OpenAI API calls using
the new BaseInstrumentation framework. Simply import this module to enable
comprehensive observability for all OpenAI usage.

Usage:
    import brokle.integrations.openai  # Enables auto-instrumentation

    from openai import OpenAI
    client = OpenAI()
    # All OpenAI calls now automatically tracked by Brokle
"""

import logging
from typing import Any, Dict, List

from .instrumentation import OpenAIInstrumentation
from .._registry import register_provider, is_provider_enabled
from .._engine import InstrumentationEngine

logger = logging.getLogger(__name__)

# Check for required dependencies
try:
    import wrapt
    HAS_WRAPT = True
except ImportError:
    HAS_WRAPT = False
    logger.warning("wrapt library not available - auto-instrumentation disabled")

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    openai = None

# Global instrumentation state
_instrumented = False
_instrumentation_errors = []

# Create OpenAI instrumentation instance
_openai_instrumentation = OpenAIInstrumentation()
_engine = InstrumentationEngine()


def _record_error(error: str) -> None:
    """Record an instrumentation error."""
    _instrumentation_errors.append(error)
    # Keep only last 10 errors
    _instrumentation_errors[:] = _instrumentation_errors[-10:]


def _instrument_openai_methods():
    """Instrument OpenAI methods using the new framework."""
    global _instrumented

    if _instrumented:
        logger.debug("OpenAI already instrumented")
        return True

    if not HAS_OPENAI:
        error_msg = "OpenAI library not available for instrumentation"
        _record_error(error_msg)
        logger.warning(error_msg)
        return False

    if not HAS_WRAPT:
        error_msg = "wrapt library not available for instrumentation"
        _record_error(error_msg)
        logger.warning(error_msg)
        return False

    # Check if provider is enabled
    if not is_provider_enabled("openai"):
        logger.debug("OpenAI instrumentation disabled via configuration")
        return False

    try:
        # Get method configurations from the instrumentation
        method_configs = _openai_instrumentation.get_method_configs()
        instrumented_count = 0

        for config in method_configs:
            try:
                # Create wrapper using the engine
                wrapper = _engine.create_wrapper(_openai_instrumentation, config)

                # Apply wrapper using wrapt
                wrapt.wrap_function_wrapper(
                    config.module,
                    config.object,
                    wrapper
                )

                instrumented_count += 1
                logger.debug(f"Instrumented {config.module}.{config.object}")

            except Exception as e:
                error_msg = f"Failed to instrument {config.object}: {e}"
                _record_error(error_msg)
                logger.debug(error_msg)

        if instrumented_count > 0:
            _instrumented = True
            logger.info(f"Brokle OpenAI auto-instrumentation enabled ({instrumented_count} methods)")
            return True
        else:
            error_msg = "No OpenAI methods could be instrumented"
            _record_error(error_msg)
            logger.warning(error_msg)
            return False

    except Exception as e:
        error_msg = f"Failed to instrument OpenAI library: {e}"
        _record_error(error_msg)
        logger.error(error_msg)
        return False


def is_instrumented() -> bool:
    """Check if OpenAI is currently instrumented."""
    return _instrumented


def get_instrumentation_errors() -> list:
    """Get list of instrumentation errors."""
    return _instrumentation_errors.copy()


def get_instrumentation_status() -> Dict[str, Any]:
    """Get detailed instrumentation status."""
    # Get status from registry if available
    try:
        from .._registry import get_provider_status
        provider_status = get_provider_status("openai")

        return {
            "instrumented": _instrumented,
            "openai_available": HAS_OPENAI,
            "wrapt_available": HAS_WRAPT,
            "errors": _instrumentation_errors.copy(),
            "provider_status": provider_status.to_dict(),
            "framework_status": {
                "using_new_framework": True,
                "engine_available": _engine is not None,
                "instrumentation_available": _openai_instrumentation is not None
            }
        }
    except Exception as e:
        return {
            "instrumented": _instrumented,
            "openai_available": HAS_OPENAI,
            "wrapt_available": HAS_WRAPT,
            "errors": _instrumentation_errors.copy() + [f"Registry error: {e}"],
            "framework_status": {
                "using_new_framework": True,
                "engine_available": _engine is not None,
                "instrumentation_available": _openai_instrumentation is not None
            }
        }


# Register provider and auto-instrument on import
try:
    # Register OpenAI provider in the registry
    register_provider("openai", _openai_instrumentation)
    logger.debug("OpenAI provider registered successfully")
except Exception as e:
    logger.debug(f"Failed to register OpenAI provider: {e}")
    _record_error(f"Provider registration failed: {e}")

# Auto-instrument if dependencies are available
if HAS_OPENAI and HAS_WRAPT:
    try:
        _instrument_openai_methods()
    except Exception as e:
        logger.warning(f"Auto-instrumentation failed on import: {e}")
        _record_error(f"Auto-instrumentation failed: {e}")
else:
    missing = []
    if not HAS_OPENAI:
        missing.append("openai")
    if not HAS_WRAPT:
        missing.append("wrapt")

    logger.debug(f"Auto-instrumentation skipped - missing dependencies: {', '.join(missing)}")
    _record_error(f"Missing dependencies: {', '.join(missing)}")


# Export public interface
__all__ = [
    "is_instrumented",
    "get_instrumentation_errors",
    "get_instrumentation_status",
    "HAS_OPENAI",
    "HAS_WRAPT"
]
