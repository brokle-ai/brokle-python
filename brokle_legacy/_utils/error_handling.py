"""
Error handling utilities for Brokle SDK.

Clean, reusable error handling functions for consistent error management
across all SDK components.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def handle_provider_error(error: Exception, provider: str, operation: str) -> Exception:
    """
    Handle provider-specific errors with consistent processing.

    Args:
        error: Original exception
        provider: Provider name (e.g., "openai", "anthropic")
        operation: Operation name (e.g., "chat.completions.create")

    Returns:
        Processed exception with additional context
    """
    try:
        from ..exceptions import AuthenticationError, ProviderError, RateLimitError

        error_msg = str(error)
        error_type = type(error).__name__

        # Check for common error patterns
        if "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
            return RateLimitError(f"{provider} rate limit exceeded: {error_msg}")

        if "auth" in error_msg.lower() or "unauthorized" in error_msg.lower():
            return AuthenticationError(f"{provider} authentication failed: {error_msg}")

        if "api key" in error_msg.lower() or "invalid key" in error_msg.lower():
            return AuthenticationError(f"{provider} API key invalid: {error_msg}")

        # Generic provider error
        return ProviderError(f"{provider} {operation} failed: {error_msg}")

    except Exception as e:
        logger.warning(f"Failed to process provider error: {e}")
        return error  # Return original error if processing fails


def safe_str(obj: Any, max_length: int = 1000) -> str:
    """
    Safely convert object to string with length limits.

    Args:
        obj: Object to convert
        max_length: Maximum string length

    Returns:
        Safe string representation
    """
    try:
        if obj is None:
            return "None"

        str_repr = str(obj)
        if len(str_repr) <= max_length:
            return str_repr

        return str_repr[:max_length] + "...[TRUNCATED]"

    except Exception:
        return f"<{type(obj).__name__} - conversion failed>"


def log_error_safely(
    logger_obj: logging.Logger, message: str, error: Exception, **kwargs
):
    """
    Safely log errors with context.

    Args:
        logger_obj: Logger instance
        message: Error message
        error: Exception to log
        **kwargs: Additional context
    """
    try:
        context = {
            "error_type": type(error).__name__,
            "error_message": safe_str(error),
            **kwargs,
        }

        logger_obj.error(message, extra=context)

    except Exception:
        # Fallback logging if structured logging fails
        try:
            logger_obj.error(f"{message}: {error}")
        except Exception:
            pass  # Silent fail to prevent logging loops


def handle_import_error(
    module_name: str, error: ImportError, required: bool = False
) -> Optional[Any]:
    """
    Handle import errors with appropriate logging and fallbacks.

    Args:
        module_name: Name of module that failed to import
        error: Import error exception
        required: Whether the module is required for operation

    Returns:
        None (module not available)

    Raises:
        ImportError: If module is required and not available
    """
    try:
        if required:
            logger.error(f"Required module {module_name} not available: {error}")
            raise ImportError(
                f"Required dependency {module_name} not installed: {error}"
            )
        else:
            logger.warning(f"Optional module {module_name} not available: {error}")
            return None

    except Exception as e:
        logger.error(f"Error handling import failure for {module_name}: {e}")
        return None


def create_error_context(operation: str, **kwargs) -> Dict[str, Any]:
    """
    Create standardized error context dictionary.

    Args:
        operation: Operation name
        **kwargs: Additional context

    Returns:
        Error context dictionary
    """
    context = {
        "operation": operation,
        "timestamp": None,  # Would be filled by logging system
    }

    # Add safe string representations of context
    for key, value in kwargs.items():
        context[key] = safe_str(value, max_length=500)

    return context
