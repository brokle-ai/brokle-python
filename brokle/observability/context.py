"""
Context management for observability.

Provides client context for Pattern 1/2 compatibility.
"""

import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, Optional

from ..client import Brokle

if TYPE_CHECKING:
    pass


class ObservabilityContext:
    """Thread-local context for observability."""

    def __init__(self):
        self._local = threading.local()

    def set_client(self, client: Brokle) -> None:
        """Set client for current thread."""
        self._local.client = client

    def get_client(self) -> Optional[Brokle]:
        """Get client for current thread."""
        return getattr(self._local, "client", None)

    def clear(self) -> None:
        """Clear context for current thread."""
        if hasattr(self._local, "client"):
            delattr(self._local, "client")

    def get_info(self) -> Dict[str, Any]:
        """Get context information."""
        client = self.get_client()
        if client:
            return {
                "has_client": True,
                "api_key": (
                    client.config.api_key[:10] + "..."
                    if client.config.api_key
                    else None
                ),
                "environment": client.config.environment,
                "host": client.config.host,
            }
        return {"has_client": False}


# Global context instance
_context = ObservabilityContext()


def get_client(
    api_key: Optional[str] = None,
    host: Optional[str] = None,
    environment: Optional[str] = None,
    otel_enabled: Optional[bool] = None,
    otel_endpoint: Optional[str] = None,
    otel_service_name: Optional[str] = None,
    otel_headers: Optional[Dict[str, str]] = None,
    telemetry_enabled: Optional[bool] = None,
    batch_max_size: Optional[int] = None,
    batch_flush_interval: Optional[float] = None,
    debug: Optional[bool] = None,
    timeout: Optional[int] = None,
    max_retries: Optional[int] = None,
    cache_enabled: Optional[bool] = None,
    routing_enabled: Optional[bool] = None,
    evaluation_enabled: Optional[bool] = None,
    **kwargs,
) -> Brokle:
    """
    Get or create Brokle client for observability.

    This function provides backward compatibility for Pattern 1/2 with thread-safe
    credential injection for production use.

    Args:
        api_key: Explicit API key (overrides environment)
        host: Explicit host URL (overrides environment)
        environment: Environment name
        otel_enabled: Enable OpenTelemetry integration
        otel_endpoint: OpenTelemetry endpoint
        otel_service_name: OpenTelemetry service name
        otel_headers: OpenTelemetry headers
        telemetry_enabled: Enable telemetry collection
        batch_max_size: Maximum events per batch (1-1000)
        batch_flush_interval: Batch flush interval in seconds
        debug: Enable debug logging
        timeout: HTTP timeout in seconds
        max_retries: Maximum retry attempts
        cache_enabled: Enable caching
        routing_enabled: Enable intelligent routing
        evaluation_enabled: Enable evaluation
        **kwargs: Additional configuration options

    Returns:
        Brokle client instance
    """
    # Check if any explicit credentials/config provided
    explicit_config = any(
        [
            api_key,
            host,
            environment,
            otel_enabled is not None,
            otel_endpoint,
            otel_service_name,
            otel_headers,
            telemetry_enabled is not None,
            batch_max_size is not None,
            batch_flush_interval is not None,
            debug is not None,
            timeout is not None,
            max_retries is not None,
            cache_enabled is not None,
            routing_enabled is not None,
            evaluation_enabled is not None,
            kwargs,
        ]
    )

    # If explicit credentials provided, create dedicated client (thread-safe)
    if explicit_config:
        # Filter out None values to avoid Config validation errors
        client_kwargs = {}

        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if host is not None:
            client_kwargs["host"] = host
        if environment is not None:
            client_kwargs["environment"] = environment
        if otel_enabled is not None:
            client_kwargs["otel_enabled"] = otel_enabled
        if otel_endpoint is not None:
            client_kwargs["otel_endpoint"] = otel_endpoint
        if otel_service_name is not None:
            client_kwargs["otel_service_name"] = otel_service_name
        if otel_headers is not None:
            client_kwargs["otel_headers"] = otel_headers
        if telemetry_enabled is not None:
            client_kwargs["telemetry_enabled"] = telemetry_enabled
        if batch_max_size is not None:
            client_kwargs["batch_max_size"] = batch_max_size
        if batch_flush_interval is not None:
            client_kwargs["batch_flush_interval"] = batch_flush_interval
        if debug is not None:
            client_kwargs["debug"] = debug
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        if max_retries is not None:
            client_kwargs["max_retries"] = max_retries
        if cache_enabled is not None:
            client_kwargs["cache_enabled"] = cache_enabled
        if routing_enabled is not None:
            client_kwargs["routing_enabled"] = routing_enabled
        if evaluation_enabled is not None:
            client_kwargs["evaluation_enabled"] = evaluation_enabled

        # Add any additional kwargs
        client_kwargs.update(kwargs)

        # Create dedicated client and store in thread-local context
        dedicated_client = Brokle(**client_kwargs)
        _context.set_client(dedicated_client)
        return dedicated_client

    # No explicit config = use thread-local singleton from environment
    client = _context.get_client()
    if client is None:
        # Create new client from environment variables (immutable)
        client = Brokle()
        _context.set_client(client)

    return client


def get_client_context() -> Optional[Brokle]:
    """
    Get client from context without creating new one.

    Returns:
        Brokle client if available, None otherwise
    """
    return _context.get_client()


def clear_context() -> None:
    """Clear the current context."""
    _context.clear()


def get_context_info() -> Dict[str, Any]:
    """
    Get context information for debugging.

    Returns:
        Dictionary with context info
    """
    return _context.get_info()


@contextmanager
def client_context(client: Brokle):
    """
    Context manager for setting client context.

    Args:
        client: Brokle client to use in context
    """
    old_client = _context.get_client()
    _context.set_client(client)
    try:
        yield client
    finally:
        if old_client:
            _context.set_client(old_client)
        else:
            _context.clear()
