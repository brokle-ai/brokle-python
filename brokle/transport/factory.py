"""
Transport factory for OTLP exporters.

Provides factory functions for creating HTTP or gRPC exporters
based on configuration.

The backend supports both protocols:
- HTTP: /v1/traces, /v1/metrics, /v1/logs on port 8080
- gRPC: Port 4317 (standard OTLP gRPC port)
"""

import logging
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional

from opentelemetry.sdk._logs.export import LogExporter
from opentelemetry.sdk.metrics.export import MetricExporter
from opentelemetry.sdk.trace.export import SpanExporter

if TYPE_CHECKING:
    from ..config import BrokleConfig

logger = logging.getLogger(__name__)


class TransportType(str, Enum):
    """Transport type for OTLP export."""

    HTTP = "http"
    """HTTP/Protobuf transport (default) - uses /v1/traces, /v1/metrics"""

    GRPC = "grpc"
    """gRPC transport - uses port 4317 (requires grpc extras)"""


def create_trace_exporter(
    config: "BrokleConfig",
    transport: TransportType = TransportType.HTTP,
) -> SpanExporter:
    """
    Create a trace exporter based on transport type.

    Args:
        config: Brokle configuration
        transport: Transport type (http or grpc)

    Returns:
        Configured SpanExporter

    Raises:
        ImportError: If required exporter package is not installed

    Example:
        >>> exporter = create_trace_exporter(config, TransportType.GRPC)
    """
    if transport == TransportType.GRPC:
        return _create_grpc_trace_exporter(config)
    else:
        return _create_http_trace_exporter(config)


def create_metric_exporter(
    config: "BrokleConfig",
    transport: TransportType = TransportType.HTTP,
) -> MetricExporter:
    """
    Create a metric exporter based on transport type.

    Args:
        config: Brokle configuration
        transport: Transport type (http or grpc)

    Returns:
        Configured MetricExporter

    Raises:
        ImportError: If required exporter package is not installed
    """
    if transport == TransportType.GRPC:
        return _create_grpc_metric_exporter(config)
    else:
        return _create_http_metric_exporter(config)


def create_log_exporter(
    config: "BrokleConfig",
    transport: TransportType = TransportType.HTTP,
) -> LogExporter:
    """
    Create a log exporter based on transport type.

    Args:
        config: Brokle configuration
        transport: Transport type (http or grpc)

    Returns:
        Configured LogExporter

    Raises:
        ImportError: If required exporter package is not installed

    Example:
        >>> exporter = create_log_exporter(config, TransportType.HTTP)
    """
    if transport == TransportType.GRPC:
        return _create_grpc_log_exporter(config)
    else:
        return _create_http_log_exporter(config)


def _create_http_trace_exporter(config: "BrokleConfig") -> SpanExporter:
    """Create HTTP/Protobuf trace exporter."""
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    endpoint = _get_http_endpoint(config, "traces")
    headers = _get_headers(config)

    logger.debug(f"Creating HTTP trace exporter for {endpoint}")

    return OTLPSpanExporter(
        endpoint=endpoint,
        headers=headers,
        timeout=config.timeout,
    )


def _create_http_metric_exporter(config: "BrokleConfig") -> MetricExporter:
    """Create HTTP/Protobuf metric exporter."""
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
        OTLPMetricExporter,
    )

    endpoint = _get_http_endpoint(config, "metrics")
    headers = _get_headers(config)

    logger.debug(f"Creating HTTP metric exporter for {endpoint}")

    return OTLPMetricExporter(
        endpoint=endpoint,
        headers=headers,
        timeout=config.timeout,
    )


def _create_grpc_trace_exporter(config: "BrokleConfig") -> SpanExporter:
    """
    Create gRPC trace exporter.

    Requires: opentelemetry-exporter-otlp-proto-grpc package

    gRPC connects to port 4317 by default (standard OTLP gRPC port).
    """
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
    except ImportError as e:
        raise ImportError(
            "gRPC transport requires opentelemetry-exporter-otlp-proto-grpc. "
            "Install with: pip install opentelemetry-exporter-otlp-proto-grpc"
        ) from e

    endpoint = _get_grpc_endpoint(config)
    headers = _get_grpc_headers(config)

    logger.debug(f"Creating gRPC trace exporter for {endpoint}")

    return OTLPSpanExporter(
        endpoint=endpoint,
        headers=headers,
        timeout=config.timeout,
        insecure=_is_insecure(config),
    )


def _create_grpc_metric_exporter(config: "BrokleConfig") -> MetricExporter:
    """
    Create gRPC metric exporter.

    Requires: opentelemetry-exporter-otlp-proto-grpc package
    """
    try:
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter,
        )
    except ImportError as e:
        raise ImportError(
            "gRPC transport requires opentelemetry-exporter-otlp-proto-grpc. "
            "Install with: pip install opentelemetry-exporter-otlp-proto-grpc"
        ) from e

    endpoint = _get_grpc_endpoint(config)
    headers = _get_grpc_headers(config)

    logger.debug(f"Creating gRPC metric exporter for {endpoint}")

    return OTLPMetricExporter(
        endpoint=endpoint,
        headers=headers,
        timeout=config.timeout,
        insecure=_is_insecure(config),
    )


def _create_http_log_exporter(config: "BrokleConfig") -> LogExporter:
    """Create HTTP/Protobuf log exporter."""
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter

    endpoint = _get_http_endpoint(config, "logs")
    headers = _get_headers(config)

    logger.debug(f"Creating HTTP log exporter for {endpoint}")

    return OTLPLogExporter(
        endpoint=endpoint,
        headers=headers,
        timeout=config.timeout,
    )


def _create_grpc_log_exporter(config: "BrokleConfig") -> LogExporter:
    """
    Create gRPC log exporter.

    Requires: opentelemetry-exporter-otlp-proto-grpc package
    """
    try:
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
    except ImportError as e:
        raise ImportError(
            "gRPC transport requires opentelemetry-exporter-otlp-proto-grpc. "
            "Install with: pip install opentelemetry-exporter-otlp-proto-grpc"
        ) from e

    endpoint = _get_grpc_endpoint(config)
    headers = _get_grpc_headers(config)

    logger.debug(f"Creating gRPC log exporter for {endpoint}")

    return OTLPLogExporter(
        endpoint=endpoint,
        headers=headers,
        timeout=config.timeout,
        insecure=_is_insecure(config),
    )


def _get_http_endpoint(config: "BrokleConfig", signal: str) -> str:
    """
    Get HTTP endpoint for a specific signal type.

    Args:
        config: Brokle configuration
        signal: Signal type (traces, metrics, logs)

    Returns:
        Full endpoint URL
    """
    base_url = config.base_url.rstrip("/")
    return f"{base_url}/v1/{signal}"


def _get_grpc_endpoint(config: "BrokleConfig") -> str:
    """
    Get gRPC endpoint from configuration.

    gRPC uses host:port format without path.
    Default OTLP gRPC port is 4317.
    """
    # Check for explicit gRPC endpoint in config
    if hasattr(config, "grpc_endpoint") and config.grpc_endpoint:
        return config.grpc_endpoint

    # Extract host from base_url and use gRPC port
    base_url = config.base_url

    # Parse the base URL to extract host
    if base_url.startswith("https://"):
        host = base_url[8:].split("/")[0].split(":")[0]
        # For HTTPS, we might need secure connection
    elif base_url.startswith("http://"):
        host = base_url[7:].split("/")[0].split(":")[0]
    else:
        host = base_url.split("/")[0].split(":")[0]

    # Use standard OTLP gRPC port
    return f"{host}:4317"


def _get_headers(config: "BrokleConfig") -> Dict[str, str]:
    """Get HTTP headers for OTLP export."""
    headers = {
        "X-API-Key": config.api_key,
    }

    if config.environment and config.environment != "default":
        headers["X-Brokle-Environment"] = config.environment

    return headers


def _get_grpc_headers(config: "BrokleConfig") -> tuple:
    """
    Get gRPC metadata headers.

    gRPC uses tuple of (key, value) pairs for metadata.
    """
    headers = [
        ("x-api-key", config.api_key),
    ]

    if config.environment and config.environment != "default":
        headers.append(("x-brokle-environment", config.environment))

    return tuple(headers)


def _is_insecure(config: "BrokleConfig") -> bool:
    """
    Determine if gRPC connection should be insecure.

    Returns True for localhost/development, False for production.
    """
    base_url = config.base_url.lower()

    # Localhost is always insecure (no TLS)
    if "localhost" in base_url or "127.0.0.1" in base_url:
        return True

    # HTTP URLs are insecure
    if base_url.startswith("http://"):
        return True

    # HTTPS URLs should use TLS
    return False
