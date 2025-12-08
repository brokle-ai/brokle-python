"""
Brokle Transport Module.

Provides transport layer abstractions for OTLP export,
supporting both HTTP and gRPC protocols.

The backend supports both:
- HTTP: /v1/traces, /v1/metrics, /v1/logs
- gRPC: Port 4317 (standard OTLP gRPC)

Example:
    >>> from brokle.transport import create_trace_exporter
    >>> # HTTP transport (default)
    >>> exporter = create_trace_exporter(config, transport="http")
    >>> # gRPC transport (requires opentelemetry-exporter-otlp-proto-grpc)
    >>> exporter = create_trace_exporter(config, transport="grpc")
"""

from .factory import (
    TransportType,
    create_log_exporter,
    create_metric_exporter,
    create_trace_exporter,
)

__all__ = [
    "create_trace_exporter",
    "create_metric_exporter",
    "create_log_exporter",
    "TransportType",
]
