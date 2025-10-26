"""
OTLP exporter configuration for Brokle backend.

Configures OpenTelemetry's OTLP exporter with Brokle-specific authentication
and transport settings.
"""

from typing import Optional, Dict
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter, Compression
from opentelemetry.sdk.trace.export import SpanExporter

from .config import BrokleConfig


def create_brokle_exporter(
    config: BrokleConfig,
    additional_headers: Optional[Dict[str, str]] = None,
) -> SpanExporter:
    """
    Create OTLP span exporter configured for Brokle backend.

    This function creates a standard OpenTelemetry OTLP/HTTP exporter with
    Brokle-specific authentication headers and configuration.

    Args:
        config: Brokle configuration instance
        additional_headers: Optional additional HTTP headers

    Returns:
        Configured OTLPSpanExporter instance

    Example:
        >>> from brokle import BrokleConfig, create_brokle_exporter
        >>> config = BrokleConfig(api_key="bk_your_secret")
        >>> exporter = create_brokle_exporter(config)
    """
    # Build headers with authentication
    headers = config.get_headers()

    # Merge additional headers if provided
    if additional_headers:
        headers.update(additional_headers)

    # Get OTLP endpoint
    endpoint = config.get_otlp_endpoint()

    # Determine content type based on format
    if config.use_protobuf:
        # Protobuf format (default) - more efficient
        # Note: OpenTelemetry SDK will automatically use protobuf encoding
        # when OTLPSpanExporter is used without explicit content_type
        pass
    else:
        # JSON format (for debugging) - human readable
        headers["Content-Type"] = "application/json"

    # Disable compression for now to simplify backend handling
    # TODO: Add compression support after basic OTLP works
    compression = None

    # Create OTLP exporter with Brokle configuration
    exporter = OTLPSpanExporter(
        endpoint=endpoint,
        headers=headers,
        timeout=config.timeout,
        compression=compression,  # None for initial testing
    )

    return exporter


def create_console_exporter() -> SpanExporter:
    """
    Create console exporter for debugging.

    Useful for local development and debugging. Prints spans to stdout
    in a human-readable format.

    Returns:
        ConsoleSpanExporter instance

    Example:
        >>> from brokle import create_console_exporter
        >>> exporter = create_console_exporter()
        >>> # Use this exporter instead of OTLP for local debugging
    """
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter

    return ConsoleSpanExporter()


class NoOpExporter(SpanExporter):
    """
    No-op exporter that discards all spans.

    Used when tracing is disabled (config.tracing_enabled = False).
    This is more efficient than filtering spans before export.
    """

    def export(self, spans) -> "SpanExportResult":
        """Discard spans and return success."""
        from opentelemetry.sdk.trace.export import SpanExportResult
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """No-op shutdown."""
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """No-op flush."""
        return True


def create_exporter_for_config(config: BrokleConfig) -> SpanExporter:
    """
    Create appropriate exporter based on configuration.

    This is the recommended way to create an exporter as it handles
    different scenarios (disabled tracing, debug mode, etc.)

    Args:
        config: Brokle configuration instance

    Returns:
        Appropriate SpanExporter instance

    Example:
        >>> config = BrokleConfig.from_env()
        >>> exporter = create_exporter_for_config(config)
    """
    # If tracing is disabled, use no-op exporter
    if not config.tracing_enabled:
        return NoOpExporter()

    # If debug mode, optionally add console exporter
    # For now, we only use OTLP exporter
    # In the future, we could support composite exporters

    # Create standard OTLP exporter
    return create_brokle_exporter(config)
