"""
LoggerProvider setup for Brokle logs/events.

Configures OpenTelemetry LoggerProvider with OTLP HTTP/gRPC exporter
for structured log emission and event tracking.
"""

import logging
from typing import TYPE_CHECKING, Optional

from opentelemetry.sdk._logs import Logger, LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource

from ..types import SchemaURLs

if TYPE_CHECKING:
    from ..config import BrokleConfig

logger = logging.getLogger(__name__)


def _create_otlp_log_exporter(config: "BrokleConfig"):
    """
    Create OTLP log exporter based on transport configuration.

    Respects config.transport to use either HTTP or gRPC transport,
    ensuring consistency with trace and metric exporting.

    Args:
        config: Brokle configuration

    Returns:
        Configured log exporter (HTTP or gRPC based on config.transport)
    """
    from ..transport import TransportType, create_log_exporter

    transport = TransportType.GRPC if config.transport == "grpc" else TransportType.HTTP
    return create_log_exporter(config, transport)


def create_logger_provider(
    config: "BrokleConfig",
    resource: Optional[Resource] = None,
) -> LoggerProvider:
    """
    Create LoggerProvider with OTLP exporter.

    This is the main factory function for setting up log collection.
    It configures:
    - OTLP HTTP/gRPC exporter for sending logs to Brokle backend
    - Batch processor for efficient log export

    Args:
        config: Brokle configuration
        resource: Optional Resource to associate with logs

    Returns:
        Configured LoggerProvider

    Example:
        >>> from brokle.config import BrokleConfig
        >>> config = BrokleConfig(api_key="bk_...")
        >>> provider = create_logger_provider(config)
        >>> logger = provider.get_logger("brokle")
    """
    try:
        exporter = _create_otlp_log_exporter(config)
    except ImportError as e:
        logger.warning(
            f"OTLP log exporter not available: {e}. "
            "Install opentelemetry-exporter-otlp-proto-http for logs support."
        )
        raise

    processor = BatchLogRecordProcessor(exporter)

    if resource is None:
        resource = Resource.create({}, schema_url=SchemaURLs.DEFAULT)

    provider = LoggerProvider(resource=resource)
    provider.add_log_record_processor(processor)

    logger.debug("Created LoggerProvider with BatchLogRecordProcessor")

    return provider


class BrokleLoggerProvider:
    """
    Wrapper around LoggerProvider with Brokle-specific configuration.

    Provides a simplified interface for creating and managing loggers
    with automatic configuration from BrokleConfig.

    Example:
        >>> provider = BrokleLoggerProvider(config)
        >>> logger = provider.get_logger()
    """

    def __init__(
        self,
        config: "BrokleConfig",
        resource: Optional[Resource] = None,
    ):
        """
        Initialize Brokle logger provider.

        Args:
            config: Brokle configuration
            resource: Optional Resource (shares with TracerProvider/MeterProvider)
        """
        self._config = config
        self._provider = create_logger_provider(config, resource)
        self._logger: Optional[Logger] = None

    def get_logger(self, name: str = "brokle", version: Optional[str] = None) -> Logger:
        """
        Get a Logger instance.

        Args:
            name: Logger name (instrumentation scope)
            version: Logger version

        Returns:
            Logger instance
        """
        if version is None:
            try:
                from .. import __version__

                version = __version__
            except (ImportError, AttributeError):
                version = "0.1.0-dev"

        return self._provider.get_logger(
            name=name,
            version=version,
            schema_url=SchemaURLs.DEFAULT,
        )

    def shutdown(self, timeout_millis: int = 30000) -> bool:
        """
        Shutdown the logger provider.

        Args:
            timeout_millis: Timeout in milliseconds

        Returns:
            True if successful, False on error
        """
        try:
            self._provider.shutdown()
            return True
        except Exception:
            return False

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """
        Force flush all pending logs.

        Args:
            timeout_millis: Timeout in milliseconds

        Returns:
            True if successful
        """
        return self._provider.force_flush(timeout_millis)

    @property
    def provider(self) -> LoggerProvider:
        """Access the underlying LoggerProvider."""
        return self._provider
