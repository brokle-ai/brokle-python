"""
MeterProvider setup for Brokle metrics.

Configures OpenTelemetry MeterProvider with OTLP HTTP exporter and
custom metric views for GenAI-optimized bucket boundaries.
"""

import logging
from typing import TYPE_CHECKING, Optional

from opentelemetry.sdk.metrics import Meter, MeterProvider
from opentelemetry.sdk.metrics.export import (
    MetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
from opentelemetry.sdk.resources import Resource

from ..types import SchemaURLs
from .constants import (
    DURATION_BOUNDARIES,
    INTER_TOKEN_BOUNDARIES,
    TOKEN_BOUNDARIES,
    TTFT_BOUNDARIES,
    MetricNames,
)

if TYPE_CHECKING:
    from ..config import BrokleConfig

logger = logging.getLogger(__name__)


def _create_metric_views() -> list:
    """
    Create metric views with custom bucket boundaries.

    Views define how metrics are aggregated. We use custom bucket
    boundaries optimized for LLM workloads rather than OTEL defaults.

    Returns:
        List of View objects for metric aggregation
    """
    return [
        View(
            instrument_name=MetricNames.TOKEN_USAGE,
            aggregation=ExplicitBucketHistogramAggregation(TOKEN_BOUNDARIES),
        ),
        View(
            instrument_name=MetricNames.OPERATION_DURATION,
            aggregation=ExplicitBucketHistogramAggregation(DURATION_BOUNDARIES),
        ),
        View(
            instrument_name=MetricNames.TIME_TO_FIRST_TOKEN,
            aggregation=ExplicitBucketHistogramAggregation(TTFT_BOUNDARIES),
        ),
        View(
            instrument_name=MetricNames.INTER_TOKEN_LATENCY,
            aggregation=ExplicitBucketHistogramAggregation(INTER_TOKEN_BOUNDARIES),
        ),
    ]


def _create_otlp_metric_exporter(config: "BrokleConfig") -> MetricExporter:
    """
    Create OTLP metric exporter based on transport configuration.

    Respects config.transport to use either HTTP or gRPC transport,
    ensuring consistency with trace exporting.

    Args:
        config: Brokle configuration

    Returns:
        Configured metric exporter (HTTP or gRPC based on config.transport)
    """
    from ..transport import TransportType, create_metric_exporter

    transport = TransportType.GRPC if config.transport == "grpc" else TransportType.HTTP
    return create_metric_exporter(config, transport)


def create_meter_provider(
    config: "BrokleConfig",
    resource: Optional[Resource] = None,
) -> MeterProvider:
    """
    Create MeterProvider with OTLP exporter and GenAI-optimized views.

    This is the main factory function for setting up metrics collection.
    It configures:
    - OTLP HTTP exporter for sending metrics to Brokle backend
    - Periodic reader for batched metric export
    - Custom views with LLM-optimized bucket boundaries

    Args:
        config: Brokle configuration
        resource: Optional Resource to associate with metrics

    Returns:
        Configured MeterProvider

    Example:
        >>> from brokle.config import BrokleConfig
        >>> config = BrokleConfig(api_key="bk_...")
        >>> provider = create_meter_provider(config)
        >>> meter = provider.get_meter("brokle")
    """
    try:
        exporter = _create_otlp_metric_exporter(config)
    except ImportError as e:
        logger.warning(
            f"OTLP metric exporter not available: {e}. "
            "Install opentelemetry-exporter-otlp-proto-http for metrics support."
        )
        raise

    export_interval_millis = int(
        getattr(config, "metrics_export_interval", 60.0) * 1000
    )

    reader = PeriodicExportingMetricReader(
        exporter=exporter,
        export_interval_millis=export_interval_millis,
    )

    if resource is None:
        resource = Resource.create({}, schema_url=SchemaURLs.DEFAULT)

    provider = MeterProvider(
        resource=resource,
        metric_readers=[reader],
        views=_create_metric_views(),
    )

    logger.debug(
        f"Created MeterProvider with export interval {export_interval_millis}ms"
    )

    return provider


class BrokleMeterProvider:
    """
    Wrapper around MeterProvider with Brokle-specific configuration.

    Provides a simplified interface for creating and managing meters
    with automatic configuration from BrokleConfig.

    Example:
        >>> provider = BrokleMeterProvider(config)
        >>> meter = provider.get_meter()
        >>> histogram = meter.create_histogram("my.metric")
    """

    def __init__(
        self,
        config: "BrokleConfig",
        resource: Optional[Resource] = None,
    ):
        """
        Initialize Brokle meter provider.

        Args:
            config: Brokle configuration
            resource: Optional Resource (shares with TracerProvider)
        """
        self._config = config
        self._provider = create_meter_provider(config, resource)
        self._meter: Optional[Meter] = None

    def get_meter(self, name: str = "brokle", version: Optional[str] = None) -> Meter:
        """
        Get a Meter instance.

        Args:
            name: Meter name (instrumentation scope)
            version: Meter version

        Returns:
            Meter instance
        """
        if version is None:
            try:
                from .. import __version__

                version = __version__
            except (ImportError, AttributeError):
                version = "0.1.0-dev"

        return self._provider.get_meter(
            name=name,
            version=version,
            schema_url=SchemaURLs.DEFAULT,
        )

    def shutdown(self, timeout_millis: int = 30000) -> bool:
        """
        Shutdown the meter provider.

        Args:
            timeout_millis: Timeout in milliseconds

        Returns:
            True if successful, False on error
        """
        # MeterProvider.shutdown() returns None on success, raises on failure
        try:
            self._provider.shutdown(timeout_millis)
            return True
        except Exception:
            return False

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """
        Force flush all pending metrics.

        Args:
            timeout_millis: Timeout in milliseconds

        Returns:
            True if successful
        """
        return self._provider.force_flush(timeout_millis)

    @property
    def provider(self) -> MeterProvider:
        """Access the underlying MeterProvider."""
        return self._provider
