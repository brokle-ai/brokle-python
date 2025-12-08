"""
Brokle Metrics Module.

Provides OpenTelemetry metrics support for GenAI observability, including:
- Token usage histograms
- Operation duration metrics
- Request counters
- Custom bucket boundaries for LLM workloads

Example:
    >>> from brokle import Brokle
    >>> client = Brokle(api_key="bk_...", metrics_enabled=True)
    >>> # Metrics are automatically recorded by wrappers
    >>> # Access metrics directly:
    >>> metrics = client.get_metrics()
    >>> metrics.record_tokens(input_tokens=100, output_tokens=50, model="gpt-4")
"""

from .constants import (
    DURATION_BOUNDARIES,
    TOKEN_BOUNDARIES,
    TTFT_BOUNDARIES,
    MetricNames,
)
from .instruments import GenAIMetrics, create_genai_metrics
from .provider import BrokleMeterProvider, create_meter_provider

__all__ = [
    # Provider
    "create_meter_provider",
    "BrokleMeterProvider",
    # Instruments
    "GenAIMetrics",
    "create_genai_metrics",
    # Constants
    "MetricNames",
    "TOKEN_BOUNDARIES",
    "DURATION_BOUNDARIES",
    "TTFT_BOUNDARIES",
]
