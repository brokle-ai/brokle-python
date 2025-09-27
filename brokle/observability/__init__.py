"""
Brokle Observability Module - Stable Public API

This module provides a stable interface for Pattern 1 (wrappers) and Pattern 2 (decorator)
to access observability features. The public API is locked down to ensure compatibility.

**CRITICAL**: This API must remain stable - Pattern 1/2 depend on these imports.
"""

from .attributes import BrokleOtelSpanAttributes
from .config import ObservabilityConfig, get_config, telemetry_enabled
from .context import clear_context, get_client, get_client_context, get_context_info
from .spans import (
    BrokleGeneration,
    BrokleSpan,
    create_span,
    get_current_span,
    record_span,
    span_context,
)

# Stable public API - DO NOT CHANGE without migration plan
__all__ = [
    # Configuration
    "get_config",
    "telemetry_enabled",
    "ObservabilityConfig",
    # Context management
    "get_client",
    "get_client_context",
    "clear_context",
    "get_context_info",
    # Span management
    "create_span",
    "record_span",
    "get_current_span",
    "span_context",
    "BrokleSpan",
    "BrokleGeneration",
    # Attributes
    "BrokleOtelSpanAttributes",
]
