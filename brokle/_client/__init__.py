"""
Brokle client implementation following LangFuse patterns.

This module provides the core client functionality for the Brokle SDK,
including OTEL integration and span management.
"""

from .client import Brokle, get_client
from .observe import observe
from .span import BrokleSpan, BrokleGeneration
from .attributes import BrokleOtelSpanAttributes

__all__ = [
    "Brokle",
    "get_client",
    "observe",
    "BrokleSpan",
    "BrokleGeneration",
    "BrokleOtelSpanAttributes",
]