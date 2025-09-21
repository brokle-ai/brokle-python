"""
OpenTelemetry span attributes for Brokle SDK.

Re-exports attributes from _client module for backward compatibility.
"""

# Import from the actual location
from .._client.attributes import BrokleOtelSpanAttributes

# Export for compatibility
__all__ = ["BrokleOtelSpanAttributes"]