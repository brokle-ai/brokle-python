"""
OpenTelemetry span attributes for Brokle SDK.

Re-exports attributes from observability module for backward compatibility.
"""

# Import from the new observability module
from ..observability.attributes import BrokleOtelSpanAttributes

# Export for compatibility
__all__ = ["BrokleOtelSpanAttributes"]
