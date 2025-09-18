"""
Decorators for Brokle SDK observability.
"""

# Import the new OTEL-based observe decorator
from ._client.observe import observe

# Export for backwards compatibility
__all__ = ["observe"]