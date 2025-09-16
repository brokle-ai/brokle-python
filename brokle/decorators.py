"""
Decorators for Brokle SDK observability.

This module provides the @observe decorator following the LangFuse pattern
but adapted for Brokle's specific features.
"""

# Import the new OTEL-based observe decorator
from ._client.observe import observe

# Export for backwards compatibility
__all__ = ["observe"]