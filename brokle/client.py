"""
Main client for Brokle SDK.

This module provides backwards compatibility with the existing client interface
while delegating to the new LangFuse-inspired OTEL client.
"""

# Import the new OTEL-based client
from ._client import Brokle, get_client

# Export for backwards compatibility
__all__ = ["Brokle", "get_client"]