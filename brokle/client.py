"""
Main client for Brokle SDK.

This module provides the main client interface with context-aware management
following industry-standard patterns for multi-project safety and thread isolation.
"""

# Import the OTEL-based client class
from ._client import Brokle

# Import context-aware client management
from ._client.context import get_client

# Export public API
__all__ = ["Brokle", "get_client"]