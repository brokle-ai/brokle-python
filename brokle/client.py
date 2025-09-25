"""
Main client for Brokle SDK v2.0.

This module provides the main client interface with OpenAI-compatible design
and clean resource management.

v2.0 Changes:
- Clean OpenAI-compatible interface
- Proper resource management with context managers
- Sync and async client variants
- Brokle extensions (routing, caching, tags)
"""

# Import new architecture clients
from .new_client import Brokle, AsyncBrokle, get_client

# Export public API
__all__ = ["Brokle", "AsyncBrokle", "get_client"]