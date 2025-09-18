"""
OpenAI-compatible client for Brokle Platform.

This module provides drop-in replacement for OpenAI SDK that routes requests
through Brokle Platform for intelligent routing, cost optimization, and observability.
"""

from .client import OpenAI, AsyncOpenAI
from .wrapper import create_openai_client, create_async_openai_client

# Create module-level instance
openai = create_openai_client()

__all__ = [
    "OpenAI",
    "AsyncOpenAI", 
    "openai",
    "create_openai_client",
    "create_async_openai_client",
]