"""
OpenAI-compatible client for Brokle Platform.

This module provides drop-in replacement for OpenAI SDK that routes requests
through Brokle Platform for intelligent routing, cost optimization, and observability.
"""

from .client import OpenAI, AsyncOpenAI, HAS_OPENAI
from .wrapper import create_openai_client, create_async_openai_client

# Create module-level instance only if OpenAI is available
openai = None
if HAS_OPENAI:
    try:
        openai = create_openai_client()
    except ImportError:
        pass

__all__ = [
    "OpenAI",
    "AsyncOpenAI",
    "openai",
    "create_openai_client",
    "create_async_openai_client",
]