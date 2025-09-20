"""
Wrapper functions for creating OpenAI-compatible clients.
"""

from .client import OpenAI, AsyncOpenAI, HAS_OPENAI
from ..client import get_client


def create_openai_client(**kwargs):
    """Create an OpenAI-compatible client instance."""
    if not HAS_OPENAI:
        raise ImportError("OpenAI package is required for OpenAI compatibility. Install with: pip install openai")

    client = get_client()
    return OpenAI(config=client.config.model_copy(), **kwargs)


def create_async_openai_client(**kwargs):
    """Create an async OpenAI-compatible client instance."""
    if not HAS_OPENAI:
        raise ImportError("OpenAI package is required for OpenAI compatibility. Install with: pip install openai")

    client = get_client()
    return AsyncOpenAI(config=client.config.model_copy(), **kwargs)
