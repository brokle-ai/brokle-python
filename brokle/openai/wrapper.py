"""
Wrapper functions for creating OpenAI-compatible clients.
"""

from .client import OpenAI, AsyncOpenAI, HAS_OPENAI
from ..config import get_config


def create_openai_client(**kwargs):
    """Create an OpenAI-compatible client instance."""
    if not HAS_OPENAI:
        raise ImportError("OpenAI package is required for OpenAI compatibility. Install with: pip install openai")

    config = get_config()
    return OpenAI(config=config, **kwargs)


def create_async_openai_client(**kwargs):
    """Create an async OpenAI-compatible client instance."""
    if not HAS_OPENAI:
        raise ImportError("OpenAI package is required for OpenAI compatibility. Install with: pip install openai")

    config = get_config()
    return AsyncOpenAI(config=config, **kwargs)