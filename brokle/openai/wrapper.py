"""
Wrapper functions for creating OpenAI-compatible clients.
"""

from typing import Optional, Any
from .client import OpenAI, AsyncOpenAI


def create_openai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    project: Optional[str] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    **kwargs: Any
) -> OpenAI:
    """Create OpenAI-compatible client that routes through Brokle."""
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        organization=organization,
        project=project,
        timeout=timeout,
        max_retries=max_retries,
        **kwargs
    )


def create_async_openai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    project: Optional[str] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    **kwargs: Any
) -> AsyncOpenAI:
    """Create async OpenAI-compatible client that routes through Brokle."""
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        organization=organization,
        project=project,
        timeout=timeout,
        max_retries=max_retries,
        **kwargs
    )