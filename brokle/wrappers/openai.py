"""
OpenAI SDK wrapper for automatic observability.

Wraps OpenAI client to automatically create OTEL spans with GenAI 1.28+ attributes.
Streaming responses are transparently instrumented with TTFT and ITL tracking.

Uses the unified factory pattern for consistent sync/async behavior.
"""

from typing import TYPE_CHECKING, TypeVar, Union

from .._client import get_client
from ._extractors import extract_openai_response
from ._factory import create_wrapper
from ._provider_config import build_openai_attrs, openai_span_name

if TYPE_CHECKING:
    import openai

C = TypeVar("C")


def wrap_openai(
    client: Union["openai.OpenAI", "openai.AsyncOpenAI"],
) -> Union["openai.OpenAI", "openai.AsyncOpenAI"]:
    """
    Wrap OpenAI client for automatic observability.

    This function wraps the OpenAI client's chat.completions.create method
    to automatically create OTEL spans with GenAI semantic attributes.
    Works with both sync and async clients - auto-detects at runtime.

    Args:
        client: OpenAI or AsyncOpenAI client instance

    Returns:
        Wrapped client (same instance with instrumented methods)

    Example:
        >>> import openai
        >>> from brokle import get_client, wrap_openai
        >>>
        >>> # Initialize Brokle
        >>> brokle = get_client()
        >>>
        >>> # Wrap sync OpenAI client
        >>> client = wrap_openai(openai.OpenAI(api_key="..."))
        >>>
        >>> # Or wrap async client (same function!)
        >>> async_client = wrap_openai(openai.AsyncOpenAI(api_key="..."))
        >>>
        >>> # All calls automatically tracked with full attribute parity
        >>> response = client.chat.completions.create(
        ...     model="gpt-4",
        ...     messages=[{"role": "user", "content": "Hello"}],
        ...     seed=42,  # Captured in both sync AND async!
        ...     service_tier="auto",  # Captured in both sync AND async!
        ... )
        >>> brokle.flush()
    """
    # Return unwrapped if SDK disabled
    brokle_client = get_client()
    if not brokle_client.config.enabled:
        return client

    original_chat_create = client.chat.completions.create

    # Create wrapper using unified factory
    # This ensures sync and async paths use identical attribute extraction
    wrapped = create_wrapper(
        original_method=original_chat_create,
        build_attrs=build_openai_attrs,
        extract_response=extract_openai_response,
        get_span_name=openai_span_name,
        get_model=lambda kw: kw.get("model", "unknown"),
        is_stream=lambda kw: kw.get("stream", False),
    )

    client.chat.completions.create = wrapped
    return client


# Keep backward compatibility alias
wrap_openai_async = wrap_openai
