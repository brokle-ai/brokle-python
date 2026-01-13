"""
Anthropic SDK wrapper for automatic observability.

Wraps Anthropic client to automatically create OTEL spans with GenAI 1.28+ attributes.
Streaming responses are transparently instrumented with TTFT and ITL tracking.

Uses the unified factory pattern for consistent sync/async behavior.
"""

from typing import TYPE_CHECKING, TypeVar, Union

from .._client import get_client
from ._extractors import extract_anthropic_response
from ._factory import create_wrapper
from ._provider_config import build_anthropic_attrs, anthropic_span_name

if TYPE_CHECKING:
    import anthropic

C = TypeVar("C")


def wrap_anthropic(
    client: Union["anthropic.Anthropic", "anthropic.AsyncAnthropic"],
) -> Union["anthropic.Anthropic", "anthropic.AsyncAnthropic"]:
    """
    Wrap Anthropic client for automatic observability.

    This function wraps the Anthropic client's messages.create method
    to automatically create OTEL spans with GenAI semantic attributes.
    Works with both sync and async clients - auto-detects at runtime.

    Args:
        client: Anthropic or AsyncAnthropic client instance

    Returns:
        Wrapped client (same instance with instrumented methods)

    Example:
        >>> import anthropic
        >>> from brokle import get_client, wrap_anthropic
        >>>
        >>> # Initialize Brokle
        >>> brokle = get_client()
        >>>
        >>> # Wrap sync Anthropic client
        >>> client = wrap_anthropic(anthropic.Anthropic(api_key="..."))
        >>>
        >>> # Or wrap async client (same function!)
        >>> async_client = wrap_anthropic(anthropic.AsyncAnthropic(api_key="..."))
        >>>
        >>> # All calls automatically tracked with full attribute parity
        >>> response = client.messages.create(
        ...     model="claude-3-opus",
        ...     max_tokens=1024,
        ...     messages=[{"role": "user", "content": "Hello"}],
        ...     top_k=40,  # Now captured in both sync AND async!
        ...     stop_sequences=["END"],  # Now captured in both sync AND async!
        ... )
        >>> brokle.flush()
    """
    # Return unwrapped if SDK disabled
    brokle_client = get_client()
    if not brokle_client.config.enabled:
        return client

    original_messages_create = client.messages.create

    # Create wrapper using unified factory
    # This ensures sync and async paths use identical attribute extraction
    wrapped = create_wrapper(
        original_method=original_messages_create,
        build_attrs=build_anthropic_attrs,
        extract_response=extract_anthropic_response,
        get_span_name=anthropic_span_name,
        get_model=lambda kw: kw.get("model", "unknown"),
        is_stream=lambda kw: kw.get("stream", False),
    )

    client.messages.create = wrapped
    return client


# Keep backward compatibility alias
wrap_anthropic_async = wrap_anthropic
