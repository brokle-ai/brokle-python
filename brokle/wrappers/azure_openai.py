"""
Azure OpenAI SDK wrapper for automatic observability.

Wraps Azure OpenAI client to automatically create OTEL spans with GenAI 1.28+ attributes.
Extends the OpenAI wrapper pattern with Azure-specific attributes.

Uses the unified factory pattern for consistent sync/async behavior.
"""

from functools import partial
from typing import TYPE_CHECKING, TypeVar, Union

from ._extractors import extract_openai_response
from ._factory import create_wrapper
from ._provider_config import build_azure_openai_attrs, azure_openai_span_name

if TYPE_CHECKING:
    from openai import AzureOpenAI, AsyncAzureOpenAI

C = TypeVar("C")


def wrap_azure_openai(
    client: Union["AzureOpenAI", "AsyncAzureOpenAI"],
) -> Union["AzureOpenAI", "AsyncAzureOpenAI"]:
    """
    Wrap Azure OpenAI client for automatic observability.

    This function wraps the Azure OpenAI client's chat.completions.create method
    to automatically create OTEL spans with GenAI semantic attributes.
    Works with both sync and async clients - auto-detects at runtime.

    Args:
        client: AzureOpenAI or AsyncAzureOpenAI client instance

    Returns:
        Wrapped client (same instance with instrumented methods)

    Example:
        >>> from openai import AzureOpenAI
        >>> from brokle import get_client, wrap_azure_openai
        >>>
        >>> # Initialize Brokle
        >>> brokle = get_client()
        >>>
        >>> # Wrap Azure OpenAI client
        >>> client = wrap_azure_openai(AzureOpenAI(
        ...     azure_endpoint="https://YOUR_RESOURCE.openai.azure.com",
        ...     api_key="...",
        ...     api_version="2024-02-15-preview"
        ... ))
        >>>
        >>> # Or wrap async client (same function!)
        >>> async_client = wrap_azure_openai(AsyncAzureOpenAI(...))
        >>>
        >>> # All calls automatically tracked with Azure-specific attributes
        >>> response = client.chat.completions.create(
        ...     model="gpt-4",  # Your deployment name
        ...     messages=[{"role": "user", "content": "Hello"}]
        ... )
        >>> brokle.flush()
    """
    # Extract Azure-specific metadata from client
    azure_endpoint = getattr(client, "_azure_endpoint", None)
    api_version = getattr(client, "_api_version", None)

    original_chat_create = client.chat.completions.create

    # Create Azure-specific attribute builder with captured endpoint/version
    def build_attrs_with_azure(kwargs):
        return build_azure_openai_attrs(
            kwargs,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )

    # Create wrapper using unified factory
    # This ensures sync and async paths use identical attribute extraction
    wrapped = create_wrapper(
        original_method=original_chat_create,
        build_attrs=build_attrs_with_azure,
        extract_response=extract_openai_response,
        get_span_name=azure_openai_span_name,
        get_model=lambda kw: kw.get("model", "unknown"),
        is_stream=lambda kw: kw.get("stream", False),
    )

    client.chat.completions.create = wrapped
    return client


# Keep backward compatibility alias
wrap_azure_openai_async = wrap_azure_openai
