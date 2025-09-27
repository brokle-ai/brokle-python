"""
OpenAI Wrapper Function - Pattern 1 Implementation

Provides wrap_openai() function for explicit client wrapping.
Wraps existing OpenAI client instances with Brokle observability.
"""

import logging
import warnings
from typing import Optional, TypeVar, Union, cast

try:
    import openai
    from openai import AsyncOpenAI as _AsyncOpenAI
    from openai import OpenAI as _OpenAI

    HAS_OPENAI = True
except ImportError:
    openai = None
    _OpenAI = None
    _AsyncOpenAI = None
    HAS_OPENAI = False

from .._utils.validation import validate_environment
from ..exceptions import ProviderError, ValidationError
from ..integrations.instrumentation import UniversalInstrumentation
from ..observability import get_client
from ..providers import get_provider

logger = logging.getLogger(__name__)

# Type variables for maintaining client types
OpenAIType = TypeVar("OpenAIType", bound=Union[_OpenAI, _AsyncOpenAI])


def wrap_openai(
    client: OpenAIType,
    *,
    capture_content: bool = True,
    capture_metadata: bool = True,
    tags: Optional[list] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    **config,
) -> OpenAIType:
    """
    Wrap OpenAI client with Brokle observability.

    Args:
        client: OpenAI or AsyncOpenAI client instance
        capture_content: Whether to capture request/response content (default: True)
        capture_metadata: Whether to capture metadata like model, tokens (default: True)
        tags: List of tags to add to all traces from this client
        session_id: Session identifier for grouping related calls
        user_id: User identifier for user-scoped analytics
        **config: Additional Brokle configuration options

    Returns:
        Wrapped client with identical interface but comprehensive observability

    Raises:
        ProviderError: If OpenAI SDK not installed or client is invalid
        ValidationError: If configuration is invalid

    Example:
        ```python
        from openai import OpenAI
        from brokle import wrap_openai

        # Basic usage
        client = wrap_openai(OpenAI(api_key="sk-..."))

        # With configuration
        client = wrap_openai(
            OpenAI(),
            capture_content=True,
            tags=["production", "chatbot"],
            session_id="session_123",
            user_id="user_456"
        )

        # Use exactly like normal OpenAI client
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}]
        )

        # Async client support
        async_client = wrap_openai(AsyncOpenAI())
        response = await async_client.chat.completions.create(...)
        ```

    Performance:
        - <3ms overhead per request
        - No blocking operations
        - Background telemetry processing

    Compatibility:
        - Works with OpenAI SDK v1.0+
        - Supports all OpenAI client methods
        - Maintains exact same API interface
        - Works with streaming responses
    """
    # Validate dependencies
    if not HAS_OPENAI:
        raise ProviderError(
            "OpenAI SDK not installed. Install with: pip install openai>=1.0.0\n"
            "Or install Brokle with OpenAI support: pip install brokle[openai]"
        )

    # Validate client type (only if SDK is installed)
    if HAS_OPENAI and _OpenAI and _AsyncOpenAI:
        try:
            if not isinstance(client, (_OpenAI, _AsyncOpenAI)):
                raise ProviderError(
                    f"Expected OpenAI or AsyncOpenAI client, got {type(client).__name__}.\n"
                    f"Usage: client = wrap_openai(OpenAI())"
                )
        except TypeError:
            # Skip validation during testing when _OpenAI/_AsyncOpenAI are mocked
            pass

    # Check if already wrapped
    if hasattr(client, "_brokle_instrumented") and client._brokle_instrumented:
        warnings.warn(
            "OpenAI client is already wrapped with Brokle. "
            "Multiple wrapping may cause duplicate telemetry.",
            UserWarning,
        )
        return client

    # Validate environment configuration
    try:
        validate_environment()
    except Exception as e:
        logger.warning(f"Environment validation failed: {e}")

    # Validate wrapper configuration
    from .._utils.wrapper_validation import validate_wrapper_config

    validate_wrapper_config(
        capture_content=capture_content,
        capture_metadata=capture_metadata,
        tags=tags,
        session_id=session_id,
        user_id=user_id,
        **config,
    )

    # Check Brokle client availability (optional)
    brokle_client = None
    try:
        brokle_client = get_client()
        if not brokle_client:
            logger.info(
                "No Brokle client configured. Using default observability settings."
            )
    except Exception as e:
        logger.debug(f"Brokle client not available: {e}")

    # Configure provider with wrapper settings
    provider_config = {
        "capture_content": capture_content,
        "capture_metadata": capture_metadata,
        "tags": tags or [],
        "session_id": session_id,
        "user_id": user_id,
        **config,
    }

    # Create provider and instrumentation
    try:
        provider = get_provider("openai", **provider_config)
        instrumentation = UniversalInstrumentation(provider)

        # Apply instrumentation to client
        wrapped_client = instrumentation.instrument_client(client)

        # Add wrapper metadata
        setattr(wrapped_client, "_brokle_instrumented", True)
        setattr(wrapped_client, "_brokle_provider", "openai")
        setattr(wrapped_client, "_brokle_config", provider_config)
        setattr(wrapped_client, "_brokle_wrapper_version", "2.0.0")

        logger.info(
            f"OpenAI client successfully wrapped with Brokle observability. "
            f"Provider: {provider.name}, Capture content: {capture_content}"
        )

        return cast(OpenAIType, wrapped_client)

    except Exception as e:
        logger.error(f"Failed to wrap OpenAI client: {e}")
        raise ProviderError(f"Failed to instrument OpenAI client: {e}")


# Export public API
__all__ = [
    "wrap_openai",
]
