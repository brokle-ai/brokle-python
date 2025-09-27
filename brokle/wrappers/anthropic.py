"""
Anthropic Wrapper Function - Pattern 1 Implementation

Provides wrap_anthropic() function for explicit client wrapping.
Wraps existing Anthropic client instances with Brokle observability.
"""

import logging
import warnings
from typing import Optional, TypeVar, Union, cast

try:
    import anthropic
    from anthropic import Anthropic as _Anthropic
    from anthropic import AsyncAnthropic as _AsyncAnthropic

    HAS_ANTHROPIC = True
except ImportError:
    anthropic = None
    _Anthropic = None
    _AsyncAnthropic = None
    HAS_ANTHROPIC = False

from .._utils.validation import validate_environment
from ..exceptions import ProviderError, ValidationError
from ..integrations.instrumentation import UniversalInstrumentation
from ..observability import get_client
from ..providers import get_provider

logger = logging.getLogger(__name__)

# Type variables for maintaining client types
AnthropicType = TypeVar("AnthropicType", bound=Union[_Anthropic, _AsyncAnthropic])


def wrap_anthropic(
    client: AnthropicType,
    *,
    capture_content: bool = True,
    capture_metadata: bool = True,
    tags: Optional[list] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    **config,
) -> AnthropicType:
    """
    Wrap Anthropic client with Brokle observability.

    Args:
        client: Anthropic or AsyncAnthropic client instance
        capture_content: Whether to capture request/response content (default: True)
        capture_metadata: Whether to capture metadata like model, tokens (default: True)
        tags: List of tags to add to all traces from this client
        session_id: Session identifier for grouping related calls
        user_id: User identifier for user-scoped analytics
        **config: Additional Brokle configuration options

    Returns:
        Wrapped client with identical interface but comprehensive observability

    Raises:
        ProviderError: If Anthropic SDK not installed or client is invalid
        ValidationError: If configuration is invalid

    Example:
        ```python
        from anthropic import Anthropic
        from brokle import wrap_anthropic

        # Basic usage
        client = wrap_anthropic(Anthropic(api_key="sk-ant-..."))

        # With configuration
        client = wrap_anthropic(
            Anthropic(),
            capture_content=True,
            tags=["production", "assistant"],
            session_id="session_123",
            user_id="user_456"
        )

        # Use exactly like normal Anthropic client
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": "Hello"}]
        )

        # Async client support
        async_client = wrap_anthropic(AsyncAnthropic())
        response = await async_client.messages.create(...)
        ```

    Performance:
        - <3ms overhead per request
        - No blocking operations
        - Background telemetry processing

    Compatibility:
        - Works with Anthropic SDK v0.5.0+
        - Supports all Anthropic client methods
        - Maintains exact same API interface
        - Works with streaming responses
    """
    # Validate dependencies
    if not HAS_ANTHROPIC:
        raise ProviderError(
            "Anthropic SDK not installed. Install with: pip install anthropic>=0.5.0\n"
            "Or install Brokle with Anthropic support: pip install brokle[anthropic]"
        )

    # Validate client type (only if SDK is installed)
    if HAS_ANTHROPIC and _Anthropic and _AsyncAnthropic:
        try:
            if not isinstance(client, (_Anthropic, _AsyncAnthropic)):
                raise ProviderError(
                    f"Expected Anthropic or AsyncAnthropic client, got {type(client).__name__}.\n"
                    f"Usage: client = wrap_anthropic(Anthropic())"
                )
        except TypeError:
            # Skip validation during testing when _Anthropic/_AsyncAnthropic are mocked
            pass

    # Check if already wrapped
    if hasattr(client, "_brokle_instrumented") and client._brokle_instrumented:
        warnings.warn(
            "Anthropic client is already wrapped with Brokle. "
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
        provider = get_provider("anthropic", **provider_config)
        instrumentation = UniversalInstrumentation(provider)

        # Apply instrumentation to client
        wrapped_client = instrumentation.instrument_client(client)

        # Add wrapper metadata
        setattr(wrapped_client, "_brokle_instrumented", True)
        setattr(wrapped_client, "_brokle_provider", "anthropic")
        setattr(wrapped_client, "_brokle_config", provider_config)
        setattr(wrapped_client, "_brokle_wrapper_version", "2.0.0")

        logger.info(
            f"Anthropic client successfully wrapped with Brokle observability. "
            f"Provider: {provider.name}, Capture content: {capture_content}"
        )

        return cast(AnthropicType, wrapped_client)

    except Exception as e:
        logger.error(f"Failed to wrap Anthropic client: {e}")
        raise ProviderError(f"Failed to instrument Anthropic client: {e}")


# Export public API
__all__ = [
    "wrap_anthropic",
]
