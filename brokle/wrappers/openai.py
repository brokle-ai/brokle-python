"""
OpenAI Wrapper Function - Pattern 1 Implementation

Provides wrap_openai() function following LangSmith/Optik patterns.
Wraps existing OpenAI client instances with Brokle observability.
"""

import logging
from typing import Any, Optional, TypeVar, cast, Union
import time
import warnings

try:
    import openai
    from openai import OpenAI as _OpenAI, AsyncOpenAI as _AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    openai = None
    _OpenAI = None
    _AsyncOpenAI = None
    HAS_OPENAI = False

from ..integrations.instrumentation import UniversalInstrumentation
from ..integrations.providers.openai import OpenAIProvider
from ..exceptions import ProviderError, ValidationError
from .._utils.validation import validate_environment
from .._utils.wrapper_validation import validate_wrapper_config
from ..client import get_client

logger = logging.getLogger(__name__)

# Type variables for maintaining client types
OpenAIType = TypeVar('OpenAIType', bound=Union[_OpenAI, _AsyncOpenAI])


def wrap_openai(
    client: OpenAIType,
    *,
    capture_content: bool = True,
    capture_metadata: bool = True,
    tags: Optional[list] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    **config
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
        Enhanced client with identical interface but comprehensive observability

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
    if hasattr(client, '_brokle_instrumented') and client._brokle_instrumented:
        warnings.warn(
            "OpenAI client is already wrapped with Brokle. "
            "Multiple wrapping may cause duplicate telemetry.",
            UserWarning
        )
        return client

    # Validate environment configuration
    try:
        validate_environment()
    except Exception as e:
        logger.warning(f"Environment validation failed: {e}")

    # Validate configuration parameters
    try:
        validate_wrapper_config(
            capture_content=capture_content,
            capture_metadata=capture_metadata,
            tags=tags,
            session_id=session_id,
            user_id=user_id,
            **config
        )
    except Exception as e:
        raise ValidationError(f"Invalid wrapper configuration: {e}")

    # Check Brokle client availability (optional)
    brokle_client = None
    try:
        brokle_client = get_client()
        if not brokle_client:
            logger.info("No Brokle client configured. Using default observability settings.")
    except Exception as e:
        logger.debug(f"Brokle client not available: {e}")

    # Configure provider with wrapper settings
    provider_config = {
        'capture_content': capture_content,
        'capture_metadata': capture_metadata,
        'tags': tags or [],
        'session_id': session_id,
        'user_id': user_id,
        **config
    }

    # Create provider and instrumentation
    try:
        provider = OpenAIProvider(**provider_config)
        instrumentation = UniversalInstrumentation(provider)

        # Apply instrumentation to client
        enhanced_client = instrumentation.instrument_client(client)

        # Add wrapper metadata
        setattr(enhanced_client, '_brokle_instrumented', True)
        setattr(enhanced_client, '_brokle_provider', 'openai')
        setattr(enhanced_client, '_brokle_config', provider_config)
        setattr(enhanced_client, '_brokle_wrapper_version', '2.0.0')

        logger.info(
            f"OpenAI client successfully wrapped with Brokle observability. "
            f"Provider: {provider.name}, Capture content: {capture_content}"
        )

        return cast(OpenAIType, enhanced_client)

    except Exception as e:
        logger.error(f"Failed to wrap OpenAI client: {e}")
        raise ProviderError(f"Failed to instrument OpenAI client: {e}")


# Convenience function for Optik-style usage
def track_openai(
    *,
    capture_content: bool = True,
    capture_metadata: bool = True,
    tags: Optional[list] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    **config
):
    """
    Function-based OpenAI tracking following Optik patterns.

    Returns a context manager that automatically wraps
    OpenAI clients found in the execution context.

    Usage:
        # As context manager
        with track_openai(tags=["production"]):
            client = OpenAI()
            response = client.chat.completions.create(...)

        # As decorator (future implementation)
        @track_openai(session_id="session_123")
        def my_ai_function():
            client = OpenAI()
            return client.chat.completions.create(...)
    """
    # This is a simplified implementation
    # Full implementation would require more sophisticated client detection
    class OpenAITracker:
        def __init__(self, **tracking_config):
            self.config = tracking_config
            self._original_openai = None
            self._original_async_openai = None

        def __enter__(self):
            if not HAS_OPENAI:
                logger.warning("OpenAI SDK not available for tracking")
                return self

            # Store original OpenAI classes
            self._original_openai = _OpenAI
            self._original_async_openai = _AsyncOpenAI

            # Create wrapper classes that auto-instrument
            class BrokleOpenAI(_OpenAI):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    # Auto-wrap after initialization
                    wrapped = wrap_openai(self, **tracking_config)
                    # Copy wrapped attributes back to self
                    self.__dict__.update(wrapped.__dict__)

            class BrokleAsyncOpenAI(_AsyncOpenAI):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    # Auto-wrap after initialization
                    wrapped = wrap_openai(self, **tracking_config)
                    # Copy wrapped attributes back to self
                    self.__dict__.update(wrapped.__dict__)

            # Monkey patch OpenAI classes
            openai.OpenAI = BrokleOpenAI
            openai.AsyncOpenAI = BrokleAsyncOpenAI

            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if not HAS_OPENAI:
                return

            # Restore original classes
            openai.OpenAI = self._original_openai
            openai.AsyncOpenAI = self._original_async_openai

    return OpenAITracker(
        capture_content=capture_content,
        capture_metadata=capture_metadata,
        tags=tags,
        session_id=session_id,
        user_id=user_id,
        **config
    )


# Export public API
__all__ = [
    "wrap_openai",
    "track_openai",
]