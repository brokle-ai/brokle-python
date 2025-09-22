"""
Pattern 1: Drop-in Replacement - Brokle SDK

Perfect drop-in replacement for OpenAI SDK with instant observability.
Part of Brokle's 3-pattern integration system.

Pattern 1 Features:
- Zero code changes beyond import
- 100% OpenAI SDK compatibility
- Instant observability and cost tracking
- <3ms performance overhead
- Perfect for existing codebases

Usage:
    # Pattern 1: Drop-in Replacement
    from brokle.openai import OpenAI  # Only change needed

    client = OpenAI()  # Everything else identical
    response = client.chat.completions.create(...)

    # Ready to upgrade to Pattern 2 (@observe) or Pattern 3 (Native SDK)
"""

import warnings
import logging
from typing import Any, Optional, Dict, List
import time

try:
    import wrapt
except ImportError:
    wrapt = None
    warnings.warn("wrapt not available - OpenAI instrumentation disabled", UserWarning)

try:
    import openai as _original_openai
    from openai import OpenAI as _OriginalOpenAI
    from openai.types.chat import ChatCompletion
    from openai.types import Completion
    HAS_OPENAI = True
except ImportError:
    _original_openai = None
    _OriginalOpenAI = None
    ChatCompletion = None
    Completion = None
    HAS_OPENAI = False
    warnings.warn(
        "OpenAI SDK not found. Install with: pip install openai>=1.0.0",
        UserWarning
    )

from ..client import get_client
from .._utils.telemetry import create_span, add_span_attributes, record_span_exception
from .._utils.error_handling import handle_provider_error
from .._utils.validation import validate_environment
from ..exceptions import ProviderError
from ..types.attributes import BrokleOtelSpanAttributes

logger = logging.getLogger(__name__)


class BrokleOpenAISpanManager:
    """Manages OpenTelemetry spans for OpenAI operations"""

    def __init__(self, operation_name: str, **kwargs):
        self.operation_name = operation_name
        self.span = None
        self.start_time = None
        self.kwargs = kwargs

    def __enter__(self):
        self.start_time = time.time()
        try:
            self.span = create_span(
                name=f"openai.{self.operation_name}",
                attributes={
                    BrokleOtelSpanAttributes.PROVIDER: "openai",
                    BrokleOtelSpanAttributes.OPERATION_TYPE: self.operation_name,
                    BrokleOtelSpanAttributes.REQUEST_START_TIME: self.start_time,
                }
            )

            # Add request-specific attributes
            self._add_request_attributes()

        except Exception as e:
            logger.warning(f"Failed to create span for {self.operation_name}: {e}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.span:
                # Record timing
                duration = time.time() - self.start_time if self.start_time else 0
                add_span_attributes(self.span, {
                    BrokleOtelSpanAttributes.REQUEST_DURATION: duration,
                    BrokleOtelSpanAttributes.RESPONSE_END_TIME: time.time()
                })

                # Record exception if any
                if exc_type:
                    record_span_exception(self.span, exc_val)

                self.span.end()

        except Exception as e:
            logger.warning(f"Failed to end span for {self.operation_name}: {e}")

    def _add_request_attributes(self):
        """Add request-specific attributes to span"""
        if not self.span:
            return

        try:
            # Extract common attributes
            if 'model' in self.kwargs:
                add_span_attributes(self.span, {
                    BrokleOtelSpanAttributes.MODEL_NAME: self.kwargs['model']
                })

            if 'messages' in self.kwargs:
                messages = self.kwargs['messages']
                if isinstance(messages, list) and messages:
                    add_span_attributes(self.span, {
                        BrokleOtelSpanAttributes.MESSAGE_COUNT: len(messages),
                        BrokleOtelSpanAttributes.INPUT_TOKENS: self._estimate_tokens(messages)
                    })

            if 'max_tokens' in self.kwargs:
                add_span_attributes(self.span, {
                    BrokleOtelSpanAttributes.MAX_TOKENS: self.kwargs['max_tokens']
                })

        except Exception as e:
            logger.warning(f"Failed to add request attributes: {e}")

    def _estimate_tokens(self, messages: List[Dict]) -> int:
        """Rough token estimation for observability"""
        try:
            total_chars = sum(len(str(msg.get('content', ''))) for msg in messages)
            return total_chars // 4  # Rough approximation
        except:
            return 0

    def record_response(self, response: Any):
        """Record response attributes"""
        if not self.span:
            return

        try:
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                add_span_attributes(self.span, {
                    BrokleOtelSpanAttributes.INPUT_TOKENS: getattr(usage, 'prompt_tokens', 0),
                    BrokleOtelSpanAttributes.OUTPUT_TOKENS: getattr(usage, 'completion_tokens', 0),
                    BrokleOtelSpanAttributes.TOTAL_TOKENS: getattr(usage, 'total_tokens', 0)
                })

            if hasattr(response, 'model'):
                add_span_attributes(self.span, {
                    BrokleOtelSpanAttributes.MODEL_NAME: response.model
                })

            # Record response content length (but not content for privacy)
            if hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    content_length = len(choice.message.content or '')
                    add_span_attributes(self.span, {
                        BrokleOtelSpanAttributes.RESPONSE_CONTENT_LENGTH: content_length
                    })

        except Exception as e:
            logger.warning(f"Failed to record response attributes: {e}")


def _wrap_openai_method(wrapped, instance, args, kwargs):
    """
    Universal wrapper for OpenAI methods.

    This function wraps any OpenAI method call with comprehensive observability
    while maintaining 100% compatibility with the original API.
    """
    # Skip wrapping if Brokle client not available
    try:
        brokle_client = get_client()
        if not brokle_client or not brokle_client.config.telemetry_enabled:
            return wrapped(*args, **kwargs)
    except:
        return wrapped(*args, **kwargs)

    # Determine operation name from method
    operation_name = getattr(wrapped, '__name__', 'unknown')
    if hasattr(instance, '__class__'):
        class_name = instance.__class__.__name__
        operation_name = f"{class_name}.{operation_name}"

    # Execute with observability
    with BrokleOpenAISpanManager(operation_name, **kwargs) as span_manager:
        try:
            response = wrapped(*args, **kwargs)
            span_manager.record_response(response)
            return response

        except Exception as e:
            # Handle provider errors gracefully
            handled_error = handle_provider_error(e, "openai", operation_name)
            raise handled_error


def _instrument_openai():
    """
    Instrument OpenAI SDK methods with observability.

    Uses wrapt to wrap specific methods that we want to track,
    following modern observability patterns of selective method wrapping.
    """
    if not HAS_OPENAI or not wrapt:
        return

    try:
        # Chat completions (most common)
        if hasattr(_original_openai.resources.chat.completions, 'Completions'):
            completions_class = _original_openai.resources.chat.completions.Completions
            wrapt.wrap_function_wrapper(
                completions_class, 'create', _wrap_openai_method
            )

        # Regular completions (legacy)
        if hasattr(_original_openai.resources.completions, 'Completions'):
            legacy_completions = _original_openai.resources.completions.Completions
            wrapt.wrap_function_wrapper(
                legacy_completions, 'create', _wrap_openai_method
            )

        # Embeddings
        if hasattr(_original_openai.resources.embeddings, 'Embeddings'):
            embeddings_class = _original_openai.resources.embeddings.Embeddings
            wrapt.wrap_function_wrapper(
                embeddings_class, 'create', _wrap_openai_method
            )

        # Fine-tuning (if available)
        if hasattr(_original_openai.resources.fine_tuning.jobs, 'Jobs'):
            jobs_class = _original_openai.resources.fine_tuning.jobs.Jobs
            for method in ['create', 'retrieve', 'list']:
                if hasattr(jobs_class, method):
                    wrapt.wrap_function_wrapper(
                        jobs_class, method, _wrap_openai_method
                    )

        logger.info("OpenAI SDK successfully instrumented for observability")

    except Exception as e:
        logger.warning(f"Failed to instrument OpenAI SDK: {e}")


# Auto-instrument when module is imported
if HAS_OPENAI and wrapt:
    _instrument_openai()


# Export everything from original OpenAI for drop-in compatibility
if HAS_OPENAI:
    # Main exports
    OpenAI = _OriginalOpenAI

    # Re-export all OpenAI modules and classes
    from openai import *

    # Ensure we maintain version compatibility
    __version__ = getattr(_original_openai, '__version__', 'unknown')

else:
    # Provide stub implementations when OpenAI not available
    class OpenAI:
        """Stub OpenAI client when SDK not available"""

        def __init__(self, *args, **kwargs):
            raise ProviderError(
                "OpenAI SDK not installed. Install with: pip install openai>=1.0.0"
            )


# Enable getattr passthrough for any missing attributes
def __getattr__(name: str) -> Any:
    """
    Passthrough any missing attributes to original OpenAI module.

    This ensures 100% compatibility even with new OpenAI SDK features
    that we haven't explicitly imported.
    """
    if HAS_OPENAI and hasattr(_original_openai, name):
        return getattr(_original_openai, name)

    raise AttributeError(f"module 'brokle.openai' has no attribute '{name}'")


# Public API
__all__ = [
    'OpenAI',
    # Add other common OpenAI exports here as needed
    'ChatCompletion',
    'Completion',
]

if HAS_OPENAI:
    # Add all original OpenAI exports to __all__
    try:
        __all__.extend(getattr(_original_openai, '__all__', []))
    except:
        pass