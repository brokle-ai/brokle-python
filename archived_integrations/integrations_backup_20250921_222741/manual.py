"""
Manual Wrapper API for Brokle Platform.

This module provides manual instrumentation functions that compete with
LangSmith, Opik, and LangFuse approaches while offering superior functionality.

Competitive Analysis:
- LangSmith: Uses decorators and wrap_* functions
- Opik: Uses track_* functions with simple client wrapping
- LangFuse: Uses SDK replacement approach

Brokle's Advantage:
- Combines best of all approaches
- More configuration options than competitors
- Reuses auto-instrumentation engine for consistency
- Works with existing codebases with minimal changes

Usage:
    # Simple client wrapping (like Opik)
    client = track_openai(OpenAI())

    # Advanced configuration (superior to competitors)
    client = track_openai(
        OpenAI(),
        sample_rate=0.1,
        tags={"team": "ai-platform"},
        metadata={"version": "1.0"}
    )

    # Decorator pattern (like LangSmith)
    @observe()
    def my_function():
        return client.chat.completions.create(...)
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Union
from functools import wraps

from ._registry import get_provider_instrumentation
from ._engine import InstrumentationEngine

logger = logging.getLogger(__name__)


class TrackedClient:
    """
    Base class for manually tracked clients.

    This provides a transparent proxy that wraps the original client
    while adding instrumentation to specific methods.
    """

    def __init__(
        self,
        original_client: Any,
        provider_name: str,
        **options
    ):
        """
        Initialize tracked client.

        Args:
            original_client: Original provider client (e.g., OpenAI())
            provider_name: Provider name for instrumentation
            **options: Configuration options (sample_rate, tags, metadata, etc.)
        """
        self._original_client = original_client
        self._provider_name = provider_name
        self._options = options
        self._engine = InstrumentationEngine()
        self._wrapped_methods = {}

        # Get instrumentation for this provider
        self._instrumentation = get_provider_instrumentation(provider_name)
        if not self._instrumentation:
            logger.warning(f"No instrumentation available for provider: {provider_name}")
            return

        # Apply manual wrappers to configured methods
        self._apply_manual_wrappers()

    def _apply_manual_wrappers(self):
        """Apply manual wrappers to configured methods."""
        if not self._instrumentation:
            return

        try:
            method_configs = self._instrumentation.get_method_configs()

            for config in method_configs:
                try:
                    # Map config.object to actual client attribute path
                    client_path = self._map_config_to_client_path(config)
                    if not client_path:
                        logger.debug(f"No client path mapping for {config.object}")
                        continue

                    # Navigate to the method using the mapped path
                    method_path = client_path.split(".")
                    current_obj = self._original_client

                    # Navigate to the method (e.g., "chat.completions.create")
                    for part in method_path[:-1]:
                        current_obj = getattr(current_obj, part)

                    method_name = method_path[-1]
                    original_method = getattr(current_obj, method_name)

                    # Create manual wrapper
                    wrapped_method = self._engine.create_manual_wrapper(
                        self._instrumentation,
                        original_method,
                        config,
                        **self._options
                    )

                    # Store wrapped method with client path as key
                    self._wrapped_methods[client_path] = wrapped_method

                    logger.debug(f"Wrapped method: {config.object} -> {client_path}")

                except AttributeError as e:
                    logger.debug(f"Method {config.object} not found on client: {e}")
                except Exception as e:
                    logger.debug(f"Failed to wrap method {config.object}: {e}")

        except Exception as e:
            logger.debug(f"Failed to apply manual wrappers: {e}")

    def _map_config_to_client_path(self, config) -> Optional[str]:
        """Map instrumentation config object to actual client attribute path."""
        # Provider-specific mappings using both config.module and config.object
        if self._provider_name == "openai":
            # Use module + object as key to distinguish different APIs
            key = f"{config.module}.{config.object}"

            openai_mappings = {
                # Chat completions
                "openai.resources.chat.completions.Completions.create": "chat.completions.create",
                "openai.resources.chat.completions.AsyncCompletions.create": "chat.completions.create",

                # Text completions (legacy)
                "openai.resources.completions.Completions.create": "completions.create",
                "openai.resources.completions.AsyncCompletions.create": "completions.create",

                # Embeddings
                "openai.resources.embeddings.Embeddings.create": "embeddings.create",
                "openai.resources.embeddings.AsyncEmbeddings.create": "embeddings.create",
            }

            mapped_path = openai_mappings.get(key)
            if mapped_path:
                return mapped_path

            # Fallback: try object-only mapping for backward compatibility
            object_mappings = {
                "Embeddings.create": "embeddings.create",
                "AsyncEmbeddings.create": "embeddings.create",
            }
            return object_mappings.get(config.object)

        elif self._provider_name == "anthropic":
            # Anthropic is simpler - module distinguishes the APIs clearly
            key = f"{config.module}.{config.object}"

            anthropic_mappings = {
                # Messages API (current)
                "anthropic.resources.messages.Messages.create": "messages.create",
                "anthropic.resources.messages.AsyncMessages.create": "messages.create",

                # Completions API (legacy)
                "anthropic.resources.completions.Completions.create": "completions.create",
                "anthropic.resources.completions.AsyncCompletions.create": "completions.create",
            }

            mapped_path = anthropic_mappings.get(key)
            if mapped_path:
                return mapped_path

            # Fallback for simple object mapping
            object_mappings = {
                "Messages.create": "messages.create",
                "AsyncMessages.create": "messages.create",
            }
            return object_mappings.get(config.object)

        # Default: try to use config.object as-is (fallback)
        return config.object

    def __getattr__(self, name: str) -> Any:
        """
        Proxy attribute access to original client.

        This ensures the tracked client behaves identically to the original
        client while intercepting instrumented methods.
        """
        # Check if this is a wrapped method path
        for client_path, wrapped_method in self._wrapped_methods.items():
            if name == client_path.split(".")[0]:
                # Return wrapped attribute that handles nested access
                return self._create_nested_wrapper(name, client_path, wrapped_method)

        # Default to original client attribute
        return getattr(self._original_client, name)

    def _create_nested_wrapper(self, attr_name: str, full_path: str, wrapped_method: Callable):
        """Create nested attribute wrapper for complex method paths."""
        parts = full_path.split(".")

        if len(parts) == 1:
            # Direct method (e.g., "create")
            return wrapped_method

        # Nested access (e.g., "chat.completions.create")
        if attr_name == parts[0]:
            # Return an object that handles the next level
            original_attr = getattr(self._original_client, attr_name)

            class NestedWrapper:
                def __init__(self, original, remaining_path, wrapped_fn):
                    self._original = original
                    self._remaining_path = remaining_path
                    self._wrapped_fn = wrapped_fn

                def __getattr__(self, name):
                    if name == self._remaining_path[0]:
                        if len(self._remaining_path) == 1:
                            # Final method
                            return self._wrapped_fn
                        else:
                            # Continue nesting
                            return NestedWrapper(
                                getattr(self._original, name),
                                self._remaining_path[1:],
                                self._wrapped_fn
                            )
                    return getattr(self._original, name)

            return NestedWrapper(original_attr, parts[1:], wrapped_method)

        return getattr(self._original_client, attr_name)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"TrackedClient({self._provider_name}, {self._original_client})"


def track_openai(
    client,
    *,
    sample_rate: float = 1.0,
    enable_caching: bool = True,
    enable_evaluation: bool = True,
    tags: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Manually track an OpenAI client with instrumentation.

    This function provides explicit control over OpenAI instrumentation,
    superior to competitors like Opik's track_openai function.

    Args:
        client: OpenAI client instance (OpenAI or AsyncOpenAI)
        sample_rate: Sampling rate (0.0-1.0, default: 1.0)
        enable_caching: Enable Brokle semantic caching
        enable_evaluation: Enable response evaluation
        tags: Per-request tags dictionary (e.g., {"team": "core", "experiment": "v2"})
        metadata: Additional per-request metadata (e.g., {"request_id": "123", "user_session": "abc"})

    Returns:
        Tracked client that behaves identically to original

    Example:
        from openai import OpenAI
        from brokle import track_openai

        # Basic tracking
        client = track_openai(OpenAI())

        # Advanced tracking with per-request context
        client = track_openai(
            OpenAI(),
            sample_rate=0.1,
            enable_caching=True,
            enable_evaluation=False,
            tags={"team": "core", "experiment": "v2"},
            metadata={"request_id": "123", "user_session": "abc"}
        )

        # Use exactly like normal OpenAI client
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}]
        )

    Note:
        For project configuration, use global client initialization:
        brokle.configure(project="production")
    """
    # Check if provider is available
    if not get_provider_instrumentation("openai"):
        logger.warning("OpenAI instrumentation not available")
        return client

    # Create tracked client with explicit options
    options = {
        "sample_rate": sample_rate,
        "enable_caching": enable_caching,
        "enable_evaluation": enable_evaluation,
    }

    # Add explicit parameters if provided
    if tags is not None:
        options["tags"] = tags
    if metadata is not None:
        options["metadata"] = metadata

    return TrackedClient(client, "openai", **options)


def track_anthropic(
    client,
    *,
    sample_rate: float = 1.0,
    enable_caching: bool = True,
    enable_evaluation: bool = True,
    tags: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Manually track an Anthropic client with instrumentation.

    Args:
        client: Anthropic client instance
        sample_rate: Sampling rate (0.0-1.0, default: 1.0)
        enable_caching: Enable Brokle semantic caching
        enable_evaluation: Enable response evaluation
        tags: Per-request tags dictionary (e.g., {"team": "core", "experiment": "v2"})
        metadata: Additional per-request metadata (e.g., {"request_id": "123", "user_session": "abc"})

    Returns:
        Tracked client that behaves identically to original

    Example:
        from anthropic import Anthropic
        from brokle import track_anthropic

        # Basic tracking
        client = track_anthropic(Anthropic())

        # Advanced tracking with per-request context
        client = track_anthropic(
            Anthropic(),
            sample_rate=0.5,
            enable_caching=False,
            enable_evaluation=True,
            tags={"team": "research", "model_type": "sonnet"},
            metadata={"session_id": "456", "experiment": "claude-eval"}
        )

        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": "Hello!"}]
        )

    Note:
        For project configuration, use global client initialization:
        brokle.configure(project="production")
    """
    # Check if provider is available
    if not get_provider_instrumentation("anthropic"):
        logger.warning("Anthropic instrumentation not available")
        return client

    options = {
        "sample_rate": sample_rate,
        "enable_caching": enable_caching,
        "enable_evaluation": enable_evaluation,
    }

    # Add explicit parameters if provided
    if tags is not None:
        options["tags"] = tags
    if metadata is not None:
        options["metadata"] = metadata

    return TrackedClient(client, "anthropic", **options)


def observe(
    *,
    name: Optional[str] = None,
    tags: Optional[Union[List[str], Dict[str, str]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    capture_input: bool = True,
    capture_output: bool = True,
    as_type: str = "span"
) -> Callable:
    """
    Decorator for observing functions (like LangSmith's @traceable).

    This provides compatibility with LangSmith's decorator pattern while
    offering enhanced functionality.

    Args:
        name: Custom observation name
        tags: Tags as list or dict
        metadata: Custom metadata
        capture_input: Whether to capture function input
        capture_output: Whether to capture function output
        as_type: Observation type ("span" or "generation")

    Returns:
        Decorated function with observation

    Example:
        # Basic observation
        @observe()
        def my_ai_function():
            return openai.chat.completions.create(...)

        # Advanced observation
        @observe(
            name="story-generation",
            tags={"type": "creative", "priority": "high"},
            metadata={"version": "2.0"},
            as_type="generation"
        )
        def generate_story(prompt: str) -> str:
            # Function automatically tracked
            return client.chat.completions.create(...)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Import here to avoid circular imports
            try:
                from ..decorators import observe as observe_decorator

                # Convert tags to dict format if needed
                processed_tags = tags
                if isinstance(tags, list):
                    processed_tags = {tag: "true" for tag in tags}

                # Use existing observe decorator
                decorated_func = observe_decorator(
                    name=name or func.__name__,
                    user_id=None,  # Could be extracted from context
                    session_id=None,  # Could be extracted from context
                    tags=list(processed_tags.keys()) if processed_tags else None,
                    metadata=metadata,
                    capture_input=capture_input,
                    capture_output=capture_output,
                    as_type=as_type
                )(func)

                return decorated_func(*args, **kwargs)

            except ImportError:
                logger.warning("Brokle observe decorator not available")
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Convenience functions for common patterns
def track_client(client, provider: str, **options) -> Any:
    """
    Generic client tracking function.

    Args:
        client: Client instance to track
        provider: Provider name ("openai", "anthropic", etc.)
        **options: Tracking options

    Returns:
        Tracked client
    """
    if provider == "openai":
        return track_openai(client, **options)
    elif provider == "anthropic":
        return track_anthropic(client, **options)
    else:
        logger.warning(f"No manual tracking available for provider: {provider}")
        return client


def get_tracking_status() -> Dict[str, Any]:
    """
    Get status of manual tracking capabilities.

    Returns:
        Dictionary with tracking status information
    """
    from ._registry import get_all_provider_status

    status = get_all_provider_status()

    return {
        "manual_tracking_available": True,
        "supported_providers": list(status.keys()),
        "provider_status": {
            name: {
                "available": info.available,
                "instrumented": info.instrumented,
                "manual_tracking": True  # All providers support manual tracking
            }
            for name, info in status.items()
        }
    }


# Export public API
__all__ = [
    "track_openai",
    "track_anthropic",
    "track_client",
    "observe",
    "get_tracking_status",
    "TrackedClient"
]
