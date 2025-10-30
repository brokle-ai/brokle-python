"""
Clean Shared Provider System for Brokle SDK.

This module provides unified AI provider implementations that serve all three patterns:
- Pattern 1: Wrapper functions (wrap_openai, wrap_anthropic)
- Pattern 2: @observe decorator
- Pattern 3: Native SDK client

All patterns use the same provider logic for consistent behavior and attributes.
"""

from typing import Dict, Type

from .anthropic import AnthropicProvider
from .base import BaseProvider
from .openai import OpenAIProvider

# Clean provider registry
PROVIDERS: Dict[str, Type[BaseProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}


def get_provider(provider_name: str, **config) -> BaseProvider:
    """
    Get a provider instance by name.

    Args:
        provider_name: Provider name ("openai", "anthropic", etc.)
        **config: Provider configuration options

    Returns:
        Configured provider instance

    Raises:
        ValueError: If provider name is not supported
    """
    if provider_name not in PROVIDERS:
        supported = ", ".join(PROVIDERS.keys())
        raise ValueError(
            f"Unsupported provider '{provider_name}'. Supported: {supported}"
        )

    provider_class = PROVIDERS[provider_name]
    return provider_class(**config)


def register_provider(name: str, provider_class: Type[BaseProvider]) -> None:
    """
    Register a new provider class.

    Args:
        name: Provider name
        provider_class: Provider implementation class
    """
    PROVIDERS[name] = provider_class


def list_providers() -> list[str]:
    """Get list of supported provider names."""
    return list(PROVIDERS.keys())


__all__ = [
    "BaseProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "get_provider",
    "register_provider",
    "list_providers",
    "PROVIDERS",
]
