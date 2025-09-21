"""
Provider Registry System for Brokle Platform.

This module manages the registration and discovery of provider instrumentations.
It provides a centralized way to manage multiple AI providers and their
configurations, enabling both auto-instrumentation and manual wrapper APIs.

The registry supports:
- Dynamic provider registration
- Version compatibility checking
- Status reporting across providers
- Runtime enable/disable controls

Usage:
    # Get instrumentation for a provider
    openai_instr = get_provider_instrumentation("openai")

    # Register custom provider
    register_provider("custom_llm", MyCustomInstrumentation())

    # Get status across all providers
    status = get_all_provider_status()
"""

import logging
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass
import importlib

from ._base import BaseInstrumentation, UnsupportedProviderError

logger = logging.getLogger(__name__)


@dataclass
class ProviderStatus:
    """Status information for a provider."""
    name: str
    available: bool
    instrumented: bool
    version: Optional[str] = None
    methods_wrapped: int = 0
    errors: List[str] = None
    supported_versions: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.supported_versions is None:
            self.supported_versions = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "available": self.available,
            "instrumented": self.instrumented,
            "version": self.version,
            "methods_wrapped": self.methods_wrapped,
            "errors": self.errors.copy(),
            "supported_versions": self.supported_versions.copy()
        }


class ProviderRegistry:
    """
    Central registry for managing provider instrumentations.

    This class handles registration, discovery, and status management
    for all available AI provider integrations.
    """

    def __init__(self):
        """Initialize the provider registry."""
        self._providers: Dict[str, BaseInstrumentation] = {}
        self._disabled_providers: Set[str] = set()
        self._global_disabled = False
        self._errors: Dict[str, List[str]] = {}

    def register_provider(
        self,
        name: str,
        instrumentation: BaseInstrumentation
    ) -> None:
        """
        Register a provider instrumentation.

        Args:
            name: Provider name (e.g., "openai", "anthropic")
            instrumentation: Implementation of BaseInstrumentation

        Raises:
            ValueError: If name is already registered or instrumentation is invalid
        """
        if not isinstance(instrumentation, BaseInstrumentation):
            raise ValueError(f"Instrumentation must inherit from BaseInstrumentation")

        if name in self._providers:
            logger.warning(f"Provider '{name}' already registered, overwriting")

        self._providers[name] = instrumentation
        logger.debug(f"Registered provider: {name}")

    def get_provider(self, name: str) -> Optional[BaseInstrumentation]:
        """
        Get instrumentation for a provider.

        Args:
            name: Provider name

        Returns:
            BaseInstrumentation instance or None if not found
        """
        return self._providers.get(name)

    def get_all_providers(self) -> Dict[str, BaseInstrumentation]:
        """Get all registered providers."""
        return self._providers.copy()

    def is_provider_available(self, name: str) -> bool:
        """Check if a provider is available (registered)."""
        return name in self._providers

    def is_provider_enabled(self, name: str) -> bool:
        """Check if a provider is enabled for instrumentation."""
        if self._global_disabled:
            return False
        return name not in self._disabled_providers

    def disable_provider(self, name: str) -> None:
        """Disable instrumentation for a specific provider."""
        self._disabled_providers.add(name)
        logger.debug(f"Disabled provider: {name}")

    def enable_provider(self, name: str) -> None:
        """Enable instrumentation for a specific provider."""
        self._disabled_providers.discard(name)
        logger.debug(f"Enabled provider: {name}")

    def disable_all(self) -> None:
        """Disable all provider instrumentations."""
        self._global_disabled = True
        logger.debug("Disabled all provider instrumentations")

    def enable_all(self) -> None:
        """Enable all provider instrumentations."""
        self._global_disabled = False
        self._disabled_providers.clear()
        logger.debug("Enabled all provider instrumentations")

    def record_error(self, provider: str, error: str) -> None:
        """Record an error for a provider."""
        if provider not in self._errors:
            self._errors[provider] = []

        self._errors[provider].append(error)

        # Keep only last 10 errors per provider
        self._errors[provider] = self._errors[provider][-10:]

    def get_provider_errors(self, provider: str) -> List[str]:
        """Get recorded errors for a provider."""
        return self._errors.get(provider, []).copy()

    def clear_provider_errors(self, provider: str) -> None:
        """Clear recorded errors for a provider."""
        if provider in self._errors:
            del self._errors[provider]

    def _get_real_instrumentation_status(self, name: str) -> bool:
        """
        Get the actual instrumentation status from provider module.

        This checks the provider's own _instrumented flag rather than
        relying on configuration state.

        Args:
            name: Provider name

        Returns:
            True if provider is actually instrumented, False otherwise
        """
        try:
            if name == "openai":
                from brokle.integrations.openai import is_instrumented
                return is_instrumented()
            elif name == "anthropic":
                from brokle.integrations.anthropic import is_instrumented
                return is_instrumented()
            else:
                # For unknown providers, fall back to configuration check
                logger.debug(f"No instrumentation status check available for provider: {name}")
                return False
        except ImportError:
            logger.debug(f"Provider module '{name}' not available, instrumentation status: False")
            return False
        except Exception as e:
            logger.debug(f"Failed to check instrumentation status for '{name}': {e}")
            return False

    def _get_provider_instrumentation_errors(self, name: str) -> List[str]:
        """
        Get instrumentation errors from provider module.

        These are distinct from registry errors and provide insight into
        why instrumentation failed at the provider level.

        Args:
            name: Provider name

        Returns:
            List of instrumentation error messages
        """
        try:
            if name == "openai":
                from brokle.integrations.openai import get_instrumentation_errors
                return get_instrumentation_errors()
            elif name == "anthropic":
                from brokle.integrations.anthropic import get_instrumentation_errors
                return get_instrumentation_errors()
            else:
                return []
        except ImportError:
            logger.debug(f"Provider module '{name}' not available for error retrieval")
            return []
        except Exception as e:
            logger.debug(f"Failed to get instrumentation errors for '{name}': {e}")
            return []

    def get_provider_status(self, name: str) -> ProviderStatus:
        """
        Get detailed status for a specific provider.

        Args:
            name: Provider name

        Returns:
            ProviderStatus object with comprehensive information
        """
        instrumentation = self._providers.get(name)

        if not instrumentation:
            return ProviderStatus(
                name=name,
                available=False,
                instrumented=False,
                errors=[f"Provider '{name}' not registered"]
            )

        # Check if provider library is available
        try:
            version = self._detect_provider_version(name)
            library_available = True
        except ImportError:
            version = None
            library_available = False

        # Get method configurations
        try:
            method_configs = instrumentation.get_method_configs()
            methods_wrapped = len(method_configs) if self.is_provider_enabled(name) else 0
        except Exception as e:
            method_configs = []
            methods_wrapped = 0
            self.record_error(name, f"Failed to get method configs: {e}")

        # Validate environment
        try:
            env_validation = instrumentation.validate_environment()
        except Exception as e:
            env_validation = {
                "supported": False,
                "issues": [f"Environment validation failed: {e}"]
            }

        # Combine registry errors with provider instrumentation errors
        registry_errors = self.get_provider_errors(name)
        instrumentation_errors = self._get_provider_instrumentation_errors(name)
        all_errors = registry_errors + instrumentation_errors

        return ProviderStatus(
            name=name,
            available=library_available,
            instrumented=self._get_real_instrumentation_status(name),
            version=version,
            methods_wrapped=methods_wrapped,
            errors=all_errors,
            supported_versions=instrumentation.get_supported_versions()
        )

    def get_all_status(self) -> Dict[str, ProviderStatus]:
        """Get status for all registered providers."""
        return {
            name: self.get_provider_status(name)
            for name in self._providers.keys()
        }

    def _detect_provider_version(self, name: str) -> Optional[str]:
        """Detect the version of a provider library."""
        try:
            if name == "openai":
                import openai
                return getattr(openai, "__version__", "unknown")
            elif name == "anthropic":
                import anthropic
                return getattr(anthropic, "__version__", "unknown")
            elif name == "google-generativeai":
                import google.generativeai as genai
                return getattr(genai, "__version__", "unknown")
            else:
                # Try generic approach
                module = importlib.import_module(name)
                return getattr(module, "__version__", "unknown")
        except ImportError:
            raise
        except Exception:
            return "unknown"

    def auto_discover_providers(self) -> None:
        """
        Automatically discover and register available providers.

        This method attempts to import and register providers that
        have been implemented in the integrations package.
        """
        provider_modules = [
            ("openai", "brokle.integrations.openai.instrumentation", "OpenAIInstrumentation"),
            ("anthropic", "brokle.integrations.anthropic.instrumentation", "AnthropicInstrumentation"),
            # Future providers can be added here
        ]

        for provider_data in provider_modules:
            provider_name, module_path, class_name = provider_data
            try:
                # Try to import the instrumentation module
                module = importlib.import_module(module_path)

                # Look for instrumentation class using explicit class name
                instrumentation_class = getattr(module, class_name, None)

                if instrumentation_class and issubclass(instrumentation_class, BaseInstrumentation):
                    # Create and register the instrumentation
                    instrumentation = instrumentation_class()
                    self.register_provider(provider_name, instrumentation)
                    logger.debug(f"Auto-discovered provider: {provider_name}")

            except ImportError:
                logger.debug(f"Provider {provider_name} not available (import failed)")
            except Exception as e:
                logger.debug(f"Failed to auto-discover provider {provider_name}: {e}")
                self.record_error(provider_name, f"Auto-discovery failed: {e}")

    def get_debug_info(self) -> Dict[str, Any]:
        """Get comprehensive debug information for troubleshooting."""
        return {
            "registry_state": {
                "total_providers": len(self._providers),
                "global_disabled": self._global_disabled,
                "disabled_providers": list(self._disabled_providers),
                "registered_providers": list(self._providers.keys())
            },
            "provider_status": {
                name: status.to_dict()
                for name, status in self.get_all_status().items()
            },
            "errors": self._errors.copy()
        }


# Global registry instance
_global_registry = ProviderRegistry()


def register_provider(name: str, instrumentation: BaseInstrumentation) -> None:
    """Register a provider in the global registry."""
    _global_registry.register_provider(name, instrumentation)


def get_provider_instrumentation(name: str) -> Optional[BaseInstrumentation]:
    """Get provider instrumentation from global registry."""
    return _global_registry.get_provider(name)


def get_all_provider_instrumentations() -> Dict[str, BaseInstrumentation]:
    """Get all provider instrumentations from global registry."""
    return _global_registry.get_all_providers()


def is_provider_enabled(name: str) -> bool:
    """Check if provider is enabled in global registry."""
    return _global_registry.is_provider_enabled(name)


def disable_provider(name: str) -> None:
    """Disable provider in global registry."""
    _global_registry.disable_provider(name)


def enable_provider(name: str) -> None:
    """Enable provider in global registry."""
    _global_registry.enable_provider(name)


def disable_all_instrumentation() -> None:
    """Disable all instrumentation in global registry."""
    _global_registry.disable_all()


def enable_all_instrumentation() -> None:
    """Enable all instrumentation in global registry."""
    _global_registry.enable_all()


def get_provider_status(name: str) -> ProviderStatus:
    """Get provider status from global registry."""
    return _global_registry.get_provider_status(name)


def get_all_provider_status() -> Dict[str, ProviderStatus]:
    """Get status for all providers from global registry."""
    return _global_registry.get_all_status()


def auto_discover_providers() -> None:
    """Auto-discover providers in global registry."""
    _global_registry.auto_discover_providers()


def get_debug_info() -> Dict[str, Any]:
    """Get debug information from global registry."""
    return _global_registry.get_debug_info()


# Initialize with auto-discovery
try:
    auto_discover_providers()
except Exception as e:
    logger.debug(f"Auto-discovery failed: {e}")