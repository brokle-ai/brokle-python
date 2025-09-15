"""
Brokle Platform Python SDK

A comprehensive Python SDK for the Brokle Platform that provides three integration patterns:
1. OpenAI drop-in replacement - Zero code changes beyond import
2. @observe decorator - LangFuse-style observability 
3. Native SDK - Full platform features

The SDK provides intelligent routing, cost optimization, semantic caching, and comprehensive
observability for AI applications with OpenTelemetry integration.
"""

from .client import Brokle, get_client
from .config import Config, configure, get_config, reset_config
from .auth import AuthManager
from .decorators import observe
from .observability_decorators import observe_enhanced
from .types.attributes import BrokleOtelSpanAttributes
from .clients.observability import ObservabilityClient
from .auto_instrumentation import auto_instrument, print_status, get_status, get_registry
from ._version import __version__

# Core client and configuration
__all__ = [
    # Main client
    "Brokle",
    "get_client",

    # Configuration
    "Config",
    "configure",
    "get_config",
    "reset_config",

    # Authentication
    "AuthManager",

    # Decorators
    "observe",
    "observe_enhanced",

    # Observability
    "ObservabilityClient",

    # Auto-instrumentation
    "auto_instrument",
    "print_status",
    "get_status",
    "get_registry",

    # OpenTelemetry attributes
    "BrokleOtelSpanAttributes",
]