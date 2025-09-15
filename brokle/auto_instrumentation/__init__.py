"""
Auto-instrumentation for popular LLM libraries.

This module provides automatic observability instrumentation for popular
LLM libraries like OpenAI, Anthropic, LangChain, and others.
"""

from .openai_instrumentation import OpenAIInstrumentation
from .anthropic_instrumentation import AnthropicInstrumentation
from .langchain_instrumentation import LangChainInstrumentation
from .registry import (
    InstrumentationRegistry,
    auto_instrument,
    print_status,
    print_health_report,
    get_status,
    get_health_report,
    reset_all_errors,
    get_registry
)

# Export error handling utilities for advanced usage
from .error_handlers import (
    InstrumentationError,
    LibraryNotAvailableError,
    ObservabilityError,
    ConfigurationError,
    ErrorSeverity,
    get_error_handler
)

__all__ = [
    # Core instrumentation
    "OpenAIInstrumentation",
    "AnthropicInstrumentation",
    "LangChainInstrumentation",
    "InstrumentationRegistry",
    "auto_instrument",

    # Status and health monitoring
    "print_status",
    "print_health_report",
    "get_status",
    "get_health_report",
    "reset_all_errors",
    "get_registry",

    # Error handling (advanced usage)
    "InstrumentationError",
    "LibraryNotAvailableError",
    "ObservabilityError",
    "ConfigurationError",
    "ErrorSeverity",
    "get_error_handler"
]