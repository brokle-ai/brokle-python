"""
OpenAI Auto-Instrumentation for Brokle Platform.

This module provides automatic instrumentation for OpenAI API calls.
Simply importing this module enables comprehensive observability for all OpenAI usage.

Usage:
    import brokle.openai  # Enables auto-instrumentation

    from openai import OpenAI
    client = OpenAI()
    # All OpenAI calls now automatically tracked by Brokle

    # For manual function observability, use @observe decorator:
    from brokle import observe

    @observe()
    def my_ai_function():
        # Your custom logic with automatic tracing
        pass
"""

# Import auto-instrumentation module - this triggers instrumentation on import
from ..integrations.openai import (
    is_instrumented,
    get_instrumentation_errors,
    get_instrumentation_status,
    HAS_OPENAI,
    HAS_WRAPT
)

__all__ = [
    "is_instrumented",
    "get_instrumentation_errors",
    "get_instrumentation_status",
    "HAS_OPENAI",
    "HAS_WRAPT"
]