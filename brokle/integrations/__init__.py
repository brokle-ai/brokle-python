"""
Core Instrumentation Engine

Universal instrumentation system for AI providers with consistent
observability patterns across OpenAI, Anthropic, Google AI, and more.

This module provides the foundation for Brokle's provider-agnostic
observability system using OpenTelemetry and wrapt.
"""

from ..observability.attributes import (
    BrokleOtelSpanAttributes as BrokleInstrumentationAttributes,
)
from .instrumentation import InstrumentationContext, UniversalInstrumentation

__all__ = [
    "UniversalInstrumentation",
    "InstrumentationContext",
    "BrokleInstrumentationAttributes",
]
