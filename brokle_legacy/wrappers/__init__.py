"""
Wrapper Functions - Pattern 1 Implementation

Explicit wrapper functions for client instrumentation.
Each function wraps the original provider SDK with Brokle observability.

Usage:
    from openai import OpenAI
    from anthropic import Anthropic
    from brokle import wrap_openai, wrap_anthropic

    openai_client = wrap_openai(OpenAI(api_key="sk-..."))
    anthropic_client = wrap_anthropic(Anthropic(api_key="sk-ant-..."))
"""

from .anthropic import wrap_anthropic
from .openai import wrap_openai


# Future providers - stubs for now
def wrap_google(*args, **kwargs):
    """Google AI wrapper - not yet implemented"""
    raise NotImplementedError("Google AI wrapper not yet implemented.")


def wrap_cohere(*args, **kwargs):
    """Cohere wrapper - not yet implemented"""
    raise NotImplementedError("Cohere wrapper not yet implemented.")


__all__ = [
    "wrap_openai",
    "wrap_anthropic",
    "wrap_google",
    "wrap_cohere",
]
