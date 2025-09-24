"""
Wrapper Functions - Pattern 1 Implementation

Explicit wrapper functions following LangSmith/Optik patterns.
Each function wraps the original provider SDK with Brokle observability.

Usage:
    from openai import OpenAI
    from anthropic import Anthropic
    from brokle import wrap_openai, wrap_anthropic

    openai_client = wrap_openai(OpenAI(api_key="sk-..."))
    anthropic_client = wrap_anthropic(Anthropic(api_key="sk-ant-..."))
"""

from .openai import wrap_openai
from .anthropic import wrap_anthropic

# Future providers - stubs for now
def wrap_google(*args, **kwargs):
    """Google AI wrapper - coming in v2.1.0"""
    raise ImportError("Google AI wrapper not yet implemented. Coming in v2.1.0")

def wrap_cohere(*args, **kwargs):
    """Cohere wrapper - coming in v2.1.0"""
    raise ImportError("Cohere wrapper not yet implemented. Coming in v2.1.0")

__all__ = [
    "wrap_openai",
    "wrap_anthropic",
    "wrap_google",
    "wrap_cohere",
]