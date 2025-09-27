"""
Brokle SDK 2.0 - The Open-Source AI Control Plane

The Brokle SDK provides three integration patterns for adding AI observability,
routing, and optimization to your applications.

Three Integration Patterns:

1. **Wrapper Functions**:
   ```python
   from openai import OpenAI
   from anthropic import Anthropic
   from brokle import wrap_openai, wrap_anthropic

   openai_client = wrap_openai(
       OpenAI(api_key="sk-..."),
       tags=["production"],
       session_id="user_session_123"
   )
   anthropic_client = wrap_anthropic(
       Anthropic(api_key="sk-ant-..."),
       tags=["claude", "analysis"]
   )
   response = openai_client.chat.completions.create(...)
   ```

2. **Universal Decorator** (Framework-Agnostic):
   ```python
   from brokle import observe

   @observe()
   def my_ai_workflow(user_query: str) -> str:
       return llm.generate(user_query)
   ```

3. **Native SDK** (Full AI Platform Features):
   ```python
   from brokle import Brokle, get_client

   client = Brokle(api_key="bk_...")
   response = await client.chat.create(
       model="gpt-4",
       messages=[{"role": "user", "content": "Hello!"}]
   )
   ```
"""

from ._version import __version__
from .auth import AuthManager

# === PATTERN 3: NATIVE SDK ===
from .client import AsyncBrokle, Brokle, get_client
from .config import Config

# === PATTERN 2: UNIVERSAL DECORATOR (UNCHANGED) ===
from .decorators import (
    ObserveConfig,
    observe,
    observe_llm,
    observe_retrieval,
    trace_workflow,
)

# Exception classes
from .exceptions import (  # EvaluationError removed - evaluation moved to backend
    APIError,
    AuthenticationError,
    BrokleError,
    CacheError,
    ConfigurationError,
    NetworkError,
    ProviderError,
    QuotaExceededError,
    RateLimitError,
    TimeoutError,
    UnsupportedOperationError,
    ValidationError,
)
from .observability.attributes import BrokleOtelSpanAttributes

# === PATTERN 1: WRAPPER FUNCTIONS (NEW IN 2.0) ===
from .wrappers import wrap_cohere  # Future
from .wrappers import wrap_google  # Future
from .wrappers import (
    wrap_anthropic,
    wrap_openai,
)

# These advanced features are available through the Native SDK Pattern 3

# Main exports - Clean 3-Pattern Architecture
__all__ = [
    # === PATTERN 1: WRAPPER FUNCTIONS ===
    "wrap_openai",  # OpenAI client wrapper
    "wrap_anthropic",  # Anthropic client wrapper
    "wrap_google",  # Google AI wrapper (future)
    "wrap_cohere",  # Cohere wrapper (future)
    # === PATTERN 2: UNIVERSAL DECORATOR (Framework-Agnostic) ===
    "observe",  # Universal @observe() decorator
    "trace_workflow",  # Workflow context manager
    "observe_llm",  # LLM-specific decorator
    "observe_retrieval",  # Retrieval-specific decorator
    "ObserveConfig",  # Decorator configuration
    # === PATTERN 3: NATIVE SDK (OpenAI-Compatible + Brokle Features) ===
    "Brokle",  # Main sync client class
    "AsyncBrokle",  # Async client class
    "get_client",  # Client accessor
    "Config",  # Configuration management
    "AuthManager",  # Authentication handling
    "BrokleOtelSpanAttributes",  # Telemetry attributes
    # === SHARED: EXCEPTION CLASSES ===
    "BrokleError",
    "AuthenticationError",
    "RateLimitError",
    "ConfigurationError",
    "APIError",
    "NetworkError",
    "ValidationError",
    "TimeoutError",
    "UnsupportedOperationError",
    "QuotaExceededError",
    "ProviderError",
    "CacheError",
    # "EvaluationError" removed - evaluation moved to backend
    # === NOTE: EVALUATION FRAMEWORK MOVED TO BACKEND ===
    # All evaluation logic is now handled by the backend
    # === NATIVE SDK: ADVANCED FEATURES (Integrated into main client) ===
    # Advanced features are available through Brokle() and AsyncBrokle() clients
    # No separate exports needed - clean architecture with unified interface
    # === METADATA ===
    "__version__",
]
