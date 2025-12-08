"""
Framework integrations for automatic instrumentation.

This module provides automatic tracing for popular LLM frameworks:
- LangChain: Callback-based tracing
- LlamaIndex: Global handler tracing

Optional dependencies are handled gracefully - if a framework SDK
is not installed, the integration will be unavailable but won't
cause import errors.

Usage:
    # LangChain
    from brokle.integrations import BrokleLangChainCallback
    callback = BrokleLangChainCallback(user_id="user-123")

    # LlamaIndex
    from brokle.integrations import set_global_handler
    set_global_handler("brokle", user_id="user-123")
"""

# LangChain integration (requires langchain package)
try:
    from .langchain import BrokleLangChainCallback

    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    BrokleLangChainCallback = None
    _langchain_import_error = str(e)


# LlamaIndex integration (requires llama-index package)
try:
    from .llamaindex import BrokleLlamaIndexHandler, set_global_handler

    LLAMAINDEX_AVAILABLE = True
except ImportError as e:
    LLAMAINDEX_AVAILABLE = False
    BrokleLlamaIndexHandler = None
    set_global_handler = None
    _llamaindex_import_error = str(e)


def check_langchain_available():
    """
    Check if LangChain integration is available.

    Returns:
        bool: True if LangChain is installed and integration is available

    Raises:
        ImportError: If LangChain is not installed (with helpful message)
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain integration requires the 'langchain' package. "
            "Install with: pip install brokle-otel[langchain]"
            f"\nOriginal error: {_langchain_import_error}"
        )
    return True


def check_llamaindex_available():
    """
    Check if LlamaIndex integration is available.

    Returns:
        bool: True if LlamaIndex is installed and integration is available

    Raises:
        ImportError: If LlamaIndex is not installed (with helpful message)
    """
    if not LLAMAINDEX_AVAILABLE:
        raise ImportError(
            "LlamaIndex integration requires the 'llama-index' package. "
            "Install with: pip install brokle-otel[llamaindex]"
            f"\nOriginal error: {_llamaindex_import_error}"
        )
    return True


__all__ = [
    # LangChain
    "BrokleLangChainCallback",
    "LANGCHAIN_AVAILABLE",
    "check_langchain_available",
    # LlamaIndex
    "BrokleLlamaIndexHandler",
    "set_global_handler",
    "LLAMAINDEX_AVAILABLE",
    "check_llamaindex_available",
]
