"""
Prompt Management Module

Centralized prompt storage with versioning, labels, and caching.

Example:
    >>> from brokle.prompt import PromptClient, Prompt
    >>>
    >>> client = PromptClient(
    ...     api_key="bk_...",
    ...     base_url="https://api.brokle.ai"
    ... )
    >>>
    >>> # Fetch a prompt (async)
    >>> prompt = await client.get("greeting", label="production")
    >>>
    >>> # Compile with variables
    >>> compiled = prompt.compile({"name": "Alice"})
    >>>
    >>> # Convert to OpenAI format
    >>> messages = prompt.to_openai_messages({"name": "Alice"})
"""

from .prompt import Prompt
from .client import PromptClient
from .cache import PromptCache, CacheOptions

from .exceptions import (
    PromptError,
    PromptNotFoundError,
    PromptCompileError,
    PromptFetchError,
)

from .compiler import (
    extract_variables,
    compile_template,
    compile_text_template,
    compile_chat_template,
    validate_variables,
    is_text_template,
    is_chat_template,
    get_compiled_content,
    get_compiled_messages,
)

from .types import (
    PromptType,
    MessageRole,
    ChatMessage,
    TextTemplate,
    ChatTemplate,
    Template,
    ModelConfig,
    PromptConfig,
    PromptVersion,
    PromptData,
    GetPromptOptions,
    ListPromptsOptions,
    Pagination,
    PaginatedResponse,
    UpsertPromptRequest,
    CacheEntry,
    OpenAIMessage,
    AnthropicMessage,
    AnthropicRequest,
    Variables,
    FallbackConfig,
)

__all__ = [
    "Prompt",
    "PromptClient",
    "PromptCache",
    "CacheOptions",
    "PromptError",
    "PromptNotFoundError",
    "PromptCompileError",
    "PromptFetchError",
    "extract_variables",
    "compile_template",
    "compile_text_template",
    "compile_chat_template",
    "validate_variables",
    "is_text_template",
    "is_chat_template",
    "get_compiled_content",
    "get_compiled_messages",
    "PromptType",
    "MessageRole",
    "ChatMessage",
    "TextTemplate",
    "ChatTemplate",
    "Template",
    "ModelConfig",
    "PromptConfig",
    "PromptVersion",
    "PromptData",
    "GetPromptOptions",
    "ListPromptsOptions",
    "Pagination",
    "PaginatedResponse",
    "UpsertPromptRequest",
    "CacheEntry",
    "OpenAIMessage",
    "AnthropicMessage",
    "AnthropicRequest",
    "Variables",
    "FallbackConfig",
]
