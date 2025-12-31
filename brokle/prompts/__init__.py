"""
Brokle Prompts Manager

Provides prompt management functionality accessed via client.prompts.

Example (Sync):
    >>> from brokle import Brokle
    >>> with Brokle(api_key="bk_...") as client:
    ...     prompt = client.prompts.get("greeting", label="production")
    ...     messages = prompt.to_openai_messages({"name": "Alice"})

Example (Async):
    >>> from brokle import AsyncBrokle
    >>> async with AsyncBrokle(api_key="bk_...") as client:
    ...     prompt = await client.prompts.get("greeting", label="production")
    ...     messages = prompt.to_openai_messages({"name": "Alice"})
"""

from ._managers import AsyncPromptManager, PromptManager
from .cache import CacheOptions, PromptCache
from .compiler import (
    compile_chat_template,
    compile_template,
    compile_text_template,
    detect_dialect,
    detect_template_dialect,
    extract_variables,
    get_compiled_content,
    get_compiled_messages,
    is_chat_template,
    is_text_template,
    validate_variables,
)
from .exceptions import (
    PromptCompileError,
    PromptError,
    PromptFetchError,
    PromptNotFoundError,
)
from .prompt import Prompt
from .types import (
    AnthropicMessage,
    AnthropicRequest,
    CacheEntry,
    ChatFallback,
    ChatMessage,
    ChatTemplate,
    Fallback,
    GetPromptOptions,
    ListPromptsOptions,
    MessageRole,
    ModelConfig,
    OpenAIMessage,
    PaginatedResponse,
    Pagination,
    PromptConfig,
    PromptData,
    PromptSummary,
    PromptType,
    PromptVersion,
    Template,
    TemplateDialect,
    TextFallback,
    TextTemplate,
    UpsertPromptRequest,
    Variables,
    VariableValue,
)

__all__ = [
    # Manager classes
    "AsyncPromptManager",
    "PromptManager",
    # Core classes
    "Prompt",
    "PromptCache",
    "CacheOptions",
    # Exceptions
    "PromptError",
    "PromptNotFoundError",
    "PromptCompileError",
    "PromptFetchError",
    # Compiler functions
    "extract_variables",
    "compile_template",
    "compile_text_template",
    "compile_chat_template",
    "validate_variables",
    "is_text_template",
    "is_chat_template",
    "get_compiled_content",
    "get_compiled_messages",
    "detect_dialect",
    "detect_template_dialect",
    # Types
    "PromptType",
    "TemplateDialect",
    "MessageRole",
    "ChatMessage",
    "TextTemplate",
    "ChatTemplate",
    "Template",
    "ModelConfig",
    "PromptConfig",
    "PromptVersion",
    "PromptData",
    "PromptSummary",
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
    "VariableValue",
    "Fallback",
    "TextFallback",
    "ChatFallback",
]
