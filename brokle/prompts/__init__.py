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
from .prompt import Prompt

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
    PromptSummary,
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
    # Manager classes
    "AsyncPromptManager",
    "PromptManager",
    # Core classes
    "Prompt",
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
    "FallbackConfig",
]
