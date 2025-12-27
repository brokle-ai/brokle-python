"""
Prompt Management Types

Type definitions for the Brokle Prompt Management system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union


class PromptType(str, Enum):
    """Prompt type - text for simple templates, chat for message arrays."""
    TEXT = "text"
    CHAT = "chat"


class TemplateDialect(str, Enum):
    """
    Template dialect for rendering.

    - SIMPLE: Basic {{variable}} substitution only
    - MUSTACHE: Full Mustache support with sections, loops, partials
    - JINJA2: Jinja2 with filters, conditionals, loops
    - AUTO: Auto-detect dialect from template syntax
    """
    SIMPLE = "simple"
    MUSTACHE = "mustache"
    JINJA2 = "jinja2"
    AUTO = "auto"


class MessageRole(str, Enum):
    """Message role in a chat template."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatMessage(TypedDict, total=False):
    """Chat message structure."""
    role: str
    content: str
    name: Optional[str]
    tool_call_id: Optional[str]


class TextTemplate(TypedDict):
    """Text template structure."""
    content: str


class ChatTemplate(TypedDict):
    """Chat template structure."""
    messages: List[ChatMessage]


Template = Union[TextTemplate, ChatTemplate]


class ModelConfig(TypedDict, total=False):
    """Model configuration for LLM execution."""
    model: Optional[str]
    temperature: Optional[float]
    max_tokens: Optional[int]
    top_p: Optional[float]
    frequency_penalty: Optional[float]
    presence_penalty: Optional[float]
    stop: Optional[List[str]]


@dataclass
class PromptVersion:
    """Prompt version data."""
    id: str
    prompt_id: str
    version: int
    template: Template
    config: Optional[ModelConfig]
    variables: List[str]
    labels: List[str]
    commit_message: str
    created_by: str
    created_at: str
    dialect: Optional[TemplateDialect] = None


@dataclass
class PromptData:
    """Full prompt data from API."""
    id: str
    project_id: str
    name: str
    type: PromptType
    description: str
    tags: List[str]
    template: Template
    config: Optional[ModelConfig]
    variables: List[str]
    labels: List[str]
    version: int
    is_fallback: bool
    commit_message: str
    created_by: str
    created_at: str
    updated_at: str
    dialect: Optional[TemplateDialect] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptData":
        """Create from API response dictionary."""
        dialect_val = data.get("dialect")
        dialect = TemplateDialect(dialect_val) if dialect_val else None
        return cls(
            id=data["id"],
            project_id=data["project_id"],
            name=data["name"],
            type=PromptType(data["type"]),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            template=data["template"],
            config=data.get("config"),
            variables=data.get("variables", []),
            labels=data.get("labels", []),
            version=data["version"],
            is_fallback=data.get("is_fallback", False),
            commit_message=data.get("commit_message", ""),
            created_by=data.get("created_by", ""),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            dialect=dialect,
        )


@dataclass
class PromptSummary:
    """Lightweight prompt summary from list endpoint."""

    id: str
    name: str
    type: PromptType
    description: str
    tags: List[str]
    labels: Dict[str, int]  # label_name -> version
    latest_version: int
    created_at: str
    updated_at: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptSummary":
        """Create from API list response."""
        # Convert labels array [{name, version}] to dict {name: version}
        labels_list = data.get("labels", [])
        labels_dict = {label["name"]: label["version"] for label in labels_list}

        return cls(
            id=data["id"],
            name=data["name"],
            type=PromptType(data["type"]),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            labels=labels_dict,
            latest_version=data["latest_version"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )


@dataclass
class GetPromptOptions:
    """Options for fetching a prompt."""
    label: Optional[str] = None
    version: Optional[int] = None
    cache_ttl: int = 60  # seconds
    force_refresh: bool = False


@dataclass
class ListPromptsOptions:
    """Options for listing prompts."""
    type: Optional[PromptType] = None
    tags: Optional[List[str]] = None
    search: Optional[str] = None
    page: int = 1
    limit: int = 20


@dataclass
class Pagination:
    """Pagination info."""
    total: int
    page: int
    limit: int
    pages: int


@dataclass
class PaginatedResponse:
    """Paginated response for prompt lists."""
    data: List["PromptSummary"]
    pagination: Pagination


@dataclass
class UpsertPromptRequest:
    """Request to create or update a prompt."""
    name: str
    type: PromptType
    template: Template
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    config: Optional[ModelConfig] = None
    commit_message: Optional[str] = None
    labels: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API request dictionary."""
        result = {
            "name": self.name,
            "type": self.type.value if isinstance(self.type, PromptType) else self.type,
            "template": self.template,
        }
        if self.description:
            result["description"] = self.description
        if self.tags:
            result["tags"] = self.tags
        if self.config:
            result["config"] = self.config
        if self.commit_message:
            result["commit_message"] = self.commit_message
        if self.labels:
            result["labels"] = self.labels
        return result


@dataclass
class CacheEntry:
    """Cache entry for prompts."""
    data: PromptData
    fetched_at: float
    ttl: int


TextFallback = str
ChatFallback = List[ChatMessage]
Fallback = Union[TextFallback, ChatFallback]


@dataclass
class PromptConfig:
    """
    Prompt client configuration.

    Controls caching behavior and retry settings for prompt operations.

    Attributes:
        cache_enabled: Whether to enable caching (default: True)
        cache_ttl_seconds: Default TTL for cached prompts in seconds (default: 60)
        cache_max_size: Maximum number of prompts to cache (default: 1000)
        max_retries: Maximum number of retry attempts for failed requests (default: 2)
        retry_delay: Base delay between retries in seconds (default: 1.0)
    """
    cache_enabled: bool = True
    cache_ttl_seconds: int = 60
    cache_max_size: int = 1000
    max_retries: int = 2
    retry_delay: float = 1.0


class OpenAIMessage(TypedDict, total=False):
    """OpenAI message format."""
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: Optional[str]
    tool_call_id: Optional[str]


class AnthropicMessage(TypedDict):
    """Anthropic message format."""
    role: Literal["user", "assistant"]
    content: str


@dataclass
class AnthropicRequest:
    """Anthropic request structure with system prompt."""
    messages: List[AnthropicMessage]
    system: Optional[str] = None


# VariableValue can be primitives, lists, or nested dicts
# This enables Mustache sections ({{#items}}...{{/items}}) and history injection
VariableValue = Union[
    str,
    int,
    float,
    bool,
    None,
    List[Any],  # For arrays (history injection, Mustache loops)
    Dict[str, Any],  # For nested access ({{user.name}})
]

Variables = Dict[str, VariableValue]
"""
Variables object for template compilation.

Supports:
- Primitives: str, int, float, bool
- Arrays: for Mustache sections ({{#items}}...{{/items}}) or history injection
- Dicts: for nested access ({{user.name}})

Example:
    variables: Variables = {
        "name": "Alice",
        "count": 42,
        "premium": True,
        # Array for Mustache loop or history injection
        "history": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ],
        # Nested dict
        "user": {"name": "Alice", "email": "alice@example.com"}
    }
"""
