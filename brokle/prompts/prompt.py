"""
Prompt Class

Represents a fetched prompt with methods for compilation and
conversion to various LLM provider formats.
"""

from typing import Any, Dict, List, Optional

from .compiler import (
    compile_chat_template,
    compile_template,
    compile_text_template,
    extract_variables,
    is_chat_template,
    is_text_template,
    validate_variables,
)
from .exceptions import PromptCompileError
from .types import (
    AnthropicMessage,
    AnthropicRequest,
    ChatMessage,
    Fallback,
    ModelConfig,
    OpenAIMessage,
    PromptData,
    PromptType,
    Template,
    Variables,
)


class Prompt:
    """
    Prompt class with compilation and provider conversion methods.

    Example:
        >>> prompt = await client.get("greeting", label="production")
        >>> compiled = prompt.compile({"name": "Alice"})
        >>> messages = prompt.to_openai_messages({"name": "Alice"})
    """

    def __init__(self, data: PromptData):
        """
        Initialize Prompt from PromptData.

        Args:
            data: PromptData from API response
        """
        self.id = data.id
        self.name = data.name
        self.type = data.type
        self.description = data.description
        self.tags = data.tags
        self.template = data.template
        self.config = data.config
        self.variables = data.variables
        self.labels = data.labels
        self.version = data.version
        self.is_fallback = data.is_fallback
        self.commit_message = data.commit_message
        self.created_at = data.created_at

    def compile(self, variables: Optional[Variables] = None) -> Template:
        """
        Compile the template with provided variables.

        Args:
            variables: Variable values for interpolation

        Returns:
            Compiled template

        Raises:
            PromptCompileError: If required variables are missing
        """
        vars_dict = variables or {}
        missing, is_valid = validate_variables(self.template, vars_dict)
        if not is_valid:
            raise PromptCompileError(
                f"Missing required variables: {', '.join(missing)}",
                missing_variables=missing,
            )
        return compile_template(self.template, vars_dict)

    def compile_text(self, variables: Optional[Variables] = None) -> str:
        """
        Get the compiled content as a string (text templates only).

        Args:
            variables: Variable values for interpolation

        Returns:
            Compiled content string

        Raises:
            PromptCompileError: If prompt is not a text template or variables are missing
        """
        if not is_text_template(self.template):
            raise PromptCompileError(
                "compile_text() can only be used with text templates"
            )
        vars_dict = variables or {}
        missing, is_valid = validate_variables(self.template, vars_dict)
        if not is_valid:
            raise PromptCompileError(
                f"Missing required variables: {', '.join(missing)}",
                missing_variables=missing,
            )
        return compile_text_template(self.template, vars_dict)["content"]

    def compile_chat(self, variables: Optional[Variables] = None) -> List[ChatMessage]:
        """
        Get the compiled messages array (chat templates only).

        Args:
            variables: Variable values for interpolation

        Returns:
            Compiled messages array

        Raises:
            PromptCompileError: If prompt is not a chat template or variables are missing
        """
        if not is_chat_template(self.template):
            raise PromptCompileError(
                "compile_chat() can only be used with chat templates"
            )
        vars_dict = variables or {}
        missing, is_valid = validate_variables(self.template, vars_dict)
        if not is_valid:
            raise PromptCompileError(
                f"Missing required variables: {', '.join(missing)}",
                missing_variables=missing,
            )
        return compile_chat_template(self.template, vars_dict)["messages"]

    def get_raw_template(self) -> Template:
        """
        Get the raw template without compilation.

        Returns:
            The raw template (TextTemplate or ChatTemplate)
        """
        return self.template

    def to_openai_messages(
        self, variables: Optional[Variables] = None
    ) -> List[OpenAIMessage]:
        """
        Convert to OpenAI messages format.

        For text templates, returns a single user message.
        For chat templates, maps messages directly.

        Args:
            variables: Variable values for interpolation

        Returns:
            OpenAI-compatible messages array
        """
        compiled = self.compile(variables)

        if is_text_template(compiled):
            return [{"role": "user", "content": compiled["content"]}]

        messages = compiled.get("messages", [])
        result: List[OpenAIMessage] = []
        for msg in messages:
            openai_msg: OpenAIMessage = {
                "role": msg["role"],
                "content": msg.get("content", ""),
            }
            if msg.get("name"):
                openai_msg["name"] = msg["name"]
            if msg.get("tool_call_id"):
                openai_msg["tool_call_id"] = msg["tool_call_id"]
            result.append(openai_msg)
        return result

    def to_anthropic_messages(
        self, variables: Optional[Variables] = None
    ) -> AnthropicRequest:
        """
        Convert to Anthropic messages format.

        Anthropic requires system messages to be separate.
        Returns an object with system prompt and messages array.

        Args:
            variables: Variable values for interpolation

        Returns:
            Anthropic-compatible request structure
        """
        compiled = self.compile(variables)

        if is_text_template(compiled):
            return AnthropicRequest(
                messages=[{"role": "user", "content": compiled["content"]}]
            )

        messages = compiled.get("messages", [])
        system_messages = [m for m in messages if m.get("role") == "system"]
        other_messages = [m for m in messages if m.get("role") != "system"]

        system_prompt = "\n\n".join(m.get("content", "") for m in system_messages)

        anthropic_messages: List[AnthropicMessage] = []
        for msg in other_messages:
            role = msg.get("role", "")
            if role in ("user", "assistant"):
                anthropic_messages.append(
                    {
                        "role": role,
                        "content": msg.get("content", ""),
                    }
                )

        return AnthropicRequest(
            system=system_prompt if system_prompt else None,
            messages=anthropic_messages,
        )

    def to_langchain(self, variables: Optional[Variables] = None):
        """
        Convert to LangChain messages format.

        Requires langchain-core to be installed.

        Args:
            variables: Variable values for interpolation

        Returns:
            List of LangChain message objects
        """
        try:
            from langchain_core.messages import (
                AIMessage,
                HumanMessage,
                SystemMessage,
            )
        except ImportError:
            raise ImportError(
                "langchain-core is required for to_langchain(). "
                "Install with: pip install langchain-core"
            )

        compiled = self.compile(variables)

        if is_text_template(compiled):
            return [HumanMessage(content=compiled["content"])]

        messages = compiled.get("messages", [])
        result = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                result.append(SystemMessage(content=content))
            elif role == "user":
                result.append(HumanMessage(content=content))
            elif role == "assistant":
                result.append(AIMessage(content=content))
        return result

    def to_llama_index(self, variables: Optional[Variables] = None):
        """
        Convert to LlamaIndex prompt template.

        Requires llama-index to be installed.

        Args:
            variables: Variable values for interpolation

        Returns:
            LlamaIndex PromptTemplate (for text) or ChatPromptTemplate (for chat)
        """
        try:
            from llama_index.core.base.llms.types import ChatMessage as LlamaMessage
            from llama_index.core.base.llms.types import (
                MessageRole,
            )
            from llama_index.core.prompts import (
                ChatPromptTemplate,
                PromptTemplate,
            )
        except ImportError:
            raise ImportError(
                "llama-index is required for to_llama_index(). "
                "Install with: pip install llama-index"
            )

        compiled = self.compile(variables)

        if is_text_template(compiled):
            return PromptTemplate(compiled["content"])

        messages = compiled.get("messages", [])
        llama_messages = []

        role_map = {
            "system": MessageRole.SYSTEM,
            "user": MessageRole.USER,
            "assistant": MessageRole.ASSISTANT,
        }

        for msg in messages:
            role_str = msg.get("role", "user")
            role = role_map.get(role_str, MessageRole.USER)
            llama_messages.append(
                LlamaMessage(role=role, content=msg.get("content", ""))
            )

        return ChatPromptTemplate(message_templates=llama_messages)

    def get_model_config(self, overrides: Optional[ModelConfig] = None) -> ModelConfig:
        """
        Get model configuration with optional overrides.

        Args:
            overrides: Optional config overrides

        Returns:
            Merged model config
        """
        config = dict(self.config or {})
        if overrides:
            config.update(overrides)
        return config

    def has_variable(self, variable_name: str) -> bool:
        """
        Check if a variable is required.

        Args:
            variable_name: Variable name to check

        Returns:
            True if variable is in the template
        """
        return variable_name in self.variables

    def get_missing_variables(self, variables: Variables) -> List[str]:
        """
        Get missing variables for a given set of values.

        Args:
            variables: Provided variable values

        Returns:
            List of missing variable names
        """
        missing, _ = validate_variables(self.template, variables)
        return missing

    def validate_variables(self, variables: Variables) -> bool:
        """
        Check if all required variables are provided.

        Args:
            variables: Provided variable values

        Returns:
            True if all variables are provided
        """
        _, is_valid = validate_variables(self.template, variables)
        return is_valid

    def is_text(self) -> bool:
        """Check if this is a text template."""
        return self.type == PromptType.TEXT

    def is_chat(self) -> bool:
        """Check if this is a chat template."""
        return self.type == PromptType.CHAT

    def __str__(self) -> str:
        """Get a formatted string representation."""
        return f"Prompt({self.name} v{self.version} [{self.type.value}])"

    def __repr__(self) -> str:
        """Get a detailed string representation."""
        return (
            f"Prompt(id={self.id!r}, name={self.name!r}, "
            f"type={self.type.value!r}, version={self.version})"
        )

    @classmethod
    def from_data(cls, data: PromptData) -> "Prompt":
        """Create a Prompt from PromptData."""
        return cls(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Prompt":
        """Create a Prompt from API response dictionary."""
        return cls(PromptData.from_dict(data))

    @classmethod
    def create_fallback(
        cls,
        name: str,
        fallback: Fallback,
    ) -> "Prompt":
        """
        Create a fallback Prompt when fetch fails.

        Auto-detects prompt type from fallback content:
        - String → TEXT template
        - List[ChatMessage] → CHAT template

        Args:
            name: Prompt name
            fallback: Fallback content - string for text, list of messages for chat

        Returns:
            Fallback Prompt instance

        Raises:
            ValueError: If fallback is not a string or list
        """
        from datetime import datetime, timezone

        if isinstance(fallback, str):
            template: Template = {"content": fallback}
            prompt_type = PromptType.TEXT
        elif isinstance(fallback, list):
            template = {"messages": fallback}
            prompt_type = PromptType.CHAT
        else:
            raise ValueError(
                f"Invalid fallback type: {type(fallback).__name__}. "
                "Expected str for text prompts or List[ChatMessage] for chat prompts."
            )

        now = datetime.now(timezone.utc).isoformat()
        data = PromptData(
            id="fallback",
            project_id="",
            name=name,
            type=prompt_type,
            description="Fallback prompt",
            tags=[],
            template=template,
            config=None,
            variables=extract_variables(template),
            labels=[],
            version=0,
            is_fallback=True,
            commit_message="",
            created_by="",
            created_at=now,
            updated_at=now,
        )
        return cls(data)
