"""
Template Compiler

Mustache-style template compilation for prompts with variable extraction
and validation.
"""

import re
from typing import Dict, List, Set, Tuple, Union

from .types import (
    ChatMessage,
    ChatTemplate,
    Template,
    TextTemplate,
    Variables,
)

VARIABLE_PATTERN = re.compile(r"\{\{(\w+)\}\}")


def extract_variables(template: Template) -> List[str]:
    """
    Extract variable names from a template.

    Args:
        template: Template (text or chat)

    Returns:
        List of unique variable names
    """
    variables: Set[str] = set()

    if "content" in template:
        content = template.get("content", "")
        for match in VARIABLE_PATTERN.finditer(content):
            variables.add(match.group(1))
    elif "messages" in template:
        messages = template.get("messages", [])
        for msg in messages:
            content = msg.get("content", "")
            if content:
                for match in VARIABLE_PATTERN.finditer(content):
                    variables.add(match.group(1))

    return list(variables)


def _compile_string(content: str, variables: Variables) -> str:
    """
    Compile a string by replacing variables.

    Args:
        content: String with {{variable}} placeholders
        variables: Variable values

    Returns:
        Compiled string
    """
    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        if var_name in variables:
            return str(variables[var_name])
        return match.group(0)

    return VARIABLE_PATTERN.sub(replacer, content)


def compile_text_template(
    template: TextTemplate,
    variables: Variables
) -> TextTemplate:
    """
    Compile a text template.

    Args:
        template: Text template
        variables: Variable values

    Returns:
        Compiled text template
    """
    return {"content": _compile_string(template.get("content", ""), variables)}


def _compile_chat_message(
    message: ChatMessage,
    variables: Variables
) -> ChatMessage:
    """
    Compile a chat message.

    Args:
        message: Chat message
        variables: Variable values

    Returns:
        Compiled chat message
    """
    result = dict(message)
    if "content" in result:
        result["content"] = _compile_string(result["content"], variables)
    return result


def compile_chat_template(
    template: ChatTemplate,
    variables: Variables
) -> ChatTemplate:
    """
    Compile a chat template.

    Args:
        template: Chat template
        variables: Variable values

    Returns:
        Compiled chat template
    """
    messages = template.get("messages", [])
    return {
        "messages": [_compile_chat_message(msg, variables) for msg in messages]
    }


def compile_template(template: Template, variables: Variables) -> Template:
    """
    Compile any template type.

    Args:
        template: Template (text or chat)
        variables: Variable values

    Returns:
        Compiled template of the same type
    """
    if "content" in template:
        return compile_text_template(template, variables)
    return compile_chat_template(template, variables)


def validate_variables(
    template: Template,
    variables: Variables
) -> Tuple[List[str], bool]:
    """
    Validate that all required variables are provided.

    Args:
        template: Template with variables
        variables: Provided variables

    Returns:
        Tuple of (missing variables list, is_valid boolean)
    """
    required = set(extract_variables(template))
    provided = set(variables.keys())
    missing = list(required - provided)

    return missing, len(missing) == 0


def is_text_template(template: Template) -> bool:
    """Check if a template is a text template."""
    return "content" in template


def is_chat_template(template: Template) -> bool:
    """Check if a template is a chat template."""
    return "messages" in template


def get_compiled_content(template: TextTemplate, variables: Variables) -> str:
    """Get the content string from a text template after compilation."""
    return compile_text_template(template, variables)["content"]


def get_compiled_messages(
    template: ChatTemplate,
    variables: Variables
) -> List[ChatMessage]:
    """Get the messages array from a chat template after compilation."""
    return compile_chat_template(template, variables)["messages"]
