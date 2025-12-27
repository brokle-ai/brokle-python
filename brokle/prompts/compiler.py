"""
Template Compiler

Multi-dialect template compilation for prompts with variable extraction
and validation. Supports simple, Mustache, and Jinja2 dialects.
"""

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import chevron
from jinja2 import Environment, BaseLoader, select_autoescape

from .types import (
    ChatMessage,
    ChatTemplate,
    Template,
    TemplateDialect,
    TextTemplate,
    Variables,
)

# Configure Jinja2 for safe rendering (no filesystem access)
_jinja_env = Environment(
    loader=BaseLoader(),
    autoescape=False,  # Prompts are not HTML
)

# Regex patterns for dialect detection and variable extraction
PATTERNS = {
    # Simple: {{variable}}
    "simple": re.compile(r"\{\{(\w+)\}\}"),
    # Mustache sections: {{#section}}, {{^inverted}}, {{>partial}}
    "mustache_section": re.compile(r"\{\{[#^>\/](\w+)\}\}"),
    # Jinja2 blocks: {% if %}, {% for %}, {{ var|filter }}
    "jinja2_block": re.compile(
        r"\{%\s*(if|for|else|elif|endif|endfor|block|extends|include|macro|endmacro|set)\b"
    ),
    "jinja2_filter": re.compile(r"\{\{\s*\w+\s*\|"),
    # Jinja2 dot notation: {{ var.attr }} or {{ var.nested.attr }}
    "jinja2_dot_notation": re.compile(r"\{\{\s*\w+\.\w+"),
    # Mustache variable extraction (including sections)
    "mustache_vars": re.compile(r"\{\{([#^\/]?)(\w+)\}\}"),
    # Jinja2 variable extraction (captures full dot-notation path)
    "jinja2_vars": re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_.]*?)(?:\s*\|[^}]*)?\s*\}\}"),
    "jinja2_for_loop": re.compile(r"\{%\s*for\s+\w+\s+in\s+(\w+)"),
    "jinja2_condition": re.compile(r"\{%\s*(?:if|elif)\s+(\w+)"),
}


def detect_dialect(content: str) -> TemplateDialect:
    """
    Detect the template dialect from content.

    Detection order:
    1. Check for Jinja2 markers ({% ... %} or {{ var|filter }})
    2. Check for Mustache markers ({{#...}}, {{^...}}, {{>...}})
    3. Default to simple (just {{var}})

    Args:
        content: Template content string

    Returns:
        Detected dialect
    """
    # Check for Jinja2 first (more specific syntax)
    if (
        PATTERNS["jinja2_block"].search(content) or
        PATTERNS["jinja2_filter"].search(content) or
        PATTERNS["jinja2_dot_notation"].search(content)
    ):
        return TemplateDialect.JINJA2

    # Check for Mustache sections
    if PATTERNS["mustache_section"].search(content):
        return TemplateDialect.MUSTACHE

    # Default to simple
    return TemplateDialect.SIMPLE


def detect_template_dialect(template: Template) -> TemplateDialect:
    """
    Detect dialect from a template (text or chat).

    Args:
        template: Template to analyze

    Returns:
        Detected dialect
    """
    if "content" in template:
        return detect_dialect(template.get("content", ""))

    if "messages" in template:
        messages = template.get("messages", [])
        for msg in messages:
            content = msg.get("content", "")
            if content:
                dialect = detect_dialect(content)
                if dialect != TemplateDialect.SIMPLE:
                    return dialect

    return TemplateDialect.SIMPLE


def _extract_variables_from_string(
    content: str, dialect: TemplateDialect
) -> List[str]:
    """
    Extract variables from a string based on dialect.

    Args:
        content: String to extract variables from
        dialect: Template dialect

    Returns:
        List of unique variable names
    """
    variables: Set[str] = set()

    if dialect == TemplateDialect.SIMPLE:
        for match in PATTERNS["simple"].finditer(content):
            variables.add(match.group(1))

    elif dialect == TemplateDialect.MUSTACHE:
        for match in PATTERNS["mustache_vars"].finditer(content):
            # match.group(1) is the prefix (#, ^, /, or empty)
            # match.group(2) is the variable name
            if match.group(2):
                variables.add(match.group(2))

    elif dialect == TemplateDialect.JINJA2:
        # Extract {{ var }} and {{ var|filter }} and {{ var.attr }}
        for match in PATTERNS["jinja2_vars"].finditer(content):
            if match.group(1):
                var_path = match.group(1)
                # Extract root variable from dot-notation path
                root_var = var_path.split(".")[0]
                variables.add(root_var)

        # Extract {% for x in items %}
        for match in PATTERNS["jinja2_for_loop"].finditer(content):
            if match.group(1):
                variables.add(match.group(1))

        # Extract {% if condition %}
        for match in PATTERNS["jinja2_condition"].finditer(content):
            if match.group(1):
                variables.add(match.group(1))

    return list(variables)


def extract_variables(
    template: Template,
    dialect: TemplateDialect = TemplateDialect.AUTO
) -> List[str]:
    """
    Extract variable names from a template.

    Args:
        template: Template (text or chat)
        dialect: Template dialect (auto-detected if AUTO)

    Returns:
        List of unique variable names
    """
    resolved_dialect = (
        detect_template_dialect(template)
        if dialect == TemplateDialect.AUTO
        else dialect
    )
    variables: Set[str] = set()

    if "content" in template:
        content = template.get("content", "")
        for v in _extract_variables_from_string(content, resolved_dialect):
            variables.add(v)

    elif "messages" in template:
        messages = template.get("messages", [])
        for msg in messages:
            content = msg.get("content", "")
            if content:
                for v in _extract_variables_from_string(content, resolved_dialect):
                    variables.add(v)

    return list(variables)


def _compile_string(
    content: str,
    variables: Variables,
    dialect: TemplateDialect
) -> str:
    """
    Compile a string using the specified dialect.

    Args:
        content: String with template syntax
        variables: Variable values
        dialect: Template dialect

    Returns:
        Compiled string
    """
    if dialect == TemplateDialect.SIMPLE:
        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            if var_name in variables:
                value = variables[var_name]
                if value is None:
                    return ""
                if isinstance(value, (dict, list)):
                    return json.dumps(value)
                return str(value)
            return match.group(0)  # Preserve unmatched

        return PATTERNS["simple"].sub(replacer, content)

    if dialect == TemplateDialect.MUSTACHE:
        return chevron.render(content, variables)

    if dialect == TemplateDialect.JINJA2:
        template = _jinja_env.from_string(content)
        return template.render(**variables)

    # Fallback to simple
    return _compile_string(content, variables, TemplateDialect.SIMPLE)


def compile_text_template(
    template: TextTemplate,
    variables: Variables,
    dialect: TemplateDialect = TemplateDialect.AUTO
) -> TextTemplate:
    """
    Compile a text template.

    Args:
        template: Text template
        variables: Variable values
        dialect: Template dialect (auto-detected if AUTO)

    Returns:
        Compiled text template
    """
    resolved_dialect = (
        detect_dialect(template.get("content", ""))
        if dialect == TemplateDialect.AUTO
        else dialect
    )
    return {
        "content": _compile_string(
            template.get("content", ""), variables, resolved_dialect
        )
    }


def _compile_chat_message(
    message: ChatMessage,
    variables: Variables,
    dialect: TemplateDialect
) -> ChatMessage:
    """
    Compile a chat message.

    Args:
        message: Chat message
        variables: Variable values
        dialect: Template dialect

    Returns:
        Compiled chat message
    """
    result = dict(message)
    if "content" in result and result["content"]:
        result["content"] = _compile_string(result["content"], variables, dialect)
    return result


def compile_chat_template(
    template: ChatTemplate,
    variables: Variables,
    dialect: TemplateDialect = TemplateDialect.AUTO
) -> ChatTemplate:
    """
    Compile a chat template with support for placeholders.

    Placeholders allow injecting conversation history:
    - Message with type: "placeholder" and name: "history"
    - Variables include history: [{"role": "user", "content": "Hi"}, ...]
    - Placeholder is replaced with the array of messages

    Args:
        template: Chat template
        variables: Variable values (may include arrays for placeholders)
        dialect: Template dialect (auto-detected if AUTO)

    Returns:
        Compiled chat template
    """
    resolved_dialect = (
        detect_template_dialect(template)
        if dialect == TemplateDialect.AUTO
        else dialect
    )
    result: List[ChatMessage] = []

    for msg in template.get("messages", []):
        # Handle placeholder messages (for history injection)
        if msg.get("type") == "placeholder" and "name" in msg:
            placeholder_name = msg["name"]
            placeholder_value = variables.get(placeholder_name)

            if isinstance(placeholder_value, list):
                # Inject messages from the array
                for item in placeholder_value:
                    if (
                        isinstance(item, dict)
                        and "role" in item
                        and "content" in item
                    ):
                        result.append({
                            "role": item["role"],
                            "content": str(item["content"]),
                            **({"name": item["name"]} if "name" in item else {}),
                            **(
                                {"tool_call_id": item["tool_call_id"]}
                                if "tool_call_id" in item
                                else {}
                            ),
                        })
            # Skip placeholder if value is not a list
            continue

        # Regular message - compile content
        result.append(_compile_chat_message(msg, variables, resolved_dialect))

    return {"messages": result}


def compile_template(
    template: Template,
    variables: Variables,
    dialect: TemplateDialect = TemplateDialect.AUTO
) -> Template:
    """
    Compile any template type.

    Args:
        template: Template (text or chat)
        variables: Variable values
        dialect: Template dialect (auto-detected if AUTO)

    Returns:
        Compiled template of the same type
    """
    if "content" in template:
        return compile_text_template(template, variables, dialect)
    return compile_chat_template(template, variables, dialect)


def validate_variables(
    template: Template,
    variables: Variables,
    dialect: TemplateDialect = TemplateDialect.AUTO
) -> Tuple[List[str], bool]:
    """
    Validate that all required variables are provided.

    Args:
        template: Template with variables
        variables: Provided variables
        dialect: Template dialect (auto-detected if AUTO)

    Returns:
        Tuple of (missing variables list, is_valid boolean)
    """
    required = set(extract_variables(template, dialect))
    provided = set(variables.keys())
    missing = list(required - provided)

    return missing, len(missing) == 0


def is_text_template(template: Template) -> bool:
    """Check if a template is a text template."""
    return "content" in template


def is_chat_template(template: Template) -> bool:
    """Check if a template is a chat template."""
    return "messages" in template


def get_compiled_content(
    template: TextTemplate,
    variables: Variables,
    dialect: TemplateDialect = TemplateDialect.AUTO
) -> str:
    """
    Get the content string from a text template after compilation.

    Args:
        template: Text template
        variables: Variable values
        dialect: Template dialect (auto-detected if AUTO)

    Returns:
        Compiled content string
    """
    return compile_text_template(template, variables, dialect)["content"]


def get_compiled_messages(
    template: ChatTemplate,
    variables: Variables,
    dialect: TemplateDialect = TemplateDialect.AUTO
) -> List[ChatMessage]:
    """
    Get the messages array from a chat template after compilation.

    Args:
        template: Chat template
        variables: Variable values
        dialect: Template dialect (auto-detected if AUTO)

    Returns:
        Compiled messages array
    """
    return compile_chat_template(template, variables, dialect)["messages"]
