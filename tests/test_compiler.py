"""Tests for template compiler dot notation handling."""
from brokle.prompts.compiler import (
    detect_dialect,
    extract_variables,
)
from brokle.prompts.types import TemplateDialect


class TestJinja2DotNotation:
    """Tests for Jinja2 dot notation detection and extraction."""

    def test_detect_dialect_dot_notation(self):
        """Dot notation should be detected as Jinja2."""
        assert detect_dialect("{{ user.name }}") == TemplateDialect.JINJA2
        assert detect_dialect("{{ data.items.count }}") == TemplateDialect.JINJA2

    def test_extract_variables_dot_notation_explicit(self):
        """Explicit Jinja2 dialect extracts root variable."""
        template = {"content": "Hello {{ user.name }}"}
        vars = extract_variables(template, TemplateDialect.JINJA2)
        assert "user" in vars

    def test_extract_variables_dot_notation_auto(self):
        """AUTO dialect detection + extraction works for dot notation."""
        template = {"content": "Hello {{ user.name }}, email: {{ user.email }}"}
        vars = extract_variables(template, TemplateDialect.AUTO)
        assert "user" in vars

    def test_extract_variables_nested_dot_notation(self):
        """Nested dot paths extract root variable."""
        template = {"content": "{{ data.items.first.value }}"}
        vars = extract_variables(template, TemplateDialect.AUTO)
        assert "data" in vars

    def test_extract_variables_mixed(self):
        """Mix of simple and dot notation extracts all roots."""
        template = {"content": "{{ name }} and {{ user.email }}"}
        vars = extract_variables(template, TemplateDialect.AUTO)
        assert "name" in vars
        assert "user" in vars

    def test_extract_variables_with_filter(self):
        """Dot notation with filter extracts root variable."""
        template = {"content": "{{ user.name|upper }}"}
        vars = extract_variables(template, TemplateDialect.JINJA2)
        assert "user" in vars

    def test_chat_template_dot_notation(self):
        """Chat template with dot notation extracts root variables."""
        template = {
            "messages": [
                {"role": "system", "content": "You are {{ config.assistant_name }}."},
                {"role": "user", "content": "Hello {{ user.name }}!"},
            ]
        }
        vars = extract_variables(template, TemplateDialect.AUTO)
        assert "config" in vars
        assert "user" in vars
