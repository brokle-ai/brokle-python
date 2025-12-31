"""
Tests for prompt fallback/guaranteed availability functionality.

Ensures prompts are always available even during network failures through fallback mechanisms.
"""

import pytest

from brokle.prompts import (
    Prompt,
    PromptType,
)
from brokle.prompts.types import PromptData


class TestPromptCreateFallback:
    """Tests for Prompt.create_fallback() method."""

    def test_create_fallback_text_from_string(self):
        """String fallback creates TEXT type prompt."""
        fallback = "Hello {{name}}, welcome to our service!"

        prompt = Prompt.create_fallback("greeting", fallback)

        assert prompt.name == "greeting"
        assert prompt.type == PromptType.TEXT
        assert prompt.is_fallback is True
        assert prompt.template == {"content": fallback}
        assert "name" in prompt.variables

    def test_create_fallback_chat_from_list(self):
        """List fallback creates CHAT type prompt."""
        fallback = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "{{user_query}}"},
        ]

        prompt = Prompt.create_fallback("assistant", fallback)

        assert prompt.name == "assistant"
        assert prompt.type == PromptType.CHAT
        assert prompt.is_fallback is True
        assert prompt.template == {"messages": fallback}
        assert "user_query" in prompt.variables

    def test_create_fallback_invalid_type_raises(self):
        """Invalid fallback type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid fallback type"):
            Prompt.create_fallback("test", 12345)  # type: ignore

        with pytest.raises(ValueError, match="Invalid fallback type"):
            Prompt.create_fallback("test", {"invalid": "dict"})  # type: ignore

    def test_create_fallback_version_is_zero(self):
        """Fallback prompts have version 0."""
        prompt = Prompt.create_fallback("test", "Hello!")

        assert prompt.version == 0

    def test_create_fallback_id_is_fallback(self):
        """Fallback prompts have id='fallback'."""
        prompt = Prompt.create_fallback("test", "Hello!")

        assert prompt.id == "fallback"


class TestFallbackTemplateCompilation:
    """Tests for template compilation on fallback prompts."""

    def test_text_fallback_compiles_with_variables(self):
        """Text fallback templates compile correctly with variables."""
        fallback = "Hello {{name}}, your order #{{order_id}} is ready!"
        prompt = Prompt.create_fallback("notification", fallback)

        compiled = prompt.compile({"name": "Alice", "order_id": "12345"})

        assert compiled["content"] == "Hello Alice, your order #12345 is ready!"

    def test_chat_fallback_compiles_with_variables(self):
        """Chat fallback templates compile correctly with variables."""
        fallback = [
            {"role": "system", "content": "You help with {{topic}}."},
            {"role": "user", "content": "{{question}}"},
        ]
        prompt = Prompt.create_fallback("qa", fallback)

        compiled = prompt.compile(
            {"topic": "Python", "question": "How do decorators work?"}
        )

        assert compiled["messages"][0]["content"] == "You help with Python."
        assert compiled["messages"][1]["content"] == "How do decorators work?"

    def test_text_fallback_to_openai_messages(self):
        """Text fallback converts to OpenAI messages format."""
        fallback = "Hello {{name}}!"
        prompt = Prompt.create_fallback("greeting", fallback)

        messages = prompt.to_openai_messages({"name": "World"})

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello World!"

    def test_chat_fallback_to_openai_messages(self):
        """Chat fallback converts to OpenAI messages format."""
        fallback = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello {{name}}!"},
        ]
        prompt = Prompt.create_fallback("assistant", fallback)

        messages = prompt.to_openai_messages({"name": "there"})

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["content"] == "Hello there!"


class TestFallbackTypeDetection:
    """Tests for automatic type detection from fallback content."""

    def test_string_detected_as_text(self):
        """String fallback is detected as TEXT type."""
        prompt = Prompt.create_fallback("test", "Simple string")

        assert prompt.is_text() is True
        assert prompt.is_chat() is False

    def test_list_detected_as_chat(self):
        """List fallback is detected as CHAT type."""
        prompt = Prompt.create_fallback("test", [{"role": "user", "content": "Hello"}])

        assert prompt.is_chat() is True
        assert prompt.is_text() is False

    def test_empty_string_works(self):
        """Empty string is valid text fallback."""
        prompt = Prompt.create_fallback("empty", "")

        assert prompt.type == PromptType.TEXT
        assert prompt.template["content"] == ""

    def test_empty_list_works(self):
        """Empty list is valid chat fallback."""
        prompt = Prompt.create_fallback("empty", [])

        assert prompt.type == PromptType.CHAT
        assert prompt.template["messages"] == []


class TestIsFallbackProperty:
    """Tests for is_fallback property."""

    def test_fallback_prompt_has_is_fallback_true(self):
        """Fallback prompts have is_fallback=True."""
        prompt = Prompt.create_fallback("test", "Fallback content")

        assert prompt.is_fallback is True

    def test_normal_prompt_has_is_fallback_false(self):
        """Normal prompts from API have is_fallback=False."""
        # Simulate a prompt from API response
        data = PromptData(
            id="123",
            project_id="proj-1",
            name="test",
            type=PromptType.TEXT,
            description="Test prompt",
            tags=[],
            template={"content": "Hello!"},
            config=None,
            variables=[],
            labels=[],
            version=1,
            is_fallback=False,
            commit_message="Initial",
            created_by="user",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )
        prompt = Prompt.from_data(data)

        assert prompt.is_fallback is False


class TestFallbackVariableExtraction:
    """Tests for variable extraction from fallback templates."""

    def test_extracts_text_variables(self):
        """Variables are extracted from text fallback templates."""
        fallback = "Hello {{name}}, your code is {{code}}."
        prompt = Prompt.create_fallback("test", fallback)

        assert set(prompt.variables) == {"name", "code"}

    def test_extracts_chat_variables(self):
        """Variables are extracted from chat fallback messages."""
        fallback = [
            {"role": "system", "content": "You assist with {{domain}}."},
            {"role": "user", "content": "Question: {{query}}"},
        ]
        prompt = Prompt.create_fallback("test", fallback)

        assert set(prompt.variables) == {"domain", "query"}

    def test_no_variables_in_static_template(self):
        """Templates without variables have empty variables list."""
        prompt = Prompt.create_fallback("static", "Hello World!")

        assert prompt.variables == []

    def test_duplicate_variables_deduplicated(self):
        """Duplicate variables are deduplicated."""
        fallback = "Hello {{name}}, welcome {{name}}!"
        prompt = Prompt.create_fallback("test", fallback)

        assert prompt.variables == ["name"]
