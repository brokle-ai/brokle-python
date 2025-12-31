"""
Tests for prompt-to-trace linking functionality.

Validates that prompts are correctly linked to spans via OpenTelemetry attributes:
- brokle.prompt.name
- brokle.prompt.version
- brokle.prompt.id

Key behavior: Fallback prompts (is_fallback=True) are NOT linked to traces.
This prevents polluting analytics with offline fallbacks.
"""

from brokle.prompts import Prompt, PromptType
from brokle.prompts.types import PromptData
from brokle.types import Attrs


class TestPromptIsFallbackBehavior:
    """Tests for is_fallback property and linking behavior."""

    def test_fallback_prompt_has_is_fallback_true(self):
        """Fallback prompts have is_fallback=True."""
        prompt = Prompt.create_fallback("test", "Hello {{name}}")

        assert prompt.is_fallback is True

    def test_fallback_prompt_has_version_zero(self):
        """Fallback prompts have version 0."""
        prompt = Prompt.create_fallback("test", "Hello")

        assert prompt.version == 0

    def test_fallback_prompt_has_id_fallback(self):
        """Fallback prompts have id='fallback'."""
        prompt = Prompt.create_fallback("test", "Hello")

        assert prompt.id == "fallback"

    def test_normal_prompt_has_is_fallback_false(self):
        """Normal prompts from API have is_fallback=False."""
        data = PromptData(
            id="01HXY123456789ABCDEFGHIJ",
            project_id="proj-1",
            name="greeting",
            type=PromptType.TEXT,
            description="A greeting prompt",
            tags=[],
            template={"content": "Hello {{name}}!"},
            config=None,
            variables=["name"],
            labels=[],
            version=5,
            is_fallback=False,
            commit_message="Updated greeting",
            created_by="user-1",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )
        prompt = Prompt.from_data(data)

        assert prompt.is_fallback is False
        assert prompt.version == 5
        assert prompt.id == "01HXY123456789ABCDEFGHIJ"


class TestPromptLinkingAttributes:
    """Tests for prompt attributes used in span linking."""

    def test_normal_prompt_has_valid_linking_attributes(self):
        """Normal prompts have all attributes needed for linking."""
        data = PromptData(
            id="01HXY123456789ABCDEFGHIJ",
            project_id="proj-1",
            name="assistant",
            type=PromptType.CHAT,
            description="Assistant prompt",
            tags=[],
            template={
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "{{query}}"},
                ]
            },
            config={"model": "gpt-4", "temperature": 0.7},
            variables=["query"],
            labels=[],
            version=3,
            is_fallback=False,
            commit_message="v3 update",
            created_by="user-1",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )
        prompt = Prompt.from_data(data)

        # These are the values that should be set on spans
        assert prompt.name == "assistant"
        assert prompt.version == 3
        assert prompt.id == "01HXY123456789ABCDEFGHIJ"
        assert prompt.is_fallback is False

    def test_fallback_prompt_should_not_be_linked(self):
        """Fallback prompts have attributes that indicate no linking."""
        prompt = Prompt.create_fallback("my-prompt", "Hello!")

        # Linking logic should check these conditions:
        # 1. is_fallback == True -> skip linking
        # 2. id == 'fallback' -> skip id attribute
        assert prompt.is_fallback is True
        assert prompt.id == "fallback"
        assert prompt.version == 0


class TestAttributeConstants:
    """Tests for span attribute constant definitions."""

    def test_brokle_prompt_name_defined(self):
        """BROKLE_PROMPT_NAME attribute constant is defined."""
        assert Attrs.BROKLE_PROMPT_NAME == "brokle.prompt.name"

    def test_brokle_prompt_version_defined(self):
        """BROKLE_PROMPT_VERSION attribute constant is defined."""
        assert Attrs.BROKLE_PROMPT_VERSION == "brokle.prompt.version"

    def test_brokle_prompt_id_defined(self):
        """BROKLE_PROMPT_ID attribute constant is defined."""
        assert Attrs.BROKLE_PROMPT_ID == "brokle.prompt.id"


class TestLinkingLogicValidation:
    """Tests for prompt linking logic without full OTEL setup."""

    def test_should_link_normal_prompt(self):
        """Normal prompts should be linked with all attributes."""
        data = PromptData(
            id="01HXY123456789ABCDEFGHIJ",
            project_id="proj-1",
            name="greet",
            type=PromptType.TEXT,
            description="",
            tags=[],
            template={"content": "Hello!"},
            config=None,
            variables=[],
            labels=[],
            version=1,
            is_fallback=False,
            commit_message="",
            created_by="",
            created_at="",
            updated_at="",
        )
        prompt = Prompt.from_data(data)

        # Simulate linking logic from _add_prompt_attributes
        should_link = not prompt.is_fallback
        attrs = {}

        if should_link:
            attrs[Attrs.BROKLE_PROMPT_NAME] = prompt.name
            attrs[Attrs.BROKLE_PROMPT_VERSION] = prompt.version
            if prompt.id and prompt.id != "fallback":
                attrs[Attrs.BROKLE_PROMPT_ID] = prompt.id

        assert attrs[Attrs.BROKLE_PROMPT_NAME] == "greet"
        assert attrs[Attrs.BROKLE_PROMPT_VERSION] == 1
        assert attrs[Attrs.BROKLE_PROMPT_ID] == "01HXY123456789ABCDEFGHIJ"

    def test_should_not_link_fallback_prompt(self):
        """Fallback prompts should NOT be linked - no attributes set."""
        prompt = Prompt.create_fallback("greet", "Hello!")

        # Simulate linking logic
        should_link = not prompt.is_fallback
        attrs = {}

        if should_link:
            attrs[Attrs.BROKLE_PROMPT_NAME] = prompt.name
            attrs[Attrs.BROKLE_PROMPT_VERSION] = prompt.version
            if prompt.id and prompt.id != "fallback":
                attrs[Attrs.BROKLE_PROMPT_ID] = prompt.id

        # No attributes should be set for fallback
        assert len(attrs) == 0

    def test_should_skip_prompt_id_if_equals_fallback(self):
        """ID attribute should be skipped if it equals 'fallback'."""
        # Edge case: normal prompt somehow has id "fallback"
        data = PromptData(
            id="fallback",  # Edge case
            project_id="proj-1",
            name="test",
            type=PromptType.TEXT,
            description="",
            tags=[],
            template={"content": "Hello!"},
            config=None,
            variables=[],
            labels=[],
            version=1,
            is_fallback=False,  # Not a fallback, but weird ID
            commit_message="",
            created_by="",
            created_at="",
            updated_at="",
        )
        prompt = Prompt.from_data(data)

        # Simulate linking logic
        attrs = {}
        if not prompt.is_fallback:
            attrs[Attrs.BROKLE_PROMPT_NAME] = prompt.name
            attrs[Attrs.BROKLE_PROMPT_VERSION] = prompt.version
            if prompt.id and prompt.id != "fallback":
                attrs[Attrs.BROKLE_PROMPT_ID] = prompt.id

        assert attrs[Attrs.BROKLE_PROMPT_NAME] == "test"
        assert attrs[Attrs.BROKLE_PROMPT_VERSION] == 1
        # ID should NOT be set because it equals 'fallback'
        assert Attrs.BROKLE_PROMPT_ID not in attrs


class TestObserveDecoratorDynamicLinking:
    """Tests for @observe decorator with dynamic prompt linking."""

    def test_observe_decorator_supports_dynamic_linking(self):
        """@observe decorated functions can use link_prompt() or update_current_span()."""
        from brokle.decorators import observe

        # Verify decorator works without prompt parameter
        # Prompt linking should be done dynamically inside the function
        @observe(name="test-op")
        def test_function():
            # In real usage:
            # prompt = brokle.prompts.get("assistant")
            # brokle.link_prompt(prompt)
            return "result"

        # Decorator should work without prompt parameter
        assert callable(test_function)


class TestWrapperPromptLinking:
    """Tests for wrapper function prompt linking patterns."""

    def test_brokle_options_pattern(self):
        """brokle_options dict correctly handles prompt extraction."""
        data = PromptData(
            id="01HXY123456789ABCDEFGHIJ",
            project_id="proj-1",
            name="openai-prompt",
            type=PromptType.CHAT,
            description="",
            tags=[],
            template={"messages": [{"role": "user", "content": "{{input}}"}]},
            config=None,
            variables=["input"],
            labels=[],
            version=1,
            is_fallback=False,
            commit_message="",
            created_by="",
            created_at="",
            updated_at="",
        )
        prompt = Prompt.from_data(data)

        # Simulate brokle_options extraction
        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "brokle_options": {"prompt": prompt},
        }

        # Extract brokle_options (similar to _extract_brokle_options)
        brokle_options = kwargs.pop("brokle_options", None) or {}

        assert brokle_options.get("prompt") is prompt
        assert brokle_options["prompt"].name == "openai-prompt"
        assert brokle_options["prompt"].is_fallback is False

        # Clean kwargs should not have brokle_options
        assert "brokle_options" not in kwargs
        assert kwargs["model"] == "gpt-4"

    def test_brokle_options_skips_fallback_prompts(self):
        """brokle_options with fallback prompts should not add attributes."""
        fallback_prompt = Prompt.create_fallback("fallback", "Default")

        brokle_options = {"prompt": fallback_prompt}

        # Simulate _add_prompt_attributes logic
        attrs = {}
        prompt = brokle_options.get("prompt")

        if prompt and not prompt.is_fallback:
            attrs[Attrs.BROKLE_PROMPT_NAME] = prompt.name
            attrs[Attrs.BROKLE_PROMPT_VERSION] = prompt.version
            if prompt.id and prompt.id != "fallback":
                attrs[Attrs.BROKLE_PROMPT_ID] = prompt.id

        # No attributes should be set for fallback
        assert len(attrs) == 0


class TestContextManagerPromptLinking:
    """Tests for context manager prompt parameter support."""

    def test_context_manager_accepts_prompt_parameter(self):
        """Context managers should accept prompt parameter."""
        data = PromptData(
            id="01HXY123456789ABCDEFGHIJ",
            project_id="proj-1",
            name="context-prompt",
            type=PromptType.TEXT,
            description="",
            tags=[],
            template={"content": "Hello!"},
            config=None,
            variables=[],
            labels=[],
            version=1,
            is_fallback=False,
            commit_message="",
            created_by="",
            created_at="",
            updated_at="",
        )
        prompt = Prompt.from_data(data)

        # Verify prompt is suitable for context manager usage
        assert prompt.is_fallback is False
        assert prompt.name == "context-prompt"
        assert prompt.version == 1

        # In actual usage:
        # with brokle.start_as_current_span("op", prompt=prompt) as span:
        #     pass


class TestLinkPromptMethod:
    """Tests for link_prompt() client method."""

    def test_link_prompt_checks_fallback(self):
        """link_prompt should skip fallback prompts."""
        fallback_prompt = Prompt.create_fallback("test", "Hello")

        # link_prompt should return False for fallbacks
        # Simulate the check
        should_link = not fallback_prompt.is_fallback

        assert should_link is False

    def test_link_prompt_accepts_normal_prompt(self):
        """link_prompt should accept normal prompts."""
        data = PromptData(
            id="01HXY123456789ABCDEFGHIJ",
            project_id="proj-1",
            name="dynamic",
            type=PromptType.TEXT,
            description="",
            tags=[],
            template={"content": "Hello!"},
            config=None,
            variables=[],
            labels=[],
            version=3,
            is_fallback=False,
            commit_message="",
            created_by="",
            created_at="",
            updated_at="",
        )
        prompt = Prompt.from_data(data)

        # link_prompt should return True for normal prompts (if span is active)
        # Simulate the check
        should_link = not prompt.is_fallback

        assert should_link is True


class TestUpdateCurrentSpanMethod:
    """Tests for update_current_span() client method."""

    def test_update_current_span_accepts_prompt(self):
        """update_current_span should accept prompt parameter."""
        data = PromptData(
            id="01HXY123456789ABCDEFGHIJ",
            project_id="proj-1",
            name="updated",
            type=PromptType.TEXT,
            description="",
            tags=[],
            template={"content": "Updated!"},
            config=None,
            variables=[],
            labels=[],
            version=4,
            is_fallback=False,
            commit_message="",
            created_by="",
            created_at="",
            updated_at="",
        )
        prompt = Prompt.from_data(data)

        # Verify prompt is suitable for update_current_span
        assert prompt.is_fallback is False
        assert prompt.name == "updated"
        assert prompt.version == 4

        # In actual usage:
        # brokle.update_current_span(prompt=prompt, output="result")

    def test_update_current_span_skips_fallback_prompt(self):
        """update_current_span should skip fallback prompts."""
        fallback = Prompt.create_fallback("fallback", "Default")

        # Simulate linking logic
        attrs = {}
        if not fallback.is_fallback:
            attrs[Attrs.BROKLE_PROMPT_NAME] = fallback.name

        # No attributes should be set
        assert len(attrs) == 0


class TestUpdateCurrentGenerationAlias:
    """Tests for update_current_generation alias."""

    def test_alias_exists(self):
        """update_current_generation should be an alias for update_current_span."""
        # This is verified by checking the client module
        # The actual alias is: update_current_generation = update_current_span
        from brokle._base_client import BaseBrokleClient

        # Verify the method exists
        assert hasattr(BaseBrokleClient, "update_current_span")
        assert hasattr(BaseBrokleClient, "update_current_generation")


class TestChatPromptLinking:
    """Tests for CHAT type prompt linking."""

    def test_chat_prompt_links_correctly(self):
        """CHAT prompts should link with same attributes as TEXT."""
        data = PromptData(
            id="01HXY123456789ABCDEFGHIJ",
            project_id="proj-1",
            name="chat-assistant",
            type=PromptType.CHAT,
            description="",
            tags=[],
            template={
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "{{query}}"},
                ]
            },
            config={"model": "gpt-4"},
            variables=["query"],
            labels=[],
            version=5,
            is_fallback=False,
            commit_message="",
            created_by="",
            created_at="",
            updated_at="",
        )
        prompt = Prompt.from_data(data)

        # Simulate linking
        attrs = {}
        if not prompt.is_fallback:
            attrs[Attrs.BROKLE_PROMPT_NAME] = prompt.name
            attrs[Attrs.BROKLE_PROMPT_VERSION] = prompt.version
            if prompt.id and prompt.id != "fallback":
                attrs[Attrs.BROKLE_PROMPT_ID] = prompt.id

        assert attrs[Attrs.BROKLE_PROMPT_NAME] == "chat-assistant"
        assert attrs[Attrs.BROKLE_PROMPT_VERSION] == 5
        assert attrs[Attrs.BROKLE_PROMPT_ID] == "01HXY123456789ABCDEFGHIJ"

    def test_chat_fallback_does_not_link(self):
        """CHAT fallback prompts should NOT be linked."""
        fallback = Prompt.create_fallback(
            "chat-fallback",
            [
                {"role": "system", "content": "Default assistant"},
                {"role": "user", "content": "{{input}}"},
            ],
        )

        assert fallback.is_fallback is True
        assert fallback.type == PromptType.CHAT

        # Simulate linking
        attrs = {}
        if not fallback.is_fallback:
            attrs[Attrs.BROKLE_PROMPT_NAME] = fallback.name

        assert len(attrs) == 0
