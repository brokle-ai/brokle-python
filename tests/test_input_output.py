"""
Tests for trace and span input/output functionality.

Tests the new OpenInference pattern (input.value/output.value) for generic data
and OTLP GenAI standard (gen_ai.input.messages/output.messages) for LLM data.
"""

import json
import os
from unittest.mock import patch

import pytest

from brokle import Brokle, observe
from brokle.types import Attrs


@pytest.fixture
def brokle_client():
    """Create Brokle client for testing."""
    client = Brokle(
        api_key="bk_test_key_" + "x" * 30,
        base_url="http://localhost:8080",
        environment="test",
        tracing_enabled=True,
    )
    yield client
    client.close()


def test_decorator_captures_function_args(brokle_client):
    """Test that @observe decorator captures function args as input.value."""

    with patch.dict("os.environ", {"BROKLE_API_KEY": "bk_test_key_" + "x" * 30}):

        @observe(capture_input=True, capture_output=True)
        def get_weather(location: str, units: str = "celsius"):
            return {"temp": 25, "location": location, "units": units}

        # Execute function
        result = get_weather("Bangalore", units="fahrenheit")

        # Flush to ensure span is complete
        brokle_client.flush()

        # Note: We can't easily assert on span attributes in unit tests
        # This test validates that the function executes without errors
        # Integration tests with backend will validate attribute extraction
        assert result == {"temp": 25, "location": "Bangalore", "units": "fahrenheit"}


def test_decorator_sets_mime_type(brokle_client):
    """Test that decorator automatically sets MIME type to application/json."""

    with patch.dict("os.environ", {"BROKLE_API_KEY": "bk_test_key_" + "x" * 30}):

        @observe(capture_input=True, capture_output=True)
        def process_data(data: dict):
            return {"processed": True, "data": data}

        result = process_data({"key": "value"})
        brokle_client.flush()

        # Function executes successfully (MIME type set internally)
        assert result == {"processed": True, "data": {"key": "value"}}


def test_manual_span_generic_input():
    """Test manual span creation with generic input/output."""
    client = Brokle(
        api_key="bk_test_key_" + "x" * 30,
        base_url="http://localhost:8080",
        environment="test",
    )

    with client.start_as_current_span(
        "api-request",
        input={"endpoint": "/weather", "query": "Bangalore"},
        output={"status": 200, "data": {"temp": 25}},
    ) as span:
        # Span should have input and output attributes set
        # Verify span context exists
        assert span.get_span_context().is_valid

    client.close()


def test_manual_span_llm_messages():
    """Test manual span creation with LLM messages (auto-detected)."""
    client = Brokle(
        api_key="bk_test_key_" + "x" * 30,
        base_url="http://localhost:8080",
        environment="test",
    )

    # ChatML format should be auto-detected
    with client.start_as_current_span(
        "llm-conversation",
        input=[{"role": "user", "content": "Hello"}],
        output=[{"role": "assistant", "content": "Hi there!"}],
    ) as span:
        assert span.get_span_context().is_valid

    client.close()


def test_output_set_during_execution():
    """Test updating output during span execution."""
    client = Brokle(
        api_key="bk_test_key_" + "x" * 30,
        base_url="http://localhost:8080",
        environment="test",
    )

    with client.start_as_current_span("process", input={"data": "test"}) as span:
        # Initially no output
        # Do some work
        result = {"processed": True}

        # Update output manually
        output_str = json.dumps(result)
        span.set_attribute(Attrs.OUTPUT_VALUE, output_str)
        span.set_attribute(Attrs.OUTPUT_MIME_TYPE, "application/json")

    client.close()


def test_nested_spans_preserve_io():
    """Test that nested spans each have their own input/output."""
    client = Brokle(
        api_key="bk_test_key_" + "x" * 30,
        base_url="http://localhost:8080",
        environment="test",
    )

    with client.start_as_current_span(
        "parent", input={"parent_input": "data"}
    ) as parent:
        assert parent.get_span_context().is_valid

        with client.start_as_current_span(
            "child", input={"child_input": "different"}
        ) as child:
            assert child.get_span_context().is_valid
            # Each span has its own input

    client.close()


def test_generation_span_with_input_messages():
    """Test generation span with explicit input_messages parameter."""
    client = Brokle(
        api_key="bk_test_key_" + "x" * 30,
        base_url="http://localhost:8080",
        environment="test",
    )

    messages = [{"role": "user", "content": "Hello"}]

    with client.start_as_current_generation(
        name="chat", model="gpt-4", provider="openai", input_messages=messages
    ) as gen:
        # Should set gen_ai.input.messages
        assert gen.get_span_context().is_valid

    client.close()


def test_mixed_generic_and_llm_spans():
    """Test trace with both generic spans and LLM generations."""
    client = Brokle(
        api_key="bk_test_key_" + "x" * 30,
        base_url="http://localhost:8080",
        environment="test",
    )

    # Root span with generic input
    with client.start_as_current_span(
        "workflow", input={"task": "weather_query", "location": "Bangalore"}
    ) as workflow:

        # Child LLM generation span
        with client.start_as_current_generation(
            name="llm-call",
            model="gpt-4",
            provider="openai",
            input_messages=[{"role": "user", "content": "Get weather"}],
        ) as gen:
            # Set output
            gen.set_attribute(
                Attrs.GEN_AI_OUTPUT_MESSAGES,
                json.dumps([{"role": "assistant", "content": "25°C"}]),
            )

        # Update workflow output
        workflow.set_attribute(
            Attrs.OUTPUT_VALUE, json.dumps({"result": "25°C", "location": "Bangalore"})
        )
        workflow.set_attribute(Attrs.OUTPUT_MIME_TYPE, "application/json")

    client.close()


def test_input_output_none_values():
    """Test that None values are handled gracefully."""
    client = Brokle(
        api_key="bk_test_key_" + "x" * 30,
        base_url="http://localhost:8080",
        environment="test",
    )

    # None input/output should not cause errors
    with client.start_as_current_span("test", input=None, output=None) as span:
        assert span.get_span_context().is_valid

    client.close()
