"""
Tests for trace and span input/output functionality.

Tests the new OpenInference pattern (input.value/output.value) for generic data
and OTLP GenAI standard (gen_ai.input.messages/output.messages) for LLM data.
"""

import json
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


# =============================================================================
# Tracer Exception Isolation Tests (Graceful Degradation)
# =============================================================================


def test_tracer_output_error_does_not_leak(brokle_client, caplog):
    """
    Tracer error in _set_output_attr should not propagate to user.

    This tests the critical graceful degradation behavior: when the tracer
    fails to set output attributes (e.g., due to serialization failure or
    span.set_attribute throwing), the user's function should still return
    its result successfully.
    """
    import logging

    from brokle import decorators

    with patch.dict("os.environ", {"BROKLE_API_KEY": "bk_test_key_" + "x" * 30}):

        @observe(capture_output=True)
        def user_function():
            return "success"

        # Mock _set_output_attr to simulate tracer failure
        with patch.object(
            decorators, "_set_output_attr", side_effect=RuntimeError("tracer error")
        ):
            with caplog.at_level(logging.WARNING):
                result = user_function()

        # User result should be returned despite tracer failure
        assert result == "success"

        # Warning should be logged for graceful degradation
        assert any("Tracer operation error" in record.message for record in caplog.records)


def test_tracer_status_error_does_not_leak(brokle_client, caplog):
    """
    Tracer error in span.set_status should not propagate to user.

    When span.set_status(StatusCode.OK) fails, the user's successful
    function call should still return its result.
    """
    import logging

    from opentelemetry.trace import Status

    with patch.dict("os.environ", {"BROKLE_API_KEY": "bk_test_key_" + "x" * 30}):

        @observe(capture_output=False)  # Avoid _set_output_attr to isolate test
        def user_function():
            return "success"

        # Mock span.set_status to simulate tracer failure
        original_set_status = Status.__init__

        def failing_set_status(*args, **kwargs):
            original_set_status(*args, **kwargs)
            raise RuntimeError("set_status failed")

        with patch.object(Status, "__init__", failing_set_status):
            with caplog.at_level(logging.WARNING):
                result = user_function()

        # User result should be returned despite tracer failure
        assert result == "success"


def test_user_exception_preserved_when_tracer_fails(brokle_client, caplog):
    """
    User exception must propagate even when tracer also fails.

    When user code raises an exception AND tracer operations fail,
    the user's original exception must be raised (not the tracer error).
    """
    import logging

    from opentelemetry.trace import Status

    class UserError(Exception):
        pass

    with patch.dict("os.environ", {"BROKLE_API_KEY": "bk_test_key_" + "x" * 30}):

        @observe()
        def user_function():
            raise UserError("user error")

        # Mock span.set_status to simulate tracer failure during error handling
        # We need to patch the actual method that sets status on ERROR path
        original_status_init = Status.__init__

        call_count = [0]

        def failing_status(*args, **kwargs):
            call_count[0] += 1
            original_status_init(*args, **kwargs)
            # Only fail on the ERROR status (second call in error path)
            if call_count[0] > 0:
                raise RuntimeError("set_status failed")

        with patch.object(Status, "__init__", failing_status):
            with caplog.at_level(logging.WARNING):
                # The key assertion: user exception must be raised, not tracer error
                with pytest.raises(UserError, match="user error"):
                    user_function()


@pytest.mark.asyncio
async def test_async_tracer_output_error_does_not_leak(brokle_client, caplog):
    """
    Async wrapper: Tracer error in _set_output_attr should not propagate.
    """
    import logging

    from brokle import decorators

    with patch.dict("os.environ", {"BROKLE_API_KEY": "bk_test_key_" + "x" * 30}):

        @observe(capture_output=True)
        async def async_user_function():
            return "async success"

        # Mock _set_output_attr to simulate tracer failure
        with patch.object(
            decorators, "_set_output_attr", side_effect=RuntimeError("tracer error")
        ):
            with caplog.at_level(logging.WARNING):
                result = await async_user_function()

        # User result should be returned despite tracer failure
        assert result == "async success"

        # Warning should be logged for graceful degradation
        assert any("Tracer operation error" in record.message for record in caplog.records)


def test_generator_tracer_output_error_does_not_leak(brokle_client, caplog):
    """
    Generator wrapper: Tracer error in _set_output_attr should not propagate.
    """
    import logging

    from brokle import decorators

    with patch.dict("os.environ", {"BROKLE_API_KEY": "bk_test_key_" + "x" * 30}):

        @observe(capture_output=True)
        def generator_function():
            yield "item1"
            yield "item2"
            yield "item3"

        # Mock _set_output_attr to simulate tracer failure
        with patch.object(
            decorators, "_set_output_attr", side_effect=RuntimeError("tracer error")
        ):
            with caplog.at_level(logging.WARNING):
                result = list(generator_function())

        # All items should be yielded despite tracer failure
        assert result == ["item1", "item2", "item3"]

        # Warning should be logged for graceful degradation
        assert any("Tracer operation error" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_async_generator_tracer_output_error_does_not_leak(brokle_client, caplog):
    """
    Async generator wrapper: Tracer error in _set_output_attr should not propagate.
    """
    import logging

    from brokle import decorators

    with patch.dict("os.environ", {"BROKLE_API_KEY": "bk_test_key_" + "x" * 30}):

        @observe(capture_output=True)
        async def async_generator_function():
            yield "async1"
            yield "async2"

        # Mock _set_output_attr to simulate tracer failure
        with patch.object(
            decorators, "_set_output_attr", side_effect=RuntimeError("tracer error")
        ):
            with caplog.at_level(logging.WARNING):
                result = [item async for item in async_generator_function()]

        # All items should be yielded despite tracer failure
        assert result == ["async1", "async2"]

        # Warning should be logged for graceful degradation
        assert any("Tracer operation error" in record.message for record in caplog.records)


def test_generator_user_exception_preserved_when_tracer_fails(brokle_client, caplog):
    """
    Generator: User exception must propagate even when tracer also fails.
    """
    import logging

    from opentelemetry.trace import Status

    class UserError(Exception):
        pass

    with patch.dict("os.environ", {"BROKLE_API_KEY": "bk_test_key_" + "x" * 30}):

        @observe()
        def generator_with_error():
            yield "item1"
            raise UserError("generator error")

        # Mock span.set_status to simulate tracer failure during error handling
        original_status_init = Status.__init__

        call_count = [0]

        def failing_status(*args, **kwargs):
            call_count[0] += 1
            original_status_init(*args, **kwargs)
            if call_count[0] > 0:
                raise RuntimeError("set_status failed")

        with patch.object(Status, "__init__", failing_status):
            with caplog.at_level(logging.WARNING):
                # The key assertion: user exception must be raised, not tracer error
                with pytest.raises(UserError, match="generator error"):
                    list(generator_with_error())


def test_none_return_not_reexecuted_on_span_exit_error(brokle_client, caplog):
    """
    Functions returning None should not be re-executed when span __exit__ fails.

    This tests the sentinel pattern: we must distinguish "function not executed"
    from "function executed and returned None" to avoid duplicate side effects.
    """
    import logging

    from brokle._client import Brokle as BrokleClient

    call_count = [0]

    with patch.dict("os.environ", {"BROKLE_API_KEY": "bk_test_key_" + "x" * 30}):

        @observe(capture_output=True)
        def function_returning_none():
            call_count[0] += 1
            return None

        # Mock the span context manager's __exit__ to raise an error
        original_start_span = BrokleClient.start_as_current_span

        class FailingSpanContext:
            def __init__(self, span):
                self.span = span

            def __enter__(self):
                return self.span

            def __exit__(self, exc_type, exc_val, exc_tb):
                raise RuntimeError("span __exit__ failed")

        def patched_start_span(self, *args, **kwargs):
            ctx = original_start_span(self, *args, **kwargs)
            span = ctx.__enter__()
            return FailingSpanContext(span)

        with patch.object(BrokleClient, "start_as_current_span", patched_start_span):
            with caplog.at_level(logging.WARNING):
                result = function_returning_none()

        # Function should have been called exactly ONCE
        assert call_count[0] == 1, f"Function was called {call_count[0]} times, expected 1"

        # Result should be None (the actual return value)
        assert result is None

        # Warning should be logged about the tracer error
        assert any("Tracer span error" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_async_none_return_not_reexecuted_on_span_exit_error(brokle_client, caplog):
    """
    Async functions returning None should not be re-executed when span __exit__ fails.
    """
    import logging

    from brokle._client import Brokle as BrokleClient

    call_count = [0]

    with patch.dict("os.environ", {"BROKLE_API_KEY": "bk_test_key_" + "x" * 30}):

        @observe(capture_output=True)
        async def async_function_returning_none():
            call_count[0] += 1
            return None

        # Mock the span context manager's __exit__ to raise an error
        original_start_span = BrokleClient.start_as_current_span

        class FailingSpanContext:
            def __init__(self, span):
                self.span = span

            def __enter__(self):
                return self.span

            def __exit__(self, exc_type, exc_val, exc_tb):
                raise RuntimeError("span __exit__ failed")

        def patched_start_span(self, *args, **kwargs):
            ctx = original_start_span(self, *args, **kwargs)
            span = ctx.__enter__()
            return FailingSpanContext(span)

        with patch.object(BrokleClient, "start_as_current_span", patched_start_span):
            with caplog.at_level(logging.WARNING):
                result = await async_function_returning_none()

        # Function should have been called exactly ONCE
        assert call_count[0] == 1, f"Function was called {call_count[0]} times, expected 1"

        # Result should be None (the actual return value)
        assert result is None

        # Warning should be logged about the tracer error
        assert any("Tracer span error" in record.message for record in caplog.records)


# =============================================================================
# Traceback Preservation Tests
# =============================================================================


def test_user_exception_traceback_preserved_sync(brokle_client):
    """
    Verify decorator preserves original traceback and user code is the innermost frame.

    The @observe decorator stores and re-raises exceptions. With proper traceback
    preservation using .with_traceback(), the innermost frame should be the user's
    actual error location, not a re-raise statement in the decorator.
    """
    with patch.dict("os.environ", {"BROKLE_API_KEY": "bk_test_key_" + "x" * 30}):

        @observe()
        def failing_function():
            raise ValueError("test error")

        try:
            failing_function()
            pytest.fail("Expected ValueError to be raised")
        except ValueError as e:
            # Get the innermost frame (where the error actually occurred)
            tb = e.__traceback__
            while tb.tb_next is not None:
                tb = tb.tb_next

            # The innermost frame should be in the user's function
            innermost_funcname = tb.tb_frame.f_code.co_name
            assert (
                innermost_funcname == "failing_function"
            ), f"Innermost frame should be user function, got: {innermost_funcname}"


@pytest.mark.asyncio
async def test_user_exception_traceback_preserved_async(brokle_client):
    """
    Verify decorator preserves original traceback (async).
    """
    with patch.dict("os.environ", {"BROKLE_API_KEY": "bk_test_key_" + "x" * 30}):

        @observe()
        async def async_failing_function():
            raise ValueError("async test error")

        try:
            await async_failing_function()
            pytest.fail("Expected ValueError to be raised")
        except ValueError as e:
            tb = e.__traceback__
            while tb.tb_next is not None:
                tb = tb.tb_next

            innermost_funcname = tb.tb_frame.f_code.co_name
            assert (
                innermost_funcname == "async_failing_function"
            ), f"Innermost frame should be user function, got: {innermost_funcname}"


def test_user_exception_traceback_preserved_generator(brokle_client):
    """
    Verify decorator preserves original traceback (generator).
    """
    with patch.dict("os.environ", {"BROKLE_API_KEY": "bk_test_key_" + "x" * 30}):

        @observe()
        def failing_generator():
            yield "item1"
            raise ValueError("generator test error")

        try:
            list(failing_generator())
            pytest.fail("Expected ValueError to be raised")
        except ValueError as e:
            tb = e.__traceback__
            while tb.tb_next is not None:
                tb = tb.tb_next

            innermost_funcname = tb.tb_frame.f_code.co_name
            assert (
                innermost_funcname == "failing_generator"
            ), f"Innermost frame should be user function, got: {innermost_funcname}"


@pytest.mark.asyncio
async def test_user_exception_traceback_preserved_async_generator(brokle_client):
    """
    Verify decorator preserves original traceback (async generator).
    """
    with patch.dict("os.environ", {"BROKLE_API_KEY": "bk_test_key_" + "x" * 30}):

        @observe()
        async def async_failing_generator():
            yield "item1"
            raise ValueError("async generator test error")

        try:
            result = [item async for item in async_failing_generator()]
            pytest.fail("Expected ValueError to be raised")
        except ValueError as e:
            tb = e.__traceback__
            while tb.tb_next is not None:
                tb = tb.tb_next

            innermost_funcname = tb.tb_frame.f_code.co_name
            assert (
                innermost_funcname == "async_failing_generator"
            ), f"Innermost frame should be user function, got: {innermost_funcname}"


def test_traceback_preserves_original_location(brokle_client):
    """
    Verify the original error location is preserved in traceback.

    The traceback should point to the exact line where the user's error occurred,
    not to re-raise statements in the decorator.
    """
    with patch.dict("os.environ", {"BROKLE_API_KEY": "bk_test_key_" + "x" * 30}):

        @observe()
        def function_with_specific_error():
            x = 1
            y = 2
            # Error is on this specific line
            raise RuntimeError("specific error at line")

        try:
            function_with_specific_error()
            pytest.fail("Expected RuntimeError to be raised")
        except RuntimeError as e:
            # Get the innermost frame (where the error actually occurred)
            tb = e.__traceback__
            while tb.tb_next is not None:
                tb = tb.tb_next

            # The innermost frame should be in the user's function, not decorator
            innermost_filename = tb.tb_frame.f_code.co_filename
            innermost_funcname = tb.tb_frame.f_code.co_name

            # Should be in test file and in user's function
            assert (
                "test_input_output" in innermost_filename
            ), f"Innermost frame should be in test file, got: {innermost_filename}"
            assert (
                innermost_funcname == "function_with_specific_error"
            ), f"Innermost frame should be in user function, got: {innermost_funcname}"


# =============================================================================
# Empty Generator Re-execution Prevention Tests
# =============================================================================


def test_empty_generator_not_reexecuted_on_span_exit_error(brokle_client, caplog):
    """
    Empty generators (0 items) should not be re-executed when span __exit__ fails.

    This tests a critical edge case: generators that perform side effects but
    yield nothing should only run once, even if tracer errors occur after completion.

    The bug scenario:
    1. Empty generator runs, performs side effects (e.g., DB writes)
    2. Yields 0 items, loop completes normally
    3. iteration_started remains False (no items yielded)
    4. Tracer span.__exit__() raises an error
    5. Old code would re-run the generator because iteration_started=False
    6. Side effects would be duplicated!

    The fix uses generator_exhausted flag to track loop completion.
    """
    import logging

    from brokle._client import Brokle as BrokleClient

    call_count = [0]

    with patch.dict("os.environ", {"BROKLE_API_KEY": "bk_test_key_" + "x" * 30}):

        @observe()
        def empty_generator_with_side_effect():
            call_count[0] += 1  # Side effect
            # Yields nothing (e.g., empty query results)
            # Must have a yield statement to be a generator, but it's unreachable
            if False:
                yield

        # Mock the span context manager's __exit__ to raise an error
        original_start_span = BrokleClient.start_as_current_span

        class FailingSpanContext:
            def __init__(self, span):
                self.span = span

            def __enter__(self):
                return self.span

            def __exit__(self, exc_type, exc_val, exc_tb):
                raise RuntimeError("span __exit__ failed")

        def patched_start_span(self, *args, **kwargs):
            ctx = original_start_span(self, *args, **kwargs)
            span = ctx.__enter__()
            return FailingSpanContext(span)

        with patch.object(BrokleClient, "start_as_current_span", patched_start_span):
            with caplog.at_level(logging.WARNING):
                result = list(empty_generator_with_side_effect())

        # Generator should have been called exactly ONCE
        assert call_count[0] == 1, f"Generator was called {call_count[0]} times, expected 1"
        assert result == []  # No items yielded

        # Warning should be logged about the tracer error
        assert any("Tracer span error" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_async_empty_generator_not_reexecuted_on_span_exit_error(brokle_client, caplog):
    """
    Async empty generators (0 items) should not be re-executed when span __exit__ fails.

    Same scenario as sync version but for async generators.
    """
    import logging

    from brokle._client import Brokle as BrokleClient

    call_count = [0]

    with patch.dict("os.environ", {"BROKLE_API_KEY": "bk_test_key_" + "x" * 30}):

        @observe()
        async def async_empty_generator_with_side_effect():
            call_count[0] += 1  # Side effect
            # Yields nothing (e.g., empty query results)
            # Must have a yield statement to be an async generator, but it's unreachable
            if False:
                yield

        # Mock the span context manager's __exit__ to raise an error
        original_start_span = BrokleClient.start_as_current_span

        class FailingSpanContext:
            def __init__(self, span):
                self.span = span

            def __enter__(self):
                return self.span

            def __exit__(self, exc_type, exc_val, exc_tb):
                raise RuntimeError("span __exit__ failed")

        def patched_start_span(self, *args, **kwargs):
            ctx = original_start_span(self, *args, **kwargs)
            span = ctx.__enter__()
            return FailingSpanContext(span)

        with patch.object(BrokleClient, "start_as_current_span", patched_start_span):
            with caplog.at_level(logging.WARNING):
                result = [item async for item in async_empty_generator_with_side_effect()]

        # Generator should have been called exactly ONCE
        assert call_count[0] == 1, f"Generator was called {call_count[0]} times, expected 1"
        assert result == []  # No items yielded

        # Warning should be logged about the tracer error
        assert any("Tracer span error" in record.message for record in caplog.records)


def test_non_empty_generator_still_works_on_span_exit_error(brokle_client, caplog):
    """
    Non-empty generators should still work correctly when span __exit__ fails.

    This is a regression test to ensure the fix for empty generators doesn't
    break the existing behavior for generators that yield items.
    """
    import logging

    from brokle._client import Brokle as BrokleClient

    call_count = [0]

    with patch.dict("os.environ", {"BROKLE_API_KEY": "bk_test_key_" + "x" * 30}):

        @observe()
        def generator_with_items():
            call_count[0] += 1
            yield "item1"
            yield "item2"
            yield "item3"

        # Mock the span context manager's __exit__ to raise an error
        original_start_span = BrokleClient.start_as_current_span

        class FailingSpanContext:
            def __init__(self, span):
                self.span = span

            def __enter__(self):
                return self.span

            def __exit__(self, exc_type, exc_val, exc_tb):
                raise RuntimeError("span __exit__ failed")

        def patched_start_span(self, *args, **kwargs):
            ctx = original_start_span(self, *args, **kwargs)
            span = ctx.__enter__()
            return FailingSpanContext(span)

        with patch.object(BrokleClient, "start_as_current_span", patched_start_span):
            with caplog.at_level(logging.WARNING):
                result = list(generator_with_items())

        # Generator should have been called exactly ONCE
        assert call_count[0] == 1, f"Generator was called {call_count[0]} times, expected 1"
        # All items should be yielded
        assert result == ["item1", "item2", "item3"]

        # Warning should be logged about the tracer error
        assert any("Tracer span error" in record.message for record in caplog.records)
