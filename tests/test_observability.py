"""
Tests for observability module.

Tests the stable public API for Pattern 1/2 compatibility.
"""

import threading
from unittest.mock import patch

import pytest

from brokle.observability import (
    BrokleGeneration,
    BrokleOtelSpanAttributes,
    BrokleSpan,
    clear_context,
    create_span,
    get_client,
    get_client_context,
    get_config,
    get_context_info,
    get_current_span,
    span_context,
    telemetry_enabled,
)


class TestObservabilityConfig:
    """Test observability configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = get_config()

        assert config.telemetry_enabled is True
        assert config.debug is False
        assert config.sample_rate == 1.0
        assert config.batch_size == 100
        assert config.flush_interval == 10000

    def test_telemetry_enabled_function(self):
        """Test telemetry_enabled function."""
        assert telemetry_enabled() is True

    def test_configure_observability(self):
        """Test configuring observability."""
        from brokle.observability.config import configure_observability, reset_config

        # Configure with custom settings
        configure_observability(
            telemetry_enabled=False,
            debug=True,
            sample_rate=0.5,
            batch_size=50,
            flush_interval=5000,
        )

        config = get_config()
        assert config.telemetry_enabled is False
        assert config.debug is True
        assert config.sample_rate == 0.5
        assert config.batch_size == 50
        assert config.flush_interval == 5000

        # Reset to defaults
        reset_config()
        config = get_config()
        assert config.telemetry_enabled is True


class TestObservabilityContext:
    """Test observability context management."""

    def teardown_method(self):
        """Clean up after each test."""
        clear_context()

    def test_get_client_creates_new(self):
        """Test get_client creates new client when none exists."""
        with patch.dict("os.environ", {"BROKLE_API_KEY": "bk_test"}):
            client = get_client()
            assert client is not None
            assert client.config.api_key == "bk_test"

    def test_get_client_context_none_initially(self):
        """Test get_client_context returns None initially."""
        assert get_client_context() is None

    def test_context_persistence(self):
        """Test context persists within thread."""
        with patch.dict("os.environ", {"BROKLE_API_KEY": "bk_test"}):
            client1 = get_client()
            client2 = get_client()

            # Should be the same client instance
            assert client1 is client2

    def test_clear_context(self):
        """Test clearing context."""
        with patch.dict("os.environ", {"BROKLE_API_KEY": "bk_test"}):
            client = get_client()
            assert get_client_context() is client

            clear_context()
            assert get_client_context() is None

    def test_get_context_info(self):
        """Test getting context information."""
        # No client initially
        info = get_context_info()
        assert info["has_client"] is False

        # With client
        with patch.dict("os.environ", {"BROKLE_API_KEY": "bk_test123"}):
            get_client()
            info = get_context_info()

            assert info["has_client"] is True
            assert info["api_key"] == "bk_test123..."

    def test_thread_isolation(self):
        """Test that contexts are isolated between threads with explicit credentials."""
        results = {}

        def thread_func(thread_id):
            # Pass credentials explicitly - no environment mutation!
            # This is production-safe and thread-safe
            client = get_client(api_key=f"bk_thread_{thread_id}")
            results[thread_id] = client.config.api_key

        # Start multiple threads
        threads = []
        for i in range(3):
            t = threading.Thread(target=thread_func, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Each thread should have its own client with correct credentials
        assert len(results) == 3
        assert results[0] == "bk_thread_0"
        assert results[1] == "bk_thread_1"
        assert results[2] == "bk_thread_2"

    def test_explicit_credentials_persist_in_thread(self):
        """Test that explicit credentials are latched into thread-local context."""

        def thread_test():
            # First call with explicit credentials - should create and store client
            client1 = get_client(api_key="bk_explicit_test")

            # Second call without credentials - should reuse the stored client
            client2 = get_client()

            # Should be the same instance with same credentials
            return (client1 is client2, client1.config.api_key, client2.config.api_key)

        import threading

        result = []

        def worker():
            result.append(thread_test())

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        same_instance, api_key1, api_key2 = result[0]

        # Verify same instance is reused
        assert same_instance, "get_client() should reuse the same instance in thread"

        # Verify credentials are preserved
        assert api_key1 == "bk_explicit_test"
        assert api_key2 == "bk_explicit_test"


class TestBrokleSpan:
    """Test BrokleSpan functionality."""

    def test_span_creation(self):
        """Test basic span creation."""
        span = BrokleSpan(name="test-span")

        assert span.name == "test-span"
        assert span.span_id.startswith("span_")
        assert span.status == "started"
        assert span.start_time is not None
        assert span.end_time is None
        assert span.attributes == {}
        assert span.tags == []

    def test_span_attributes(self):
        """Test setting span attributes."""
        span = BrokleSpan(name="test-span")

        span.set_attribute("test.key", "test_value")
        assert span.attributes["test.key"] == "test_value"

        span.set_attribute("test.number", 42)
        assert span.attributes["test.number"] == 42

    def test_span_status(self):
        """Test setting span status."""
        span = BrokleSpan(name="test-span")

        span.set_status("completed", "Operation completed successfully")
        assert span.status == "completed"
        assert (
            span.attributes["status_description"] == "Operation completed successfully"
        )

    def test_span_tags(self):
        """Test adding span tags."""
        span = BrokleSpan(name="test-span")

        span.add_tag("production")
        span.add_tag("ai-call")
        span.add_tag("production")  # Duplicate should be ignored

        assert "production" in span.tags
        assert "ai-call" in span.tags
        assert len(span.tags) == 2

    def test_span_finish(self):
        """Test finishing a span."""
        span = BrokleSpan(name="test-span")
        assert span.end_time is None
        assert span.status == "started"

        span.finish()

        assert span.end_time is not None
        assert span.status == "completed"

    def test_span_to_dict(self):
        """Test converting span to dictionary."""
        span = BrokleSpan(
            name="test-span",
            trace_id="trace_123",
            attributes={"test": "value"},
            tags=["tag1", "tag2"],
        )
        span.finish()

        data = span.to_dict()

        assert data["name"] == "test-span"
        assert data["trace_id"] == "trace_123"
        assert data["status"] == "completed"
        assert data["attributes"]["test"] == "value"
        assert data["tags"] == ["tag1", "tag2"]
        assert data["start_time"] is not None
        assert data["end_time"] is not None


class TestBrokleGeneration:
    """Test BrokleGeneration span."""

    def test_generation_creation(self):
        """Test generation span creation."""
        gen = BrokleGeneration(name="llm-call")

        assert gen.name == "llm-call"
        assert gen.model is None
        assert gen.provider is None
        assert gen.input_tokens is None
        assert gen.output_tokens is None
        assert gen.cost_usd is None

    def test_set_model_info(self):
        """Test setting model information."""
        gen = BrokleGeneration(name="llm-call")

        gen.set_model_info(
            model="gpt-4",
            provider="openai",
            input_tokens=50,
            output_tokens=20,
            cost_usd=0.003,
        )

        assert gen.model == "gpt-4"
        assert gen.provider == "openai"
        assert gen.input_tokens == 50
        assert gen.output_tokens == 20
        assert gen.cost_usd == 0.003


class TestSpanManagement:
    """Test span management functions."""

    def teardown_method(self):
        """Clean up after each test."""
        from brokle.observability.spans import _set_current_span

        _set_current_span(None)

    def test_create_span(self):
        """Test creating a span."""
        span = create_span("test-operation", attributes={"key": "value"}, tags=["test"])

        assert span.name == "test-operation"
        assert span.trace_id is not None
        assert span.attributes["key"] == "value"
        assert "test" in span.tags

    def test_span_hierarchy(self):
        """Test parent-child span relationships."""
        parent_span = create_span("parent-op")

        # Mock current span
        from brokle.observability.spans import _set_current_span

        _set_current_span(parent_span)

        child_span = create_span("child-op")

        assert child_span.parent_span_id == parent_span.span_id
        assert child_span.trace_id == parent_span.trace_id

    def test_get_current_span(self):
        """Test getting current span."""
        assert get_current_span() is None

        span = create_span("test-op")
        from brokle.observability.spans import _set_current_span

        _set_current_span(span)

        assert get_current_span() is span

    def test_span_context_manager(self):
        """Test span context manager."""
        with span_context("test-operation", attributes={"test": "value"}) as span:
            assert span.name == "test-operation"
            assert span.attributes["test"] == "value"
            assert get_current_span() is span

        # After context, span should be finished
        assert span.status == "completed"
        assert span.end_time is not None

    def test_span_context_error_handling(self):
        """Test span context manager with errors."""
        with pytest.raises(ValueError):
            with span_context("test-operation") as span:
                raise ValueError("Test error")

        # Span should be marked as error
        assert span.status == "error"
        assert span.attributes.get("status_description") == "Test error"


class TestBrokleOtelSpanAttributes:
    """Test OpenTelemetry span attributes."""

    def test_attribute_constants(self):
        """Test that all attribute constants are defined."""
        assert BrokleOtelSpanAttributes.REQUEST_ID == "brokle.request_id"
        assert BrokleOtelSpanAttributes.LLM_MODEL == "llm.model"
        assert BrokleOtelSpanAttributes.LLM_PROVIDER == "llm.provider"
        assert (
            BrokleOtelSpanAttributes.BROKLE_ROUTING_STRATEGY
            == "brokle.routing.strategy"
        )
        assert BrokleOtelSpanAttributes.LLM_COST_USD == "llm.cost.usd"

    def test_get_all_attributes(self):
        """Test getting all attributes."""
        attributes = BrokleOtelSpanAttributes.get_all_attributes()

        assert "REQUEST_ID" in attributes
        assert "LLM_MODEL" in attributes
        assert "BROKLE_ROUTING_STRATEGY" in attributes

        # Should not include methods
        assert "get_all_attributes" not in attributes
        assert "is_brokle_attribute" not in attributes

    def test_is_brokle_attribute(self):
        """Test checking if attribute is Brokle-specific."""
        assert BrokleOtelSpanAttributes.is_brokle_attribute("brokle.request_id") is True
        assert BrokleOtelSpanAttributes.is_brokle_attribute("llm.model") is True
        assert BrokleOtelSpanAttributes.is_brokle_attribute("http.method") is False
