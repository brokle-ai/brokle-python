"""Integration tests for backward compatibility."""

import warnings

import pytest

from brokle.types.responses import BaseResponse, BrokleMetadata


def test_end_to_end_compatibility():
    """Test complete compatibility scenario."""
    # Simulate response from API
    metadata = BrokleMetadata(
        request_id="req_test",
        provider="openai",
        cost_usd=0.001,
        cache_hit=False,
        quality_score=0.88,
        input_tokens=150,
        output_tokens=75,
        total_tokens=225,
        latency_ms=200.0,
        routing_reason="balanced",
    )

    response = BaseResponse(brokle=metadata)

    # Legacy code should work with warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Simulate legacy application code
        if response.provider == "openai":
            cost = response.cost_usd
            is_cached = response.cache_hit
            tokens = response.total_tokens

        assert len(w) == 4  # Four property accesses = four warnings
        assert cost == 0.001
        assert is_cached is False
        assert tokens == 225

    # New code should work without warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        if response.brokle and response.brokle.provider == "openai":
            cost = response.brokle.cost_usd
            is_cached = response.brokle.cache_hit
            tokens = response.brokle.total_tokens

        assert len(w) == 0  # No warnings for new pattern
        assert cost == 0.001
        assert is_cached is False
        assert tokens == 225


def test_mixed_usage_patterns():
    """Test mixed usage of old and new patterns in the same code."""
    metadata = BrokleMetadata(
        request_id="req_mixed",
        provider="anthropic",
        cost_usd=0.003,
        cache_hit=True,
        quality_score=0.92,
        latency_ms=150.5,
    )

    response = BaseResponse(brokle=metadata)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Mix of old and new patterns
        old_provider = response.provider  # Should warn
        new_cost = (
            response.brokle.cost_usd if response.brokle else None
        )  # Should not warn
        old_cache = response.cache_hit  # Should warn
        new_quality = (
            response.brokle.quality_score if response.brokle else None
        )  # Should not warn

        # Should have exactly 2 warnings (for the 2 legacy accesses)
        assert len(w) == 2
        for warning in w:
            assert issubclass(warning.category, DeprecationWarning)

        # Both patterns should return the same values
        assert old_provider == "anthropic"
        assert new_cost == 0.003
        assert old_cache is True
        assert new_quality == 0.92


def test_type_consistency():
    """Test that legacy properties maintain type consistency."""
    metadata = BrokleMetadata(
        request_id="req_types",
        provider="openai",
        cost_usd=0.002,
        cache_hit=True,
        quality_score=0.95,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        latency_ms=200.5,
        routing_reason="cost_optimized",
    )

    response = BaseResponse(brokle=metadata)

    # Suppress warnings for type checking
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Check types match between old and new patterns
        assert type(response.request_id) == type(response.brokle.request_id)
        assert type(response.provider) == type(response.brokle.provider)
        assert type(response.cost_usd) == type(response.brokle.cost_usd)
        assert type(response.cache_hit) == type(response.brokle.cache_hit)
        assert type(response.quality_score) == type(response.brokle.quality_score)
        assert type(response.input_tokens) == type(response.brokle.input_tokens)
        assert type(response.output_tokens) == type(response.brokle.output_tokens)
        assert type(response.total_tokens) == type(response.brokle.total_tokens)
        assert type(response.latency_ms) == type(response.brokle.latency_ms)
        assert type(response.routing_reason) == type(response.brokle.routing_reason)


def test_none_safety():
    """Test that legacy properties handle None safely."""
    response = BaseResponse()  # No brokle metadata

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore deprecation warnings

        # All legacy properties should return None gracefully
        assert response.request_id is None
        assert response.provider is None
        assert response.cost_usd is None
        assert response.cache_hit is None
        assert response.quality_score is None
        assert response.input_tokens is None
        assert response.output_tokens is None
        assert response.total_tokens is None
        assert response.latency_ms is None
        assert response.routing_reason is None


def test_inheritance_compatibility():
    """Test that inheritance works with compatibility properties."""
    from brokle.types.responses.core import (
        ChatCompletionChoice,
        ChatCompletionMessage,
        ChatCompletionResponse,
    )

    # Create a full chat completion response
    metadata = BrokleMetadata(
        request_id="req_inherit",
        provider="openai",
        cost_usd=0.004,
        cache_hit=False,
        quality_score=0.89,
    )

    message = ChatCompletionMessage(role="assistant", content="Hello!")
    choice = ChatCompletionChoice(index=0, message=message, finish_reason="stop")

    response = ChatCompletionResponse(
        id="chatcmpl-123",
        object="chat.completion",
        created=1234567890,
        model="gpt-4",
        choices=[choice],
        usage={"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        brokle=metadata,
    )

    # Test that inherited compatibility properties work
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        assert response.provider == "openai"  # Should work with warning
        assert response.cost_usd == 0.004  # Should work with warning

        assert len(w) == 2
        for warning in w:
            assert issubclass(warning.category, DeprecationWarning)

    # Test that new pattern works without warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        if response.brokle:
            assert response.brokle.provider == "openai"
            assert response.brokle.cost_usd == 0.004

        assert len(w) == 0
