"""
Tests for response model mixins and base classes.

These tests ensure that our mixins work correctly with Pydantic validation,
serialization, and multiple inheritance patterns.
"""

from datetime import datetime, timezone
from typing import Any, Dict

import pytest

from brokle.types.responses.base import (
    BaseResponse,
    BrokleMetadata,
    BrokleResponseBase,
    CostTrackingMixin,
    FullContextResponse,
    MetadataMixin,
    OrganizationContextMixin,
    PaginatedResponse,
    PaginationMixin,
    ProviderMixin,
    ProviderResponse,
    RequestTrackingMixin,
    StatusMixin,
    TimestampedResponse,
    TimestampMixin,
    TokenUsageMixin,
    TrackedResponse,
)


class TestIndividualMixins:
    """Test each mixin individually."""

    def test_timestamp_mixin(self):
        """Test TimestampMixin fields and validation."""

        class TestModel(TimestampMixin):
            name: str

        # Test with required created_at
        now = datetime.now(timezone.utc)
        model = TestModel(name="test", created_at=now)

        assert model.created_at == now
        assert model.updated_at is None

        # Test with both timestamps
        updated = datetime.now(timezone.utc)
        model = TestModel(name="test", created_at=now, updated_at=updated)

        assert model.created_at == now
        assert model.updated_at == updated

    def test_metadata_mixin(self):
        """Test MetadataMixin optional fields."""

        class TestModel(MetadataMixin):
            name: str

        # Test with no metadata
        model = TestModel(name="test")
        assert model.metadata is None
        assert model.tags is None

        # Test with metadata
        metadata = {"key": "value", "nested": {"data": 123}}
        tags = {"env": "test", "version": "1.0"}

        model = TestModel(name="test", metadata=metadata, tags=tags)
        assert model.metadata == metadata
        assert model.tags == tags

    def test_token_usage_mixin(self):
        """Test TokenUsageMixin fields."""

        class TestModel(TokenUsageMixin):
            name: str

        # Test with no tokens
        model = TestModel(name="test")
        assert model.prompt_tokens is None
        assert model.completion_tokens is None
        assert model.total_tokens is None

        # Test with token values
        model = TestModel(
            name="test", prompt_tokens=100, completion_tokens=50, total_tokens=150
        )

        assert model.prompt_tokens == 100
        assert model.completion_tokens == 50
        assert model.total_tokens == 150

    def test_cost_tracking_mixin(self):
        """Test CostTrackingMixin fields."""

        class TestModel(CostTrackingMixin):
            name: str

        # Test with no costs
        model = TestModel(name="test")
        assert model.input_cost is None
        assert model.output_cost is None
        assert model.total_cost_usd is None

        # Test with cost values
        model = TestModel(
            name="test", input_cost=0.01, output_cost=0.02, total_cost_usd=0.03
        )

        assert model.input_cost == 0.01
        assert model.output_cost == 0.02
        assert model.total_cost_usd == 0.03

    def test_pagination_mixin(self):
        """Test PaginationMixin required fields."""

        class TestModel(PaginationMixin):
            name: str

        model = TestModel(name="test", total_count=100, page=0, page_size=20)

        assert model.total_count == 100
        assert model.page == 0
        assert model.page_size == 20

    def test_provider_mixin(self):
        """Test ProviderMixin optional fields."""

        class TestModel(ProviderMixin):
            name: str

        # Test with no provider info
        model = TestModel(name="test")
        assert model.provider is None
        assert model.model is None

        # Test with provider info
        model = TestModel(name="test", provider="openai", model="gpt-4")
        assert model.provider == "openai"
        assert model.model == "gpt-4"


class TestMultipleInheritance:
    """Test models with multiple mixin inheritance."""

    def test_combined_mixins(self):
        """Test model inheriting from multiple mixins."""

        class TestModel(
            BrokleResponseBase,
            TimestampMixin,
            ProviderMixin,
            TokenUsageMixin,
            CostTrackingMixin,
        ):
            name: str

        now = datetime.now(timezone.utc)
        model = TestModel(
            name="test",
            created_at=now,
            provider="openai",
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            input_cost=0.01,
            output_cost=0.02,
        )

        # Test all fields are accessible
        assert model.name == "test"
        assert model.created_at == now
        assert model.provider == "openai"
        assert model.model == "gpt-4"
        assert model.prompt_tokens == 100
        assert model.completion_tokens == 50
        assert model.total_tokens == 150
        assert model.input_cost == 0.01
        assert model.output_cost == 0.02


class TestPrebuiltResponseClasses:
    """Test the prebuilt response base classes."""

    def test_timestamped_response(self):
        """Test TimestampedResponse base class."""

        class TestResponse(TimestampedResponse):
            message: str

        now = datetime.now(timezone.utc)
        response = TestResponse(message="hello", created_at=now)

        assert response.message == "hello"
        assert response.created_at == now
        assert response.updated_at is None

    def test_provider_response(self):
        """Test ProviderResponse with AI provider fields."""

        class TestResponse(ProviderResponse):
            result: str

        response = TestResponse(
            result="success",
            provider="openai",
            model="gpt-4",
            prompt_tokens=10,
            total_cost_usd=0.001,
        )

        assert response.result == "success"
        assert response.provider == "openai"
        assert response.model == "gpt-4"
        assert response.prompt_tokens == 10
        assert response.total_cost_usd == 0.001

    def test_paginated_response(self):
        """Test PaginatedResponse base class."""

        class TestResponse(PaginatedResponse):
            items: list

        response = TestResponse(items=[1, 2, 3], total_count=100, page=0, page_size=3)

        assert response.items == [1, 2, 3]
        assert response.total_count == 100
        assert response.page == 0
        assert response.page_size == 3

    def test_full_context_response(self):
        """Test FullContextResponse with all context fields."""

        class TestResponse(FullContextResponse):
            data: str

        now = datetime.now(timezone.utc)
        response = TestResponse(
            data="test",
            request_id="req_123",
            user_id="user_456",
            organization_id="org_789",
            environment="production",
            created_at=now,
            metadata={"source": "api"},
        )

        assert response.data == "test"
        assert response.request_id == "req_123"
        assert response.user_id == "user_456"
        assert response.organization_id == "org_789"
        assert response.environment == "production"
        assert response.created_at == now
        assert response.metadata == {"source": "api"}


class TestSerialization:
    """Test serialization behavior of mixins."""

    def test_json_serialization(self):
        """Test that models with mixins serialize correctly."""

        class TestModel(TimestampMixin, MetadataMixin):
            name: str

        now = datetime.now(timezone.utc)
        model = TestModel(name="test", created_at=now, metadata={"key": "value"})

        # Test dict serialization
        data = model.model_dump()
        assert data["name"] == "test"
        assert "created_at" in data
        assert data["metadata"] == {"key": "value"}
        assert data["updated_at"] is None

        # Test JSON serialization
        json_str = model.model_dump_json()
        assert isinstance(json_str, str)
        assert "test" in json_str

    def test_optional_field_serialization(self):
        """Test that None values for optional fields are handled correctly."""

        class TestModel(TokenUsageMixin, CostTrackingMixin):
            name: str

        model = TestModel(name="test")
        data = model.model_dump()

        # Optional fields should be None
        assert data["prompt_tokens"] is None
        assert data["completion_tokens"] is None
        assert data["total_tokens"] is None
        assert data["input_cost"] is None
        assert data["output_cost"] is None
        assert data["total_cost_usd"] is None


class TestValidation:
    """Test Pydantic validation behavior with mixins."""

    def test_required_field_validation(self):
        """Test that required fields in mixins are validated."""

        class TestModel(TimestampMixin):
            name: str

        # Should fail without required created_at
        with pytest.raises(ValueError):
            TestModel(name="test")

        # Should succeed with created_at
        now = datetime.now(timezone.utc)
        model = TestModel(name="test", created_at=now)
        assert model.created_at == now

    def test_type_validation(self):
        """Test that field types are validated correctly."""

        class TestModel(TokenUsageMixin):
            name: str

        # Should fail with wrong type for tokens
        with pytest.raises(ValueError):
            TestModel(name="test", prompt_tokens="invalid")

        # Should succeed with correct type
        model = TestModel(name="test", prompt_tokens=100)
        assert model.prompt_tokens == 100

    def test_optional_field_defaults(self):
        """Test that optional fields get proper default values."""

        class TestModel(MetadataMixin, ProviderMixin):
            name: str

        model = TestModel(name="test")

        # All optional fields should be None by default
        assert model.metadata is None
        assert model.tags is None
        assert model.provider is None
        assert model.model is None


class TestIndustryStandardBaseResponse:
    """Test the industry standard BaseResponse model with clean response.brokle.* pattern."""

    def test_base_response_clean_pattern(self):
        """Test BaseResponse follows industry standard pattern like AWS, Google Cloud."""

        # Create BrokleMetadata with comprehensive platform data
        now = datetime.now(timezone.utc)
        metadata = BrokleMetadata(
            request_id="req_123",
            provider="openai",
            model_used="gpt-4",
            latency_ms=150.5,
            cost_usd=0.002,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cache_hit=True,
            quality_score=0.95,
            routing_reason="cost_optimized",
            routing_decision={"strategy": "cost_optimized"},
            cached=True,
            evaluation_scores={"relevance": 0.9, "accuracy": 0.85},
            created_at=now,
            custom_tags={"env": "production"},
        )

        # Clean BaseResponse with only brokle metadata field
        response = BaseResponse(brokle=metadata)

        # Test industry standard pattern: response.brokle.*
        assert response.brokle is not None
        assert response.brokle.request_id == "req_123"
        assert response.brokle.provider == "openai"
        assert response.brokle.model_used == "gpt-4"
        assert response.brokle.latency_ms == 150.5
        assert response.brokle.cost_usd == 0.002
        assert response.brokle.input_tokens == 100
        assert response.brokle.output_tokens == 50
        assert response.brokle.total_tokens == 150
        assert response.brokle.cache_hit is True
        assert response.brokle.quality_score == 0.95
        assert response.brokle.routing_reason == "cost_optimized"
        assert response.brokle.routing_decision == {"strategy": "cost_optimized"}
        assert response.brokle.cached is True
        assert response.brokle.evaluation_scores == {"relevance": 0.9, "accuracy": 0.85}
        assert response.brokle.created_at == now
        assert response.brokle.custom_tags == {"env": "production"}

    def test_base_response_minimal_clean(self):
        """Test BaseResponse works with minimal fields - clean architecture."""

        response = BaseResponse()

        # Only the brokle field should exist
        assert response.brokle is None

    def test_base_response_direct_access_deprecated(self):
        """Test that direct field access works but is deprecated with warnings."""
        import warnings

        response = BaseResponse()

        # These fields now work via compatibility properties but issue deprecation warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # All should return None when no brokle metadata
            assert response.request_id is None
            assert response.provider is None
            assert response.cache_hit is None
            assert response.quality_score is None
            assert response.cost_usd is None
            assert response.input_tokens is None
            assert response.routing_reason is None

            # Verify deprecation warnings were issued
            assert len(w) == 7
            for warning in w:
                assert issubclass(warning.category, DeprecationWarning)
                assert "deprecated" in str(warning.message)

    def test_brokle_metadata_comprehensive(self):
        """Test that BrokleMetadata has complete platform metadata coverage."""

        metadata = BrokleMetadata(
            # Request tracking
            request_id="req_123",
            # Provider and routing info
            provider="anthropic",
            model_used="claude-3-sonnet",
            routing_strategy="quality_optimized",
            routing_reason="High quality required",
            routing_decision={"score": 0.95, "alternatives": ["openai"]},
            # Performance metrics
            latency_ms=200.5,
            # Complete cost tracking
            cost_usd=0.005,
            cost_per_token=0.0001,
            input_cost_usd=0.003,
            output_cost_usd=0.002,
            # Token usage
            input_tokens=150,
            output_tokens=75,
            total_tokens=225,
            # Caching info
            cache_hit=True,
            cache_similarity_score=0.95,
            cached=True,
            # Quality assessment
            quality_score=0.92,
            evaluation_scores={"relevance": 0.95, "accuracy": 0.89},
            # Platform insights
            optimization_applied=["semantic_cache", "cost_optimization"],
            cost_savings_usd=0.003,
            # Metadata
            created_at=datetime.now(timezone.utc),
            custom_tags={"project": "test", "env": "staging"},
        )

        # Verify all comprehensive fields are accessible
        assert metadata.request_id == "req_123"
        assert metadata.provider == "anthropic"
        assert metadata.model_used == "claude-3-sonnet"
        assert metadata.routing_strategy == "quality_optimized"
        assert metadata.routing_reason == "High quality required"
        assert metadata.routing_decision == {"score": 0.95, "alternatives": ["openai"]}
        assert metadata.latency_ms == 200.5
        assert metadata.cost_usd == 0.005
        assert metadata.cost_per_token == 0.0001
        assert metadata.input_cost_usd == 0.003
        assert metadata.output_cost_usd == 0.002
        assert metadata.input_tokens == 150
        assert metadata.output_tokens == 75
        assert metadata.total_tokens == 225
        assert metadata.cache_hit is True
        assert metadata.cache_similarity_score == 0.95
        assert metadata.cached is True
        assert metadata.quality_score == 0.92
        assert metadata.evaluation_scores == {"relevance": 0.95, "accuracy": 0.89}
        assert metadata.optimization_applied == ["semantic_cache", "cost_optimization"]
        assert metadata.cost_savings_usd == 0.003
        assert metadata.custom_tags == {"project": "test", "env": "staging"}


class TestBackwardCompatibility:
    """Test backward compatibility properties on BaseResponse."""

    def test_compatibility_properties_work(self):
        """Test that legacy properties still work but warn."""
        import warnings

        metadata = BrokleMetadata(
            request_id="req_123",
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

        # Test that legacy properties work with warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            assert response.request_id == "req_123"
            assert response.provider == "openai"
            assert response.cost_usd == 0.002
            assert response.cache_hit is True
            assert response.quality_score == 0.95
            assert response.input_tokens == 100
            assert response.output_tokens == 50
            assert response.total_tokens == 150
            assert response.latency_ms == 200.5
            assert response.routing_reason == "cost_optimized"

            # Verify deprecation warnings were issued
            assert len(w) == 10
            for warning in w:
                assert issubclass(warning.category, DeprecationWarning)
                assert "deprecated" in str(warning.message)
                assert "response.brokle." in str(warning.message)

    def test_compatibility_properties_none_when_no_brokle(self):
        """Test that legacy properties return None when brokle is None."""
        import warnings

        response = BaseResponse()  # No brokle metadata

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore deprecation warnings for this test

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

    def test_new_pattern_no_warnings(self):
        """Test that new response.brokle.* pattern generates no warnings."""
        import warnings

        metadata = BrokleMetadata(
            request_id="req_456",
            provider="anthropic",
            cost_usd=0.005,
            cache_hit=False,
            quality_score=0.88,
        )

        response = BaseResponse(brokle=metadata)

        # Test that new pattern works without warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            if response.brokle:
                assert response.brokle.request_id == "req_456"
                assert response.brokle.provider == "anthropic"
                assert response.brokle.cost_usd == 0.005
                assert response.brokle.cache_hit is False
                assert response.brokle.quality_score == 0.88

            # Verify no warnings were issued
            assert len(w) == 0
