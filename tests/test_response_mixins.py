"""
Tests for response model mixins and base classes.

These tests ensure that our mixins work correctly with Pydantic validation,
serialization, and multiple inheritance patterns.
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from brokle.types.responses.base import (
    TimestampMixin,
    MetadataMixin,
    TokenUsageMixin,
    CostTrackingMixin,
    PaginationMixin,
    ProviderMixin,
    RequestTrackingMixin,
    OrganizationContextMixin,
    StatusMixin,
    BrokleResponseBase,
    TimestampedResponse,
    ProviderResponse,
    PaginatedResponse,
    TrackedResponse,
    FullContextResponse,
)


class TestIndividualMixins:
    """Test each mixin individually."""

    def test_timestamp_mixin(self):
        """Test TimestampMixin fields and validation."""

        class TestModel(TimestampMixin):
            name: str

        # Test with required created_at
        now = datetime.utcnow()
        model = TestModel(name="test", created_at=now)

        assert model.created_at == now
        assert model.updated_at is None

        # Test with both timestamps
        updated = datetime.utcnow()
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
            name="test",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
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
            name="test",
            input_cost=0.01,
            output_cost=0.02,
            total_cost_usd=0.03
        )

        assert model.input_cost == 0.01
        assert model.output_cost == 0.02
        assert model.total_cost_usd == 0.03

    def test_pagination_mixin(self):
        """Test PaginationMixin required fields."""

        class TestModel(PaginationMixin):
            name: str

        model = TestModel(
            name="test",
            total_count=100,
            page=0,
            page_size=20
        )

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
            CostTrackingMixin
        ):
            name: str

        now = datetime.utcnow()
        model = TestModel(
            name="test",
            created_at=now,
            provider="openai",
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            input_cost=0.01,
            output_cost=0.02
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

        now = datetime.utcnow()
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
            total_cost_usd=0.001
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

        response = TestResponse(
            items=[1, 2, 3],
            total_count=100,
            page=0,
            page_size=3
        )

        assert response.items == [1, 2, 3]
        assert response.total_count == 100
        assert response.page == 0
        assert response.page_size == 3

    def test_full_context_response(self):
        """Test FullContextResponse with all context fields."""

        class TestResponse(FullContextResponse):
            data: str

        now = datetime.utcnow()
        response = TestResponse(
            data="test",
            request_id="req_123",
            user_id="user_456",
            organization_id="org_789",
            project_id="proj_abc",
            environment="production",
            created_at=now,
            metadata={"source": "api"}
        )

        assert response.data == "test"
        assert response.request_id == "req_123"
        assert response.user_id == "user_456"
        assert response.organization_id == "org_789"
        assert response.project_id == "proj_abc"
        assert response.environment == "production"
        assert response.created_at == now
        assert response.metadata == {"source": "api"}


class TestSerialization:
    """Test serialization behavior of mixins."""

    def test_json_serialization(self):
        """Test that models with mixins serialize correctly."""

        class TestModel(TimestampMixin, MetadataMixin):
            name: str

        now = datetime.utcnow()
        model = TestModel(
            name="test",
            created_at=now,
            metadata={"key": "value"}
        )

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
        now = datetime.utcnow()
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