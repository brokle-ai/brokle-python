"""Tests for custom exceptions."""

import pytest

from brokle.exceptions import (
    AuthenticationError,
    BrokleError,
    CacheError,
    EvaluationError,
    ProviderError,
    QuotaExceededError,
    RateLimitError,
    ValidationError,
)


class TestBrokleExceptions:
    """Test custom exception classes."""

    def test_brokle_error_base(self):
        """Test base BrokleError."""
        error = BrokleError("Test error message")
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.status_code == 500
        assert error.error_code == "brokle_error"
        assert error.details is None

    def test_brokle_error_with_details(self):
        """Test BrokleError with details."""
        details = {"request_id": "req_123", "timestamp": "2024-01-15T10:30:00Z"}
        error = BrokleError(
            "Test error", status_code=400, error_code="custom_error", details=details
        )
        assert error.message == "Test error"
        assert error.status_code == 400
        assert error.error_code == "custom_error"
        assert error.details == details

    def test_authentication_error(self):
        """Test AuthenticationError."""
        error = AuthenticationError("Invalid API key")
        assert str(error) == "Invalid API key"
        assert error.status_code == 401
        assert error.error_code == "authentication_error"
        assert isinstance(error, BrokleError)

    def test_rate_limit_error(self):
        """Test RateLimitError."""
        error = RateLimitError("Rate limit exceeded")
        assert str(error) == "Rate limit exceeded"
        assert error.status_code == 429
        assert error.error_code == "rate_limit_error"
        assert isinstance(error, BrokleError)

    def test_rate_limit_error_with_retry_after(self):
        """Test RateLimitError with retry_after."""
        error = RateLimitError("Rate limit exceeded", retry_after=60)
        assert error.retry_after == 60
        assert error.status_code == 429

    def test_quota_exceeded_error(self):
        """Test QuotaExceededError."""
        error = QuotaExceededError("Monthly quota exceeded")
        assert str(error) == "Monthly quota exceeded"
        assert error.status_code == 429
        assert error.error_code == "quota_exceeded"
        assert isinstance(error, BrokleError)

    def test_quota_exceeded_error_with_quota_info(self):
        """Test QuotaExceededError with quota information."""
        quota_info = {
            "current_usage": 950,
            "quota_limit": 1000,
            "reset_time": "2024-02-01T00:00:00Z",
        }
        error = QuotaExceededError("Monthly quota exceeded", quota_info=quota_info)
        assert error.quota_info == quota_info
        assert error.status_code == 429

    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError("Invalid request format")
        assert str(error) == "Invalid request format"
        assert error.status_code == 400
        assert error.error_code == "validation_error"
        assert isinstance(error, BrokleError)

    def test_validation_error_with_field_errors(self):
        """Test ValidationError with field errors."""
        field_errors = {
            "model": ["Model is required"],
            "messages": ["Messages cannot be empty"],
        }
        error = ValidationError("Request validation failed", field_errors=field_errors)
        assert error.field_errors == field_errors
        assert error.status_code == 400

    def test_provider_error(self):
        """Test ProviderError."""
        error = ProviderError("OpenAI API error")
        assert str(error) == "OpenAI API error"
        assert error.status_code == 502
        assert error.error_code == "provider_error"
        assert isinstance(error, BrokleError)

    def test_provider_error_with_provider_details(self):
        """Test ProviderError with provider details."""
        provider_details = {
            "provider": "openai",
            "provider_error_code": "invalid_request_error",
            "provider_message": "Invalid model specified",
        }
        error = ProviderError(
            "Provider request failed", provider_details=provider_details
        )
        assert error.provider_details == provider_details
        assert error.status_code == 502

    def test_cache_error(self):
        """Test CacheError."""
        error = CacheError("Cache lookup failed")
        assert str(error) == "Cache lookup failed"
        assert error.status_code == 500
        assert error.error_code == "cache_error"
        assert isinstance(error, BrokleError)

    def test_cache_error_with_cache_details(self):
        """Test CacheError with cache details."""
        cache_details = {
            "cache_key": "embedding_abc123",
            "cache_type": "semantic",
            "operation": "get",
        }
        error = CacheError("Cache operation failed", cache_details=cache_details)
        assert error.cache_details == cache_details
        assert error.status_code == 500

    def test_evaluation_error(self):
        """Test EvaluationError."""
        error = EvaluationError("Response evaluation failed")
        assert str(error) == "Response evaluation failed"
        assert error.status_code == 500
        assert error.error_code == "evaluation_error"
        assert isinstance(error, BrokleError)

    def test_evaluation_error_with_evaluation_details(self):
        """Test EvaluationError with evaluation details."""
        evaluation_details = {
            "evaluation_id": "eval_123",
            "evaluator": "quality_scorer",
            "input_tokens": 150,
            "output_tokens": 75,
        }
        error = EvaluationError(
            "Quality evaluation failed", evaluation_details=evaluation_details
        )
        assert error.evaluation_details == evaluation_details
        assert error.status_code == 500

    def test_exception_inheritance(self):
        """Test that all exceptions inherit from BrokleError."""
        exceptions = [
            AuthenticationError("test"),
            RateLimitError("test"),
            QuotaExceededError("test"),
            ValidationError("test"),
            ProviderError("test"),
            CacheError("test"),
            EvaluationError("test"),
        ]

        for exc in exceptions:
            assert isinstance(exc, BrokleError)
            assert isinstance(exc, Exception)

    def test_exception_repr(self):
        """Test exception repr method."""
        error = AuthenticationError("Invalid API key")
        repr_str = repr(error)
        assert "AuthenticationError" in repr_str
        assert "Invalid API key" in repr_str

    def test_exception_equality(self):
        """Test exception equality."""
        error1 = AuthenticationError("Invalid API key")
        error2 = AuthenticationError("Invalid API key")
        error3 = AuthenticationError("Different message")

        # Same message should be equal
        assert error1 == error2

        # Different message should not be equal
        assert error1 != error3

        # Different exception types should not be equal
        rate_limit_error = RateLimitError("Invalid API key")
        assert error1 != rate_limit_error

    def test_exception_hash(self):
        """Test exception hash method."""
        error1 = AuthenticationError("Invalid API key")
        error2 = AuthenticationError("Invalid API key")

        # Same errors should have same hash
        assert hash(error1) == hash(error2)

        # Should be usable in sets
        error_set = {error1, error2}
        assert len(error_set) == 1
