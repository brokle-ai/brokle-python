"""
Unit tests for HTTPBase class.

Tests authentication, headers, request preparation, and error handling.
"""

from unittest.mock import Mock, patch

import httpx
import pytest

from brokle.config import Config
from brokle.exceptions import APIError, AuthenticationError
from brokle.http.base import BrokleResponse, HTTPBase


class TestHTTPBase:
    """Test HTTPBase functionality."""

    def test_init_with_valid_credentials(self):
        """Test initialization with valid credentials."""
        base = HTTPBase(
            api_key="bk_test_secret", host="http://localhost:8080", environment="test"
        )

        assert base.config.api_key == "bk_test_secret"
        assert base.config.environment == "test"
        assert base.config.host == "http://localhost:8080"
        assert base.config.environment == "test"
        assert base.config.timeout == 30  # default

    def test_init_missing_api_key(self, caplog):
        """Test initialization creates disabled client without API key."""
        import logging

        with caplog.at_level(logging.WARNING, logger="brokle"):
            base = HTTPBase(environment="test")

        # Should be disabled and log warning
        assert getattr(base, '_disabled', False) is True
        assert "Authentication error:" in caplog.text
        assert "initialized without api_key" in caplog.text
        assert "Client will be disabled" in caplog.text

    def test_build_headers(self):
        """Test default headers are built correctly."""
        base = HTTPBase(api_key="bk_test_secret", environment="production")

        headers = base.default_headers

        from brokle._version import __version__

        assert headers["Content-Type"] == "application/json"
        assert headers["User-Agent"] == f"brokle-python/{__version__}"
        assert headers["X-API-Key"] == "bk_test_secret"
        assert headers["X-Environment"] == "production"
        assert headers["X-SDK-Version"] == __version__

    def test_prepare_url(self):
        """Test URL preparation handles various endpoint formats."""
        base = HTTPBase(api_key="bk_test_secret", host="http://localhost:8080")

        # Test various endpoint formats
        assert (
            base._prepare_url("/v1/chat/completions")
            == "http://localhost:8080/v1/chat/completions"
        )
        assert (
            base._prepare_url("v1/chat/completions")
            == "http://localhost:8080/v1/chat/completions"
        )
        assert (
            base._prepare_url("/v1/embeddings") == "http://localhost:8080/v1/embeddings"
        )

        # Test with trailing slash in host
        base.config.host = "http://localhost:8080/"
        assert base._prepare_url("/v1/models") == "http://localhost:8080/v1/models"

    def test_prepare_request_kwargs(self):
        """Test request kwargs preparation."""
        base = HTTPBase(api_key="bk_test_secret", timeout=60)

        # Test with no additional kwargs
        kwargs = base._prepare_request_kwargs()

        assert kwargs["headers"]["X-API-Key"] == "bk_test_secret"
        assert kwargs["timeout"] == 60
        assert "X-Request-Timestamp" in kwargs["headers"]

        # Test with additional headers
        kwargs = base._prepare_request_kwargs(
            headers={"X-Custom": "value"}, json={"model": "gpt-4"}
        )

        assert kwargs["headers"]["X-API-Key"] == "bk_test_secret"  # Default header
        assert kwargs["headers"]["X-Custom"] == "value"  # Custom header
        assert kwargs["json"]["model"] == "gpt-4"  # Other kwargs preserved
        assert kwargs["timeout"] == 60

    def test_handle_http_error_401(self):
        """Test authentication error handling."""
        base = HTTPBase(api_key="bk_test_secret")

        # Mock 401 response
        response = Mock(spec=httpx.Response)
        response.status_code = 401
        response.text = "Unauthorized"

        with pytest.raises(AuthenticationError, match="Authentication failed"):
            base._handle_http_error(response)

    def test_handle_http_error_429(self):
        """Test rate limit error handling."""
        base = HTTPBase(api_key="bk_test_secret")

        # Mock 429 response
        response = Mock(spec=httpx.Response)
        response.status_code = 429
        response.text = "Rate limit exceeded"

        with pytest.raises(APIError, match="Rate limit exceeded"):
            base._handle_http_error(response)

    def test_handle_http_error_generic(self):
        """Test generic API error handling."""
        base = HTTPBase(api_key="bk_test_secret")

        # Mock 400 response with JSON error
        response = Mock(spec=httpx.Response)
        response.status_code = 400
        response.text = "Bad request"
        response.json.return_value = {"error": {"message": "Invalid model parameter"}}

        with pytest.raises(APIError, match="Invalid model parameter"):
            base._handle_http_error(response)

    def test_process_response_success(self):
        """Test successful response processing."""
        base = HTTPBase(api_key="bk_test_secret")

        # Mock successful response
        response = Mock(spec=httpx.Response)
        response.status_code = 200
        response.json.return_value = {
            "choices": [{"message": {"content": "Hello!"}}],
            "brokle": {
                "provider": "openai",
                "request_id": "req_123",
                "latency_ms": 150,
            },
        }

        result = base._process_response(response)

        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["brokle"]["provider"] == "openai"

    def test_process_response_http_error(self):
        """Test response processing with HTTP error."""
        base = HTTPBase(api_key="bk_test_secret")

        # Mock error response
        response = Mock(spec=httpx.Response)
        response.status_code = 400
        response.text = "Bad request"
        response.json.return_value = {"error": {"message": "Invalid request"}}

        with pytest.raises(APIError, match="Invalid request"):
            base._process_response(response)

    def test_process_response_invalid_json(self):
        """Test response processing with invalid JSON."""
        base = HTTPBase(api_key="bk_test_secret")

        # Mock response with invalid JSON
        response = Mock(spec=httpx.Response)
        response.status_code = 200
        response.json.side_effect = ValueError("Invalid JSON")

        with pytest.raises(APIError, match="Failed to parse response JSON"):
            base._process_response(response)

    def test_environment_defaults(self):
        """Test environment variable defaults."""
        with patch.dict(
            "os.environ",
            {
                "BROKLE_API_KEY": "bk_env_secret",
                "BROKLE_HOST": "https://api.brokle.ai",
                "BROKLE_ENVIRONMENT": "staging",
            },
        ):
            base = HTTPBase()

            assert base.config.api_key == "bk_env_secret"
            assert base.config.host == "https://api.brokle.ai"
            assert base.config.environment == "staging"

    def test_parameter_override_environment(self):
        """Test parameters override environment variables."""
        with patch.dict("os.environ", {"BROKLE_API_KEY": "bk_env_secret"}):
            base = HTTPBase(api_key="bk_param_secret")

            # Parameters should override environment
            assert base.config.api_key == "bk_param_secret"


class TestBrokleResponse:
    """Test BrokleResponse model."""

    def test_brokle_metadata_model(self):
        """Test BrokleMetadata model validation."""
        metadata = BrokleResponse.BrokleMetadata(
            provider="openai",
            request_id="req_123",
            latency_ms=150,
            cost_usd=0.002,
            tokens_used=50,
            cache_hit=True,
            cache_key="cache_abc",
            routing_strategy="cost_optimized",
            quality_score=0.95,
        )

        assert metadata.provider == "openai"
        assert metadata.request_id == "req_123"
        assert metadata.latency_ms == 150
        assert metadata.cost_usd == 0.002
        assert metadata.tokens_used == 50
        assert metadata.cache_hit is True
        assert metadata.cache_key == "cache_abc"
        assert metadata.routing_strategy == "cost_optimized"
        assert metadata.quality_score == 0.95

    def test_brokle_metadata_required_fields(self):
        """Test BrokleMetadata required fields."""
        # Should work with only required fields
        metadata = BrokleResponse.BrokleMetadata(
            provider="anthropic", request_id="req_456", latency_ms=200
        )

        assert metadata.provider == "anthropic"
        assert metadata.request_id == "req_456"
        assert metadata.latency_ms == 200

        # Optional fields should have defaults
        assert metadata.cost_usd is None
        assert metadata.tokens_used is None
        assert metadata.cache_hit is False
        assert metadata.cache_key is None
        assert metadata.routing_strategy is None
        assert metadata.quality_score is None
