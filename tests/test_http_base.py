"""
Unit tests for HTTPBase class.

Tests authentication, headers, request preparation, and error handling.
"""

import pytest
import httpx
from unittest.mock import Mock, patch

from brokle.http.base import HTTPBase, BrokleResponse
from brokle.config import Config
from brokle.exceptions import AuthenticationError, APIError


class TestHTTPBase:
    """Test HTTPBase functionality."""

    def test_init_with_valid_credentials(self):
        """Test initialization with valid credentials."""
        base = HTTPBase(
            api_key="ak_test123",
            host="http://localhost:8080",
            project_id="proj_test",
            environment="test"
        )

        assert base.config.api_key == "ak_test123"
        assert base.config.host == "http://localhost:8080"
        assert base.config.project_id == "proj_test"
        assert base.config.environment == "test"
        assert base.config.timeout == 30  # default

    def test_init_missing_api_key(self):
        """Test initialization fails without API key."""
        with pytest.raises(AuthenticationError, match="API key is required"):
            HTTPBase(project_id="proj_test")

    def test_init_missing_project_id(self):
        """Test initialization fails without project ID."""
        with pytest.raises(AuthenticationError, match="Project ID is required"):
            HTTPBase(api_key="ak_test123")

    def test_build_headers(self):
        """Test default headers are built correctly."""
        base = HTTPBase(
            api_key="ak_test123",
            project_id="proj_test",
            environment="production"
        )

        headers = base.default_headers

        assert headers["Content-Type"] == "application/json"
        assert headers["User-Agent"] == "brokle-python/0.1.0"
        assert headers["X-API-Key"] == "ak_test123"
        assert headers["X-Project-ID"] == "proj_test"
        assert headers["X-Environment"] == "production"
        assert headers["X-SDK-Version"] == "0.1.0"

    def test_prepare_url(self):
        """Test URL preparation handles various endpoint formats."""
        base = HTTPBase(
            api_key="ak_test123",
            host="http://localhost:8080",
            project_id="proj_test"
        )

        # Test various endpoint formats
        assert base._prepare_url("/v1/chat/completions") == "http://localhost:8080/v1/chat/completions"
        assert base._prepare_url("v1/chat/completions") == "http://localhost:8080/v1/chat/completions"
        assert base._prepare_url("/v1/embeddings") == "http://localhost:8080/v1/embeddings"

        # Test with trailing slash in host
        base.config.host = "http://localhost:8080/"
        assert base._prepare_url("/v1/models") == "http://localhost:8080/v1/models"

    def test_prepare_request_kwargs(self):
        """Test request kwargs preparation."""
        base = HTTPBase(
            api_key="ak_test123",
            project_id="proj_test",
            timeout=60
        )

        # Test with no additional kwargs
        kwargs = base._prepare_request_kwargs()

        assert kwargs["headers"]["X-API-Key"] == "ak_test123"
        assert kwargs["timeout"] == 60
        assert "X-Request-Timestamp" in kwargs["headers"]

        # Test with additional headers
        kwargs = base._prepare_request_kwargs(
            headers={"X-Custom": "value"},
            json={"model": "gpt-4"}
        )

        assert kwargs["headers"]["X-API-Key"] == "ak_test123"  # Default header
        assert kwargs["headers"]["X-Custom"] == "value"  # Custom header
        assert kwargs["json"]["model"] == "gpt-4"  # Other kwargs preserved
        assert kwargs["timeout"] == 60

    def test_handle_http_error_401(self):
        """Test authentication error handling."""
        base = HTTPBase(api_key="ak_test123", project_id="proj_test")

        # Mock 401 response
        response = Mock(spec=httpx.Response)
        response.status_code = 401
        response.text = "Unauthorized"

        with pytest.raises(AuthenticationError, match="Authentication failed"):
            base._handle_http_error(response)

    def test_handle_http_error_429(self):
        """Test rate limit error handling."""
        base = HTTPBase(api_key="ak_test123", project_id="proj_test")

        # Mock 429 response
        response = Mock(spec=httpx.Response)
        response.status_code = 429
        response.text = "Rate limit exceeded"

        with pytest.raises(APIError, match="Rate limit exceeded"):
            base._handle_http_error(response)

    def test_handle_http_error_generic(self):
        """Test generic API error handling."""
        base = HTTPBase(api_key="ak_test123", project_id="proj_test")

        # Mock 400 response with JSON error
        response = Mock(spec=httpx.Response)
        response.status_code = 400
        response.text = "Bad request"
        response.json.return_value = {
            "error": {"message": "Invalid model parameter"}
        }

        with pytest.raises(APIError, match="Invalid model parameter"):
            base._handle_http_error(response)

    def test_process_response_success(self):
        """Test successful response processing."""
        base = HTTPBase(api_key="ak_test123", project_id="proj_test")

        # Mock successful response
        response = Mock(spec=httpx.Response)
        response.status_code = 200
        response.json.return_value = {
            "choices": [{"message": {"content": "Hello!"}}],
            "brokle_metadata": {
                "provider": "openai",
                "request_id": "req_123",
                "latency_ms": 150
            }
        }

        result = base._process_response(response)

        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["brokle_metadata"]["provider"] == "openai"

    def test_process_response_http_error(self):
        """Test response processing with HTTP error."""
        base = HTTPBase(api_key="ak_test123", project_id="proj_test")

        # Mock error response
        response = Mock(spec=httpx.Response)
        response.status_code = 400
        response.text = "Bad request"
        response.json.return_value = {"error": {"message": "Invalid request"}}

        with pytest.raises(APIError, match="Invalid request"):
            base._process_response(response)

    def test_process_response_invalid_json(self):
        """Test response processing with invalid JSON."""
        base = HTTPBase(api_key="ak_test123", project_id="proj_test")

        # Mock response with invalid JSON
        response = Mock(spec=httpx.Response)
        response.status_code = 200
        response.json.side_effect = ValueError("Invalid JSON")

        with pytest.raises(APIError, match="Failed to parse response JSON"):
            base._process_response(response)

    def test_environment_defaults(self):
        """Test environment variable defaults."""
        with patch.dict("os.environ", {
            "BROKLE_API_KEY": "ak_env_key",
            "BROKLE_PROJECT_ID": "proj_env",
            "BROKLE_HOST": "https://api.brokle.ai",
            "BROKLE_ENVIRONMENT": "staging"
        }):
            base = HTTPBase()

            assert base.config.api_key == "ak_env_key"
            assert base.config.project_id == "proj_env"
            assert base.config.host == "https://api.brokle.ai"
            assert base.config.environment == "staging"

    def test_parameter_override_environment(self):
        """Test parameters override environment variables."""
        with patch.dict("os.environ", {
            "BROKLE_API_KEY": "ak_env_key",
            "BROKLE_PROJECT_ID": "proj_env"
        }):
            base = HTTPBase(
                api_key="ak_param_key",
                project_id="proj_param"
            )

            # Parameters should override environment
            assert base.config.api_key == "ak_param_key"
            assert base.config.project_id == "proj_param"


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
            quality_score=0.95
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
            provider="anthropic",
            request_id="req_456",
            latency_ms=200
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