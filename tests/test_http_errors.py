"""
Tests for HTTP error handling with non-dict JSON responses.

Ensures error classes handle various JSON response types gracefully:
- dict (expected format)
- list (proxy/CDN responses)
- str (plain text responses)
- int, float, bool (edge cases)
- None (no body)
"""

from unittest.mock import MagicMock

import pytest

from brokle._http.client import _check_response_status
from brokle._http.errors import (
    AuthenticationError,
    BrokleError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
    _safe_extract_error_message,
    _safe_get_value,
    raise_for_status,
)


# =============================================================================
# Helper Function Tests: _safe_extract_error_message
# =============================================================================


class TestSafeExtractErrorMessage:
    """Tests for _safe_extract_error_message helper function."""

    def test_dict_with_nested_error_message(self):
        """Standard dict response with error.message path."""
        body = {"error": {"message": "Invalid credentials"}}
        result = _safe_extract_error_message(body, default="default")
        assert result == "Invalid credentials"

    def test_dict_with_missing_error_key(self):
        """Dict response without error key returns default."""
        body = {"status": "failed"}
        result = _safe_extract_error_message(body, default="default message")
        assert result == "default message"

    def test_dict_with_missing_message_key(self):
        """Dict response with error but no message returns default."""
        body = {"error": {"code": "AUTH_FAILED"}}
        result = _safe_extract_error_message(body, default="default message")
        assert result == "default message"

    def test_dict_with_non_string_message(self):
        """Dict with non-string message value returns default."""
        body = {"error": {"message": 12345}}
        result = _safe_extract_error_message(body, default="default message")
        assert result == "default message"

    def test_dict_with_message_as_dict(self):
        """Dict with message as another dict returns default."""
        body = {"error": {"message": {"detail": "nested"}}}
        result = _safe_extract_error_message(body, default="default message")
        assert result == "default message"

    def test_dict_with_error_as_string(self):
        """Dict with error as string (not dict) returns default."""
        body = {"error": "simple error string"}
        result = _safe_extract_error_message(body, default="default message")
        assert result == "default message"

    def test_list_response_returns_default(self):
        """List JSON response returns default."""
        body = ["error1", "error2"]
        result = _safe_extract_error_message(body, default="list response")
        assert result == "list response"

    def test_string_response_returns_default(self):
        """String JSON response returns default."""
        body = "unauthorized"
        result = _safe_extract_error_message(body, default="string response")
        assert result == "string response"

    def test_int_response_returns_default(self):
        """Integer JSON response returns default."""
        body = 401
        result = _safe_extract_error_message(body, default="int response")
        assert result == "int response"

    def test_float_response_returns_default(self):
        """Float JSON response returns default."""
        body = 3.14
        result = _safe_extract_error_message(body, default="float response")
        assert result == "float response"

    def test_bool_true_returns_default(self):
        """Boolean True returns default."""
        body = True
        result = _safe_extract_error_message(body, default="bool response")
        assert result == "bool response"

    def test_bool_false_returns_default(self):
        """Boolean False returns default."""
        body = False
        result = _safe_extract_error_message(body, default="bool response")
        assert result == "bool response"

    def test_none_response_returns_default(self):
        """None response returns default."""
        body = None
        result = _safe_extract_error_message(body, default="none response")
        assert result == "none response"

    def test_empty_dict_returns_default(self):
        """Empty dict returns default."""
        body = {}
        result = _safe_extract_error_message(body, default="empty dict")
        assert result == "empty dict"

    def test_custom_nested_path(self):
        """Custom nested path extraction works."""
        body = {"data": {"detail": "custom error"}}
        result = _safe_extract_error_message(
            body, default="default", nested_path=("data", "detail")
        )
        assert result == "custom error"

    def test_custom_nested_path_missing(self):
        """Custom nested path missing returns default."""
        body = {"data": {"other": "value"}}
        result = _safe_extract_error_message(
            body, default="default", nested_path=("data", "detail")
        )
        assert result == "default"


# =============================================================================
# Helper Function Tests: _safe_get_value
# =============================================================================


class TestSafeGetValue:
    """Tests for _safe_get_value helper function."""

    def test_dict_with_key_present(self):
        """Dict with key present returns value."""
        body = {"retry_after": 30}
        result = _safe_get_value(body, "retry_after")
        assert result == 30

    def test_dict_with_key_missing(self):
        """Dict with key missing returns default."""
        body = {"other": "value"}
        result = _safe_get_value(body, "retry_after")
        assert result is None

    def test_dict_with_custom_default(self):
        """Dict with key missing returns custom default."""
        body = {"other": "value"}
        result = _safe_get_value(body, "retry_after", default=60)
        assert result == 60

    def test_list_returns_default(self):
        """List response returns default."""
        body = ["item1", "item2"]
        result = _safe_get_value(body, "key")
        assert result is None

    def test_string_returns_default(self):
        """String response returns default."""
        body = "some string"
        result = _safe_get_value(body, "key")
        assert result is None

    def test_int_returns_default(self):
        """Integer response returns default."""
        body = 123
        result = _safe_get_value(body, "key")
        assert result is None

    def test_none_returns_default(self):
        """None response returns default."""
        body = None
        result = _safe_get_value(body, "key")
        assert result is None

    def test_bool_returns_default(self):
        """Boolean response returns default."""
        body = True
        result = _safe_get_value(body, "key")
        assert result is None


# =============================================================================
# AuthenticationError.from_response Tests
# =============================================================================


class TestAuthenticationErrorFromResponse:
    """Tests for AuthenticationError.from_response with various JSON types."""

    def test_dict_response_extracts_message(self):
        """Standard dict response extracts error message."""
        body = {"error": {"message": "Invalid API key"}}
        error = AuthenticationError.from_response(401, body)
        assert "Invalid API key" in error.message
        assert error.details["response"] == body

    def test_list_response_uses_default(self):
        """List response uses default message without crashing."""
        body = ["unauthorized", "access denied"]
        error = AuthenticationError.from_response(401, body)
        assert "Unknown authentication error" in error.message
        assert error.details["response"] == body

    def test_string_response_uses_default(self):
        """String response uses default message."""
        body = "Forbidden"
        error = AuthenticationError.from_response(403, body)
        assert "Unknown authentication error" in error.message
        assert error.details["response"] == body

    def test_int_response_uses_default(self):
        """Integer response uses default message."""
        body = 401
        error = AuthenticationError.from_response(401, body)
        assert "Unknown authentication error" in error.message

    def test_none_response_uses_default(self):
        """None response uses default message."""
        error = AuthenticationError.from_response(401, None)
        assert "Unknown authentication error" in error.message

    def test_bool_response_uses_default(self):
        """Boolean response uses default message."""
        body = False
        error = AuthenticationError.from_response(401, body)
        assert "Unknown authentication error" in error.message

    def test_preserves_api_key_prefix(self):
        """API key prefix is preserved in message."""
        error = AuthenticationError.from_response(401, None, api_key_prefix="bk_abc")
        assert "bk_abc" in error.message


# =============================================================================
# ValidationError.from_response Tests
# =============================================================================


class TestValidationErrorFromResponse:
    """Tests for ValidationError.from_response with various JSON types."""

    def test_dict_response_extracts_message(self):
        """Standard dict response extracts error message."""
        body = {"error": {"message": "Field 'email' is required"}}
        error = ValidationError.from_response(body)
        assert "Field 'email' is required" in error.message
        assert error.details["response"] == body

    def test_list_response_uses_default(self):
        """List response uses default message."""
        body = [{"field": "email", "error": "required"}]
        error = ValidationError.from_response(body)
        assert "Validation failed" in error.message
        assert error.details["response"] == body

    def test_string_response_uses_default(self):
        """String response uses default message."""
        body = "Invalid input"
        error = ValidationError.from_response(body)
        assert "Validation failed" in error.message

    def test_none_response_uses_default(self):
        """None response uses default message."""
        error = ValidationError.from_response(None)
        assert "Validation failed" in error.message

    def test_preserves_field_info(self):
        """Field info is preserved when provided."""
        error = ValidationError.from_response({}, field="email")
        assert "(field: email)" in error.message
        assert error.details["field"] == "email"


# =============================================================================
# RateLimitError.from_response Tests
# =============================================================================


class TestRateLimitErrorFromResponse:
    """Tests for RateLimitError.from_response with various JSON types."""

    def test_dict_response_with_retry_after(self):
        """Dict response with retry_after is handled."""
        body = {"retry_after": 60}
        error = RateLimitError.from_response(body, retry_after=60)
        assert "retry after 60s" in error.message
        assert error.details["retry_after"] == 60

    def test_list_response_handles_gracefully(self):
        """List response is handled gracefully."""
        body = ["rate limited"]
        error = RateLimitError.from_response(body)
        assert "Rate limit exceeded" in error.message
        assert error.details["response"] == body

    def test_none_response_handles_gracefully(self):
        """None response is handled gracefully."""
        error = RateLimitError.from_response(None)
        assert "Rate limit exceeded" in error.message


# =============================================================================
# ServerError.from_response Tests
# =============================================================================


class TestServerErrorFromResponse:
    """Tests for ServerError.from_response with various JSON types."""

    def test_dict_response_preserves_body(self):
        """Dict response body is preserved in details."""
        body = {"error": {"message": "Internal server error"}}
        error = ServerError.from_response(500, body)
        assert "Server error (HTTP 500)" in error.message
        assert error.details["response"] == body

    def test_list_response_preserves_body(self):
        """List response body is preserved in details."""
        body = ["internal error", "database timeout"]
        error = ServerError.from_response(500, body)
        assert "Server error (HTTP 500)" in error.message
        assert error.details["response"] == body

    def test_string_response_preserves_body(self):
        """String response body is preserved in details."""
        body = "Service Unavailable"
        error = ServerError.from_response(503, body)
        assert error.details["response"] == body


# =============================================================================
# raise_for_status Tests
# =============================================================================


class TestRaiseForStatus:
    """Tests for raise_for_status with various JSON types."""

    def test_success_status_does_not_raise(self):
        """Success status codes don't raise."""
        raise_for_status(200, {"data": "ok"})
        raise_for_status(201, ["created"])
        raise_for_status(204, None)

    def test_401_with_list_raises_auth_error(self):
        """401 with list response raises AuthenticationError."""
        with pytest.raises(AuthenticationError) as exc_info:
            raise_for_status(401, ["unauthorized"])
        assert "Unknown authentication error" in exc_info.value.message
        assert exc_info.value.details["response"] == ["unauthorized"]

    def test_401_with_string_raises_auth_error(self):
        """401 with string response raises AuthenticationError."""
        with pytest.raises(AuthenticationError) as exc_info:
            raise_for_status(401, "Forbidden")
        assert "Unknown authentication error" in exc_info.value.message

    def test_403_with_int_raises_auth_error(self):
        """403 with integer response raises AuthenticationError."""
        with pytest.raises(AuthenticationError) as exc_info:
            raise_for_status(403, 0)
        assert "Unknown authentication error" in exc_info.value.message

    def test_422_with_list_raises_validation_error(self):
        """422 with list response raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            raise_for_status(422, [{"field": "name", "error": "required"}])
        assert "Validation failed" in exc_info.value.message

    def test_422_with_none_raises_validation_error(self):
        """422 with None response raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            raise_for_status(422, None)
        assert "Validation failed" in exc_info.value.message

    def test_429_with_list_raises_rate_limit_error(self):
        """429 with list response raises RateLimitError."""
        with pytest.raises(RateLimitError) as exc_info:
            raise_for_status(429, ["too many requests"])
        assert "Rate limit exceeded" in exc_info.value.message
        # retry_after should be None since list doesn't have .get()
        assert exc_info.value.details["retry_after"] is None

    def test_429_with_dict_extracts_retry_after(self):
        """429 with dict response extracts retry_after."""
        with pytest.raises(RateLimitError) as exc_info:
            raise_for_status(429, {"retry_after": 30})
        assert exc_info.value.details["retry_after"] == 30

    def test_500_with_list_raises_server_error(self):
        """500 with list response raises ServerError."""
        with pytest.raises(ServerError) as exc_info:
            raise_for_status(500, ["internal error"])
        assert "Server error (HTTP 500)" in exc_info.value.message

    def test_generic_error_with_list(self):
        """Other status codes with list raise generic BrokleError."""
        with pytest.raises(BrokleError) as exc_info:
            raise_for_status(400, ["bad request"])
        assert "HTTP 400: Request failed" in exc_info.value.message
        assert exc_info.value.details["response"] == ["bad request"]

    def test_generic_error_with_string(self):
        """Other status codes with string raise generic BrokleError."""
        with pytest.raises(BrokleError) as exc_info:
            raise_for_status(418, "I'm a teapot")
        assert "HTTP 418: Request failed" in exc_info.value.message

    def test_generic_error_with_dict_extracts_message(self):
        """Other status codes with dict extract error message."""
        with pytest.raises(BrokleError) as exc_info:
            raise_for_status(400, {"error": {"message": "Bad request format"}})
        assert "HTTP 400: Bad request format" in exc_info.value.message


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Edge case tests for error handling."""

    def test_deeply_nested_non_dict(self):
        """Handles deeply nested structures where intermediate is non-dict."""
        body = {"error": ["not", "a", "dict"]}
        result = _safe_extract_error_message(body, default="fallback")
        assert result == "fallback"

    def test_empty_list_response(self):
        """Empty list response is handled."""
        with pytest.raises(AuthenticationError) as exc_info:
            raise_for_status(401, [])
        assert exc_info.value.details["response"] == []

    def test_empty_string_response(self):
        """Empty string response is handled."""
        with pytest.raises(AuthenticationError) as exc_info:
            raise_for_status(401, "")
        assert exc_info.value.details["response"] == ""

    def test_nested_list_in_dict(self):
        """Dict with list value for error key is handled."""
        body = {"error": [{"code": "E001"}, {"code": "E002"}]}
        result = _safe_extract_error_message(body, default="fallback")
        assert result == "fallback"

    def test_message_as_none(self):
        """Dict with message explicitly set to None."""
        body = {"error": {"message": None}}
        result = _safe_extract_error_message(body, default="fallback")
        assert result == "fallback"

    def test_error_as_none(self):
        """Dict with error explicitly set to None."""
        body = {"error": None}
        result = _safe_extract_error_message(body, default="fallback")
        assert result == "fallback"

    def test_tuple_response(self):
        """Tuple response (unlikely but possible) is handled."""
        body = ("error1", "error2")
        result = _safe_extract_error_message(body, default="fallback")
        assert result == "fallback"

    def test_set_response(self):
        """Set response (unlikely but possible) is handled."""
        body = {"error1", "error2"}
        result = _safe_extract_error_message(body, default="fallback")
        assert result == "fallback"


# =============================================================================
# _check_response_status Tests (client.py error handling)
# =============================================================================


def _mock_response(status_code: int, json_body=None, json_error: Exception = None):
    """Create a mock httpx.Response with given status and JSON body."""
    response = MagicMock()
    response.status_code = status_code
    response.headers = {}
    if json_error:
        response.json.side_effect = json_error
    else:
        response.json.return_value = json_body
    return response


class TestCheckResponseStatus:
    """Tests for _check_response_status with non-dict JSON bodies.

    These tests verify that the actual HTTP client error handling path
    (in client.py) correctly handles non-dict JSON responses without crashing.
    """

    def test_success_status_does_not_raise(self):
        """Success status codes don't raise."""
        _check_response_status(_mock_response(200, {"data": "ok"}))
        _check_response_status(_mock_response(201, ["created"]))
        _check_response_status(_mock_response(204, None))

    # -------------------------------------------------------------------------
    # 401/403 Authentication errors with non-dict bodies
    # -------------------------------------------------------------------------

    def test_401_with_list_body_raises_auth_error(self):
        """401 with list body raises AuthenticationError, not AttributeError."""
        response = _mock_response(401, ["unauthorized", "access denied"])
        with pytest.raises(AuthenticationError) as exc_info:
            _check_response_status(response)
        assert "Unknown authentication error" in exc_info.value.message
        assert exc_info.value.details["response"] == ["unauthorized", "access denied"]

    def test_401_with_string_body_raises_auth_error(self):
        """401 with string body raises AuthenticationError."""
        response = _mock_response(401, "Forbidden")
        with pytest.raises(AuthenticationError) as exc_info:
            _check_response_status(response)
        assert "Unknown authentication error" in exc_info.value.message

    def test_403_with_int_body_raises_auth_error(self):
        """403 with integer body raises AuthenticationError."""
        response = _mock_response(403, 0)
        with pytest.raises(AuthenticationError) as exc_info:
            _check_response_status(response)
        assert "Unknown authentication error" in exc_info.value.message

    # -------------------------------------------------------------------------
    # 404 Not Found errors with non-dict bodies
    # -------------------------------------------------------------------------

    def test_404_with_list_body_raises_not_found_error(self):
        """404 with list body raises NotFoundError."""
        response = _mock_response(404, ["not found"])
        with pytest.raises(NotFoundError) as exc_info:
            _check_response_status(response)
        assert "not found" in exc_info.value.message.lower()

    def test_404_with_resource_info(self):
        """404 with resource type and identifier uses specialized message."""
        response = _mock_response(404, None)
        with pytest.raises(NotFoundError) as exc_info:
            _check_response_status(response, resource_type="prompt", identifier="greeting")
        assert "Prompt not found" in exc_info.value.message
        assert "greeting" in exc_info.value.message

    # -------------------------------------------------------------------------
    # 422 Validation errors with non-dict bodies
    # -------------------------------------------------------------------------

    def test_422_with_list_body_raises_validation_error(self):
        """422 with list body raises ValidationError."""
        response = _mock_response(422, [{"field": "name", "error": "required"}])
        with pytest.raises(ValidationError) as exc_info:
            _check_response_status(response)
        assert "Validation" in exc_info.value.message

    # -------------------------------------------------------------------------
    # 429 Rate limit errors with non-dict bodies
    # -------------------------------------------------------------------------

    def test_429_with_list_body_raises_rate_limit_error(self):
        """429 with list body raises RateLimitError."""
        response = _mock_response(429, ["too many requests"])
        with pytest.raises(RateLimitError) as exc_info:
            _check_response_status(response)
        assert "Rate limit exceeded" in exc_info.value.message

    def test_429_with_retry_after_header(self):
        """429 with Retry-After header extracts retry seconds."""
        response = _mock_response(429, {"error": "rate limited"})
        response.headers = {"Retry-After": "60"}
        with pytest.raises(RateLimitError) as exc_info:
            _check_response_status(response)
        assert exc_info.value.details["retry_after"] == 60

    # -------------------------------------------------------------------------
    # 5xx Server errors with non-dict bodies
    # -------------------------------------------------------------------------

    def test_500_with_list_body_raises_server_error(self):
        """500 with list body raises ServerError."""
        response = _mock_response(500, ["internal error", "database timeout"])
        with pytest.raises(ServerError) as exc_info:
            _check_response_status(response)
        assert "Server error (HTTP 500)" in exc_info.value.message
        assert exc_info.value.details["response"] == ["internal error", "database timeout"]

    def test_503_with_string_body_raises_server_error(self):
        """503 with string body raises ServerError."""
        response = _mock_response(503, "Service Unavailable")
        with pytest.raises(ServerError) as exc_info:
            _check_response_status(response)
        assert "Server error (HTTP 503)" in exc_info.value.message

    # -------------------------------------------------------------------------
    # Catch-all (400, 405, 409, etc.) with non-dict bodies - THE BUG FIX
    # -------------------------------------------------------------------------

    def test_400_with_list_body_raises_brokle_error(self):
        """400 with list body raises BrokleError, not AttributeError.

        This is the main regression test for the bug fix.
        """
        response = _mock_response(400, ["bad", "request"])
        with pytest.raises(BrokleError) as exc_info:
            _check_response_status(response)
        # Should NOT be AttributeError
        assert type(exc_info.value) is BrokleError
        assert "HTTP 400: Request failed" in exc_info.value.message
        assert exc_info.value.details["response"] == ["bad", "request"]

    def test_400_with_string_body_raises_brokle_error(self):
        """400 with string body raises BrokleError."""
        response = _mock_response(400, "Bad Request")
        with pytest.raises(BrokleError) as exc_info:
            _check_response_status(response)
        assert type(exc_info.value) is BrokleError
        assert "HTTP 400: Request failed" in exc_info.value.message

    def test_400_with_int_body_raises_brokle_error(self):
        """400 with integer body raises BrokleError."""
        response = _mock_response(400, 123)
        with pytest.raises(BrokleError) as exc_info:
            _check_response_status(response)
        assert type(exc_info.value) is BrokleError
        assert "HTTP 400" in exc_info.value.message

    def test_405_with_list_body_raises_brokle_error(self):
        """405 Method Not Allowed with list body raises BrokleError."""
        response = _mock_response(405, ["GET", "POST"])
        with pytest.raises(BrokleError) as exc_info:
            _check_response_status(response)
        assert type(exc_info.value) is BrokleError
        assert "HTTP 405: Request failed" in exc_info.value.message

    def test_409_with_string_body_raises_brokle_error(self):
        """409 Conflict with string body raises BrokleError."""
        response = _mock_response(409, "Conflict detected")
        with pytest.raises(BrokleError) as exc_info:
            _check_response_status(response)
        assert type(exc_info.value) is BrokleError
        assert "HTTP 409: Request failed" in exc_info.value.message

    def test_400_with_dict_extracts_message(self):
        """400 with proper dict body extracts error message."""
        response = _mock_response(400, {"error": {"message": "Missing parameter 'name'"}})
        with pytest.raises(BrokleError) as exc_info:
            _check_response_status(response)
        assert "HTTP 400: Missing parameter 'name'" in exc_info.value.message

    def test_400_with_none_body_uses_default(self):
        """400 with None body uses default message."""
        response = _mock_response(400, None)
        with pytest.raises(BrokleError) as exc_info:
            _check_response_status(response)
        assert "HTTP 400: Request failed" in exc_info.value.message

    def test_400_with_json_parse_error_uses_none_body(self):
        """400 with JSON parse error handles gracefully."""
        response = _mock_response(400, None, json_error=ValueError("No JSON"))
        with pytest.raises(BrokleError) as exc_info:
            _check_response_status(response)
        assert "HTTP 400: Request failed" in exc_info.value.message
        assert exc_info.value.details["response"] is None

    def test_400_with_bool_body_raises_brokle_error(self):
        """400 with boolean body raises BrokleError."""
        response = _mock_response(400, False)
        with pytest.raises(BrokleError) as exc_info:
            _check_response_status(response)
        assert type(exc_info.value) is BrokleError

    def test_418_teapot_with_list_body(self):
        """418 I'm a teapot with list body raises BrokleError."""
        response = _mock_response(418, ["I'm", "a", "teapot"])
        with pytest.raises(BrokleError) as exc_info:
            _check_response_status(response)
        assert "HTTP 418: Request failed" in exc_info.value.message
