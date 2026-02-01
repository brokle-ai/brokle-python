"""
Brokle SDK Error Classes

Enhanced error classes with actionable guidance for common issues.
Follows Langfuse naming pattern (no prefix on specific errors).
"""

from typing import Any, Optional


def _safe_extract_error_message(
    response_body: Any,
    default: str,
    *,
    nested_path: tuple[str, ...] = ("error", "message"),
) -> str:
    """
    Safely extract error message from response body with type validation.

    Handles non-dict JSON responses (lists, strings, numbers, null, bool) by
    returning the default message.
    """
    if not isinstance(response_body, dict):
        return default

    current: Any = response_body
    for key in nested_path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default

    return current if isinstance(current, str) else default


def _safe_get_value(
    response_body: Any,
    key: str,
    default: Any = None,
) -> Any:
    """Safely get a value from response body with type validation."""
    if not isinstance(response_body, dict):
        return default
    return response_body.get(key, default)


class BrokleError(Exception):
    """
    Base error for all Brokle SDK errors.

    Includes actionable guidance to help users resolve issues.
    """

    def __init__(
        self,
        message: str,
        *,
        hint: Optional[str] = None,
        original_error: Optional[Exception] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize BrokleError.

        Args:
            message: Main error message.
            hint: Optional actionable guidance.
            original_error: Original exception that caused this error.
            details: Additional error details.
        """
        self.message = message
        self.hint = hint
        self.original_error = original_error
        self.details = details or {}

        # Build full message with hint
        full_message = message
        if hint:
            full_message = f"{message}\n\nTo fix:\n{hint}"

        super().__init__(full_message)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r})"


class AuthenticationError(BrokleError):
    """
    Authentication failed.

    Raised when API key is invalid, missing, or expired.
    """

    @classmethod
    def from_response(
        cls,
        status_code: int,
        response_body: Any = None,
        api_key_prefix: Optional[str] = None,
    ) -> "AuthenticationError":
        """
        Create authentication error from HTTP response.

        Args:
            status_code: HTTP status code.
            response_body: Response JSON body (may be non-dict for some servers/proxies).
            api_key_prefix: First few characters of the API key for debugging.

        Returns:
            AuthenticationError with actionable guidance.
        """
        error_msg = _safe_extract_error_message(
            response_body,
            default="Unknown authentication error",
        )

        key_info = f" (key prefix: {api_key_prefix}...)" if api_key_prefix else ""

        hint = """
1. Check your API key is set:
   export BROKLE_API_KEY=bk_your_secret_key

2. Verify the key is valid (should start with 'bk_'):
   python -c "import os; print(os.getenv('BROKLE_API_KEY', 'NOT SET')[:10])"

3. Test connection:
   python -c "from brokle import get_client; print('OK' if get_client().auth_check() else 'FAILED')"

4. If using a custom base URL, verify it's correct:
   export BROKLE_BASE_URL=https://your-brokle-server.com
""".strip()

        return cls(
            f"Authentication failed (HTTP {status_code}){key_info}: {error_msg}",
            hint=hint,
            details={"status_code": status_code, "response": response_body},
        )


class ConnectionError(BrokleError):
    """
    Connection to Brokle server failed.

    Raised when the server is unreachable or connection times out.

    Note: Named ConnectionError (not BrokleConnectionError) following Langfuse pattern.
    Import explicitly to avoid conflict with builtins.ConnectionError.
    """

    @classmethod
    def from_exception(
        cls,
        original: Exception,
        base_url: Optional[str] = None,
    ) -> "ConnectionError":
        """
        Create connection error from underlying exception.

        Args:
            original: Original connection exception.
            base_url: The base URL that failed to connect.

        Returns:
            ConnectionError with actionable guidance.
        """
        url_info = f" ({base_url})" if base_url else ""

        hint = f"""
1. Check if the Brokle server is running{url_info}:
   curl -s {base_url or 'http://localhost:8080'}/health || echo "Server not reachable"

2. Verify your base URL is correct:
   export BROKLE_BASE_URL=http://localhost:8080  # Local development
   export BROKLE_BASE_URL=https://api.brokle.com  # Production

3. Check network connectivity:
   ping {(base_url or 'localhost').replace('http://', '').replace('https://', '').split(':')[0]}

4. If using Docker, ensure the container is running:
   docker ps | grep brokle
""".strip()

        return cls(
            f"Failed to connect to Brokle server{url_info}: {original}",
            hint=hint,
            original_error=original,
            details={"base_url": base_url},
        )


class ValidationError(BrokleError):
    """
    Request validation failed.

    Raised when the request contains invalid data.
    """

    @classmethod
    def from_response(
        cls,
        response_body: Any,
        field: Optional[str] = None,
    ) -> "ValidationError":
        """
        Create validation error from API response.

        Args:
            response_body: Response JSON body with validation details (may be non-dict).
            field: Specific field that failed validation.

        Returns:
            ValidationError with guidance.
        """
        error_msg = _safe_extract_error_message(
            response_body,
            default="Validation failed",
        )
        field_info = f" (field: {field})" if field else ""

        hint = """
1. Check required fields are provided
2. Verify data types match expected format
3. Check string lengths and numeric ranges
4. Review API documentation for valid values
""".strip()

        return cls(
            f"Validation error{field_info}: {error_msg}",
            hint=hint,
            details={"response": response_body, "field": field},
        )


class RateLimitError(BrokleError):
    """
    Rate limit exceeded.

    Raised when too many requests are sent in a short period.
    """

    @classmethod
    def from_response(
        cls,
        response_body: Any = None,
        retry_after: Optional[int] = None,
    ) -> "RateLimitError":
        """
        Create rate limit error from response.

        Args:
            response_body: Response JSON body (may be non-dict for some servers/proxies).
            retry_after: Seconds to wait before retrying.

        Returns:
            RateLimitError with retry guidance.
        """
        wait_info = f" (retry after {retry_after}s)" if retry_after else ""

        hint = f"""
1. Wait before retrying{wait_info or ' (check Retry-After header)'}
2. Reduce request frequency
3. Implement exponential backoff
4. Consider batching operations
5. Contact support for higher limits if needed
""".strip()

        return cls(
            f"Rate limit exceeded{wait_info}",
            hint=hint,
            details={"response": response_body, "retry_after": retry_after},
        )


class NotFoundError(BrokleError):
    """
    Resource not found.

    Raised when the requested resource doesn't exist.
    """

    @classmethod
    def for_resource(
        cls,
        resource_type: str,
        identifier: str,
    ) -> "NotFoundError":
        """
        Create not found error for a specific resource.

        Args:
            resource_type: Type of resource (e.g., "prompt", "dataset").
            identifier: Resource identifier.

        Returns:
            NotFoundError with guidance.
        """
        hint = f"""
1. Verify the {resource_type} exists:
   - Check the dashboard for existing {resource_type}s
   - Ensure correct project context

2. Check for typos in the identifier: '{identifier}'

3. If creating new, ensure the {resource_type} was created successfully
""".strip()

        return cls(
            f"{resource_type.capitalize()} not found: '{identifier}'",
            hint=hint,
            details={"resource_type": resource_type, "identifier": identifier},
        )


class ServerError(BrokleError):
    """
    Server-side error.

    Raised when the Brokle server encounters an internal error.
    """

    @classmethod
    def from_response(
        cls,
        status_code: int,
        response_body: Any = None,
    ) -> "ServerError":
        """
        Create server error from response.

        Args:
            status_code: HTTP status code (5xx).
            response_body: Response JSON body (may be non-dict for some servers/proxies).

        Returns:
            ServerError with guidance.
        """
        hint = """
1. Retry the request after a brief delay
2. If persistent, check Brokle server status
3. Check server logs for more details
4. Contact support if the issue continues
""".strip()

        return cls(
            f"Server error (HTTP {status_code})",
            hint=hint,
            details={"status_code": status_code, "response": response_body},
        )


# Convenience function for raising appropriate error based on status code
def raise_for_status(
    status_code: int,
    response_body: Any = None,
    *,
    api_key_prefix: Optional[str] = None,
    base_url: Optional[str] = None,
    resource_type: Optional[str] = None,
    identifier: Optional[str] = None,
) -> None:
    """
    Raise appropriate error based on HTTP status code.

    Args:
        status_code: HTTP status code.
        response_body: Response JSON body (may be non-dict for some servers/proxies).
        api_key_prefix: API key prefix for auth error messages.
        base_url: Base URL for connection error messages.
        resource_type: Resource type for not found errors.
        identifier: Resource identifier for not found errors.

    Raises:
        BrokleError: Appropriate error subclass based on status code.
    """
    if status_code >= 200 and status_code < 300:
        return  # Success, no error

    if status_code == 401 or status_code == 403:
        raise AuthenticationError.from_response(
            status_code, response_body, api_key_prefix
        )

    if status_code == 404:
        if resource_type and identifier:
            raise NotFoundError.for_resource(resource_type, identifier)
        raise NotFoundError(
            "Resource not found",
            hint="Check the resource identifier and project context.",
            details={"status_code": status_code, "response": response_body},
        )

    if status_code == 422:
        raise ValidationError.from_response(response_body or {})

    if status_code == 429:
        retry_after = _safe_get_value(response_body, "retry_after")
        raise RateLimitError.from_response(response_body, retry_after)

    if status_code >= 500:
        raise ServerError.from_response(status_code, response_body)

    # Generic error for other status codes
    error_msg = _safe_extract_error_message(
        response_body,
        default="Request failed",
    )

    raise BrokleError(
        f"HTTP {status_code}: {error_msg}",
        details={"status_code": status_code, "response": response_body},
    )
