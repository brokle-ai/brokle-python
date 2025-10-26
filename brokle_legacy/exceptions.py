"""Exception classes for Brokle SDK."""

from typing import Any, Dict, Optional


class BrokleError(Exception):
    """Base exception for all Brokle errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "brokle_error",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details

    def __eq__(self, other):
        if not isinstance(other, BrokleError):
            return False
        return (
            self.message == other.message
            and self.status_code == other.status_code
            and self.error_code == other.error_code
            and self.details == other.details
        )

    def __hash__(self):
        return hash((self.message, self.status_code, self.error_code))


class AuthenticationError(BrokleError):
    """Raised when authentication fails."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            status_code=kwargs.get("status_code", 401),
            error_code=kwargs.get("error_code", "authentication_error"),
            details=kwargs.get("details"),
        )


class RateLimitError(BrokleError):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        super().__init__(
            message,
            status_code=kwargs.get("status_code", 429),
            error_code=kwargs.get("error_code", "rate_limit_error"),
            details=kwargs.get("details"),
        )
        self.retry_after = retry_after


class ConfigurationError(BrokleError):
    """Raised when configuration is invalid."""

    pass


class APIError(BrokleError):
    """Raised when API request fails."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            status_code=status_code or kwargs.get("status_code", 500),
            error_code=kwargs.get("error_code", "api_error"),
            details=details or kwargs.get("details"),
        )


class NetworkError(BrokleError):
    """Raised when network request fails."""

    pass


class ValidationError(BrokleError):
    """Raised when data validation fails."""

    def __init__(
        self, message: str, field_errors: Optional[Dict[str, Any]] = None, **kwargs
    ):
        super().__init__(
            message,
            status_code=kwargs.get("status_code", 400),
            error_code=kwargs.get("error_code", "validation_error"),
            details=kwargs.get("details"),
        )
        self.field_errors = field_errors or {}


class TimeoutError(BrokleError):
    """Raised when operation times out."""

    pass


class UnsupportedOperationError(BrokleError):
    """Raised when operation is not supported."""

    pass


class QuotaExceededError(BrokleError):
    """Raised when quota is exceeded."""

    def __init__(
        self, message: str, quota_info: Optional[Dict[str, Any]] = None, **kwargs
    ):
        super().__init__(
            message,
            status_code=kwargs.get("status_code", 429),
            error_code=kwargs.get("error_code", "quota_exceeded"),
            details=kwargs.get("details"),
        )
        self.quota_info = quota_info


class ProviderError(BrokleError):
    """Raised when AI provider fails."""

    def __init__(
        self, message: str, provider_details: Optional[Dict[str, Any]] = None, **kwargs
    ):
        super().__init__(
            message,
            status_code=kwargs.get("status_code", 502),
            error_code=kwargs.get("error_code", "provider_error"),
            details=kwargs.get("details"),
        )
        self.provider_details = provider_details


class CacheError(BrokleError):
    """Raised when cache operation fails."""

    def __init__(
        self, message: str, cache_details: Optional[Dict[str, Any]] = None, **kwargs
    ):
        super().__init__(
            message,
            status_code=kwargs.get("status_code", 500),
            error_code=kwargs.get("error_code", "cache_error"),
            details=kwargs.get("details"),
        )
        self.cache_details = cache_details


class EvaluationError(BrokleError):
    """Raised when evaluation fails."""

    def __init__(
        self,
        message: str,
        evaluation_details: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            status_code=kwargs.get("status_code", 500),
            error_code=kwargs.get("error_code", "evaluation_error"),
            details=kwargs.get("details"),
        )
        self.evaluation_details = evaluation_details
