"""
Error handling utilities for Brokle SDK.
"""

import logging
from typing import Any, Dict, Optional, Type

import httpx

logger = logging.getLogger(__name__)


class BrokleError(Exception):
    """Base exception for Brokle SDK errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        status_code: Optional[int] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.status_code = status_code
        self.request_id = request_id


class AuthenticationError(BrokleError):
    """Authentication related errors."""
    pass


class AuthorizationError(BrokleError):
    """Authorization related errors."""
    pass


class ValidationError(BrokleError):
    """Validation related errors."""
    pass


class NetworkError(BrokleError):
    """Network related errors."""
    pass


class RateLimitError(BrokleError):
    """Rate limiting errors."""
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        limit_type: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after
        self.limit_type = limit_type


class QuotaExceededError(BrokleError):
    """Quota exceeded errors."""
    
    def __init__(
        self,
        message: str,
        quota_type: Optional[str] = None,
        current_usage: Optional[int] = None,
        quota_limit: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.quota_type = quota_type
        self.current_usage = current_usage
        self.quota_limit = quota_limit


class ProviderError(BrokleError):
    """AI provider related errors."""
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        provider_error_code: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.provider = provider
        self.provider_error_code = provider_error_code


class TimeoutError(BrokleError):
    """Timeout related errors."""
    pass


class ConfigurationError(BrokleError):
    """Configuration related errors."""
    pass


def handle_api_error(response_data: Dict[str, Any], status_code: int) -> BrokleError:
    """Handle API error responses and convert to appropriate exception."""
    error_data = response_data.get('error', {})
    message = error_data.get('message', 'Unknown error')
    error_code = error_data.get('code', 'unknown_error')
    details = error_data.get('details', {})
    request_id = response_data.get('meta', {}).get('request_id')
    
    # Determine error type based on status code and error code
    if status_code == 401:
        return AuthenticationError(
            message=message,
            error_code=error_code,
            details=details,
            status_code=status_code,
            request_id=request_id
        )
    elif status_code == 403:
        return AuthorizationError(
            message=message,
            error_code=error_code,
            details=details,
            status_code=status_code,
            request_id=request_id
        )
    elif status_code == 400:
        return ValidationError(
            message=message,
            error_code=error_code,
            details=details,
            status_code=status_code,
            request_id=request_id
        )
    elif status_code == 429:
        retry_after = details.get('retry_after')
        limit_type = details.get('limit_type')
        return RateLimitError(
            message=message,
            error_code=error_code,
            details=details,
            status_code=status_code,
            request_id=request_id,
            retry_after=retry_after,
            limit_type=limit_type
        )
    elif error_code == 'quota_exceeded':
        return QuotaExceededError(
            message=message,
            error_code=error_code,
            details=details,
            status_code=status_code,
            request_id=request_id,
            quota_type=details.get('quota_type'),
            current_usage=details.get('current_usage'),
            quota_limit=details.get('quota_limit')
        )
    elif error_code.startswith('provider_'):
        return ProviderError(
            message=message,
            error_code=error_code,
            details=details,
            status_code=status_code,
            request_id=request_id,
            provider=details.get('provider'),
            provider_error_code=details.get('provider_error_code')
        )
    elif status_code == 408 or 'timeout' in error_code:
        return TimeoutError(
            message=message,
            error_code=error_code,
            details=details,
            status_code=status_code,
            request_id=request_id
        )
    else:
        return BrokleError(
            message=message,
            error_code=error_code,
            details=details,
            status_code=status_code,
            request_id=request_id
        )


def handle_http_error(error: httpx.HTTPError) -> BrokleError:
    """Handle HTTP errors and convert to appropriate exception."""
    if isinstance(error, httpx.TimeoutException):
        return TimeoutError(
            message=f"Request timeout: {error}",
            error_code="timeout_error"
        )
    elif isinstance(error, httpx.NetworkError):
        return NetworkError(
            message=f"Network error: {error}",
            error_code="network_error"
        )
    elif isinstance(error, httpx.HTTPStatusError):
        # Try to parse error response
        try:
            response_data = error.response.json()
            return handle_api_error(response_data, error.response.status_code)
        except Exception:
            # Fallback to generic error
            return BrokleError(
                message=f"HTTP error {error.response.status_code}: {error}",
                error_code="http_error",
                status_code=error.response.status_code
            )
    else:
        return BrokleError(
            message=f"HTTP error: {error}",
            error_code="http_error"
        )


def get_error_class(error_code: str) -> Type[BrokleError]:
    """Get appropriate error class for error code."""
    error_mapping = {
        'authentication_error': AuthenticationError,
        'authorization_error': AuthorizationError,
        'validation_error': ValidationError,
        'network_error': NetworkError,
        'rate_limit_error': RateLimitError,
        'quota_exceeded': QuotaExceededError,
        'provider_error': ProviderError,
        'timeout_error': TimeoutError,
        'configuration_error': ConfigurationError,
    }
    
    return error_mapping.get(error_code, BrokleError)