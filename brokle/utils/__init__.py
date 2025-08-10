"""
Utility modules for Brokle SDK.
"""

from .error_handler import (
    BrokleError,
    AuthenticationError,
    AuthorizationError,
    ValidationError,
    NetworkError,
    RateLimitError,
    QuotaExceededError,
    ProviderError,
    TimeoutError,
    ConfigurationError,
    handle_api_error,
    handle_http_error,
)
from .response_parser import (
    parse_api_response,
    parse_error_response,
    extract_response_metadata,
)

__all__ = [
    # Error handling
    "BrokleError",
    "AuthenticationError",
    "AuthorizationError",
    "ValidationError",
    "NetworkError", 
    "RateLimitError",
    "QuotaExceededError",
    "ProviderError",
    "TimeoutError",
    "ConfigurationError",
    "handle_api_error",
    "handle_http_error",
    
    # Response parsing
    "parse_api_response",
    "parse_error_response",
    "extract_response_metadata",
]