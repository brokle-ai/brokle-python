"""
Exceptions for the query module.

Provides query-specific error types:
- QueryError: Base error for query operations
- InvalidFilterError: Filter syntax is invalid
- QueryAPIError: API request failed
"""

from typing import Optional


class QueryError(Exception):
    """
    Base error for query operations.

    All query-specific errors inherit from this class.
    """

    pass


class InvalidFilterError(QueryError):
    """
    Filter syntax is invalid.

    Raised when a filter expression fails validation.

    Attributes:
        filter: The invalid filter expression
    """

    def __init__(self, filter_expr: str, message: Optional[str] = None):
        self.filter = filter_expr
        msg = f"Invalid filter '{filter_expr}'"
        if message:
            msg = f"{msg}: {message}"
        super().__init__(msg)


class QueryAPIError(QueryError):
    """
    API request failed.

    Raised when the query API returns an error.

    Attributes:
        status_code: HTTP status code (if available)
        code: Error code from API (if available)
    """

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        code: Optional[str] = None,
    ):
        self.status_code = status_code
        self.code = code
        super().__init__(message)
