"""
Query Managers (THE WEDGE)

Provides both synchronous and asynchronous query managers for querying
production spans. This is Brokle's key differentiator - evaluating existing
production telemetry without re-instrumenting applications.

Sync Usage:
    >>> from brokle import Brokle
    >>>
    >>> client = Brokle(api_key="bk_...")
    >>>
    >>> result = client.query.query(
    ...     filter="service.name=chatbot AND gen_ai.system=openai",
    ...     start_time=datetime.now() - timedelta(days=7),
    ... )
    >>> for span in result.spans:
    ...     print(span.name, span.model)

Async Usage:
    >>> async with AsyncBrokle(api_key="bk_...") as client:
    ...     async for span in client.query.query_iter(
    ...         filter="gen_ai.system=openai",
    ...     ):
    ...         print(span.input, span.output)
"""

from datetime import datetime
from typing import Any, AsyncIterator, Dict, Iterator, Optional

from .._http import AsyncHTTPClient, SyncHTTPClient
from ..config import BrokleConfig
from .exceptions import InvalidFilterError, QueryAPIError
from .types import QueriedSpan, QueryResult, ValidationResult


class _BaseQueryManagerMixin:
    """
    Shared functionality for both sync and async query managers.
    """

    _config: BrokleConfig

    def _log(self, message: str, *args: Any) -> None:
        """Log debug messages."""
        if self._config.debug:
            print(f"[Brokle Query] {message}", *args)

    def _build_query_body(
        self,
        filter: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
        page: int = 1,
    ) -> Dict[str, Any]:
        """Build request body for query API."""
        body: Dict[str, Any] = {
            "filter": filter,
            "limit": limit,
            "page": page,
        }
        if start_time:
            body["start_time"] = (
                start_time.isoformat() + "Z"
                if start_time.tzinfo is None
                else start_time.isoformat()
            )
        if end_time:
            body["end_time"] = (
                end_time.isoformat() + "Z"
                if end_time.tzinfo is None
                else end_time.isoformat()
            )
        return body


class QueryManager(_BaseQueryManagerMixin):
    """
    Sync query manager for Brokle.

    All methods are synchronous. Uses SyncHTTPClient (httpx.Client) internally.

    Example:
        >>> from brokle import Brokle
        >>> from datetime import datetime, timedelta
        >>>
        >>> client = Brokle(api_key="bk_...")
        >>>
        >>> # Query spans from the last 7 days
        >>> result = client.query.query(
        ...     filter="service.name=chatbot AND gen_ai.system=openai",
        ...     start_time=datetime.now() - timedelta(days=7),
        ... )
        >>>
        >>> # Access spans and their convenience fields
        >>> for span in result.spans:
        ...     print(f"{span.model}: {span.input[:50]}...")
    """

    def __init__(
        self,
        http_client: SyncHTTPClient,
        config: BrokleConfig,
    ):
        """
        Initialize sync query manager.

        Args:
            http_client: Sync HTTP client
            config: Brokle configuration
        """
        self._http = http_client
        self._config = config

    def query(
        self,
        filter: str,
        *,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
        page: int = 1,
    ) -> QueryResult:
        """
        Query spans using filter expression.

        Args:
            filter: Filter expression (e.g., "service.name=chatbot AND gen_ai.system=openai")
            start_time: Start of time range (optional)
            end_time: End of time range (optional)
            limit: Maximum number of spans to return (default: 1000)
            page: Page number for pagination, 1-indexed (default: 1)

        Returns:
            QueryResult with spans and pagination metadata

        Raises:
            InvalidFilterError: If filter syntax is invalid
            QueryAPIError: If API request fails

        Example:
            >>> result = client.query.query(
            ...     filter="gen_ai.response.model=gpt-4",
            ...     limit=100,
            ... )
            >>> print(f"Found {result.total} spans")
        """
        self._log(f"Querying spans: filter={filter}, limit={limit}, page={page}")

        body = self._build_query_body(filter, start_time, end_time, limit, page)

        try:
            raw_response = self._http.post("/v1/spans/query", json=body)

            # Check for validation errors
            if not raw_response.get("success"):
                error = raw_response.get("error", {})
                error_msg = error.get("message", "Unknown error")
                code = error.get("code")

                if code == "INVALID_FILTER" or "filter" in error_msg.lower():
                    raise InvalidFilterError(filter, error_msg)

                raise QueryAPIError(
                    message=error_msg,
                    code=code,
                )

            data = raw_response["data"]
            return QueryResult.from_dict(data)

        except (InvalidFilterError, QueryAPIError):
            raise
        except Exception as e:
            raise QueryAPIError(f"Query failed: {e}")

    def query_iter(
        self,
        filter: str,
        *,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        batch_size: int = 100,
    ) -> Iterator[QueriedSpan]:
        """
        Iterate spans with auto-pagination.

        Yields spans one at a time, automatically fetching next pages.
        More memory-efficient than query() for large result sets.

        Args:
            filter: Filter expression
            start_time: Start of time range (optional)
            end_time: End of time range (optional)
            batch_size: Number of spans per API request (default: 100)

        Yields:
            QueriedSpan objects

        Raises:
            InvalidFilterError: If filter syntax is invalid
            QueryAPIError: If API request fails

        Example:
            >>> for span in client.query.query_iter("gen_ai.system=openai"):
            ...     process_span(span)
        """
        page = 1
        while True:
            result = self.query(
                filter,
                start_time=start_time,
                end_time=end_time,
                limit=batch_size,
                page=page,
            )
            for span in result.spans:
                yield span

            if not result.has_more:
                break

            page = result.next_page if result.next_page is not None else (page + 1)

    def validate(self, filter: str) -> ValidationResult:
        """
        Validate filter syntax.

        Check if a filter expression is syntactically valid without executing a query.

        Args:
            filter: Filter expression to validate

        Returns:
            ValidationResult with valid flag and message/error

        Raises:
            QueryAPIError: If API request fails

        Example:
            >>> result = client.query.validate("service.name=chatbot")
            >>> if result.valid:
            ...     print("Filter is valid!")
        """
        self._log(f"Validating filter: {filter}")

        try:
            raw_response = self._http.post(
                "/v1/spans/query/validate", json={"filter": filter}
            )

            if not raw_response.get("success"):
                error = raw_response.get("error", {})
                error_msg = error.get("message", "Unknown error")
                raise QueryAPIError(f"Validation request failed: {error_msg}")

            data = raw_response["data"]
            return ValidationResult.from_dict(data)

        except QueryAPIError:
            raise
        except Exception as e:
            raise QueryAPIError(f"Validation failed: {e}")

    def validate_or_raise(self, filter: str) -> None:
        """
        Validate filter and raise if invalid.

        Convenience method that raises InvalidFilterError if the filter is invalid.

        Args:
            filter: Filter expression to validate

        Raises:
            InvalidFilterError: If filter is invalid
            QueryAPIError: If API request fails

        Example:
            >>> client.query.validate_or_raise("service.name=chatbot")  # Returns None if valid
            >>> client.query.validate_or_raise("invalid syntax")  # Raises InvalidFilterError
        """
        result = self.validate(filter)
        if not result.valid:
            raise InvalidFilterError(filter, result.error)


class AsyncQueryManager(_BaseQueryManagerMixin):
    """
    Async query manager for Brokle.

    All methods are asynchronous. Uses AsyncHTTPClient (httpx.AsyncClient) internally.

    Example:
        >>> from brokle import AsyncBrokle
        >>> from datetime import datetime, timedelta
        >>>
        >>> async with AsyncBrokle(api_key="bk_...") as client:
        ...     result = await client.query.query(
        ...         filter="service.name=chatbot",
        ...         start_time=datetime.now() - timedelta(days=7),
        ...     )
        ...     for span in result.spans:
        ...         print(span.model)
    """

    def __init__(
        self,
        http_client: AsyncHTTPClient,
        config: BrokleConfig,
    ):
        """
        Initialize async query manager.

        Args:
            http_client: Async HTTP client
            config: Brokle configuration
        """
        self._http = http_client
        self._config = config

    async def query(
        self,
        filter: str,
        *,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
        page: int = 1,
    ) -> QueryResult:
        """
        Query spans using filter expression (async).

        Args:
            filter: Filter expression (e.g., "service.name=chatbot AND gen_ai.system=openai")
            start_time: Start of time range (optional)
            end_time: End of time range (optional)
            limit: Maximum number of spans to return (default: 1000)
            page: Page number for pagination, 1-indexed (default: 1)

        Returns:
            QueryResult with spans and pagination metadata

        Raises:
            InvalidFilterError: If filter syntax is invalid
            QueryAPIError: If API request fails

        Example:
            >>> result = await client.query.query(
            ...     filter="gen_ai.response.model=gpt-4",
            ...     limit=100,
            ... )
            >>> print(f"Found {result.total} spans")
        """
        self._log(f"Querying spans: filter={filter}, limit={limit}, page={page}")

        body = self._build_query_body(filter, start_time, end_time, limit, page)

        try:
            raw_response = await self._http.post("/v1/spans/query", json=body)

            # Check for validation errors
            if not raw_response.get("success"):
                error = raw_response.get("error", {})
                error_msg = error.get("message", "Unknown error")
                code = error.get("code")

                if code == "INVALID_FILTER" or "filter" in error_msg.lower():
                    raise InvalidFilterError(filter, error_msg)

                raise QueryAPIError(
                    message=error_msg,
                    code=code,
                )

            data = raw_response["data"]
            return QueryResult.from_dict(data)

        except (InvalidFilterError, QueryAPIError):
            raise
        except Exception as e:
            raise QueryAPIError(f"Query failed: {e}")

    async def query_iter(
        self,
        filter: str,
        *,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        batch_size: int = 100,
    ) -> AsyncIterator[QueriedSpan]:
        """
        Iterate spans with auto-pagination (async).

        Yields spans one at a time, automatically fetching next pages.
        More memory-efficient than query() for large result sets.

        Args:
            filter: Filter expression
            start_time: Start of time range (optional)
            end_time: End of time range (optional)
            batch_size: Number of spans per API request (default: 100)

        Yields:
            QueriedSpan objects

        Raises:
            InvalidFilterError: If filter syntax is invalid
            QueryAPIError: If API request fails

        Example:
            >>> async for span in client.query.query_iter("gen_ai.system=openai"):
            ...     await process_span(span)
        """
        page = 1
        while True:
            result = await self.query(
                filter,
                start_time=start_time,
                end_time=end_time,
                limit=batch_size,
                page=page,
            )
            for span in result.spans:
                yield span

            if not result.has_more:
                break

            page = result.next_page if result.next_page is not None else (page + 1)

    async def validate(self, filter: str) -> ValidationResult:
        """
        Validate filter syntax (async).

        Check if a filter expression is syntactically valid without executing a query.

        Args:
            filter: Filter expression to validate

        Returns:
            ValidationResult with valid flag and message/error

        Raises:
            QueryAPIError: If API request fails

        Example:
            >>> result = await client.query.validate("service.name=chatbot")
            >>> if result.valid:
            ...     print("Filter is valid!")
        """
        self._log(f"Validating filter: {filter}")

        try:
            raw_response = await self._http.post(
                "/v1/spans/query/validate", json={"filter": filter}
            )

            if not raw_response.get("success"):
                error = raw_response.get("error", {})
                error_msg = error.get("message", "Unknown error")
                raise QueryAPIError(f"Validation request failed: {error_msg}")

            data = raw_response["data"]
            return ValidationResult.from_dict(data)

        except QueryAPIError:
            raise
        except Exception as e:
            raise QueryAPIError(f"Validation failed: {e}")

    async def validate_or_raise(self, filter: str) -> None:
        """
        Validate filter and raise if invalid (async).

        Convenience method that raises InvalidFilterError if the filter is invalid.

        Args:
            filter: Filter expression to validate

        Raises:
            InvalidFilterError: If filter is invalid
            QueryAPIError: If API request fails

        Example:
            >>> await client.query.validate_or_raise("service.name=chatbot")  # Returns None if valid
            >>> await client.query.validate_or_raise("invalid syntax")  # Raises InvalidFilterError
        """
        result = await self.validate(filter)
        if not result.valid:
            raise InvalidFilterError(filter, result.error)
