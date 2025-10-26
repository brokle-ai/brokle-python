"""
Smart retry logic with exponential backoff for network operations.
"""

import asyncio
import logging
import random
import time
from typing import Any, Callable, Optional, Tuple, Type, Union

import httpx

from ..exceptions import RateLimitError

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        delay = min(delay, self.max_delay)

        if self.jitter:
            # Add ±25% jitter to avoid thundering herd
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0.1, delay)  # Minimum 100ms delay


def is_retryable_error(error: Exception) -> bool:
    """Check if error is retryable."""

    # Network errors are generally retryable
    if isinstance(error, (httpx.ConnectError, httpx.TimeoutException)):
        return True

    # HTTP errors
    if isinstance(error, httpx.HTTPStatusError):
        status_code = error.response.status_code
        # Retry on server errors (5xx) and some client errors
        if 500 <= status_code < 600:
            return True
        if status_code == 429:  # Rate limiting
            return True
        if status_code == 408:  # Request timeout
            return True
        if status_code == 502:  # Bad gateway
            return True
        if status_code == 503:  # Service unavailable
            return True
        if status_code == 504:  # Gateway timeout
            return True

    # Rate limit errors
    if isinstance(error, RateLimitError):
        return True

    return False


def extract_retry_after(error: Exception) -> Optional[float]:
    """Extract Retry-After header value from error."""
    if hasattr(error, "response") and error.response:
        retry_after = error.response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return float(retry_after)
            except (ValueError, TypeError):
                pass
    return None


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
):
    """Decorator for adding retry logic with exponential backoff."""

    retry_config = RetryConfig(
        max_retries, base_delay, max_delay, exponential_base, jitter
    )

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            last_error = None

            for attempt in range(retry_config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e

                    # Don't retry on final attempt
                    if attempt == retry_config.max_retries:
                        break

                    # Check if error is retryable
                    if not is_retryable_error(e):
                        break

                    # Calculate delay
                    retry_after = extract_retry_after(e)
                    if retry_after:
                        delay = min(retry_after, retry_config.max_delay)
                    else:
                        delay = retry_config.calculate_delay(attempt + 1)

                    logger.debug(
                        f"Attempt {attempt + 1}/{retry_config.max_retries} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    time.sleep(delay)

            # Re-raise the last error
            raise last_error

        return wrapper

    return decorator


def async_retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
):
    """Async decorator for adding retry logic with exponential backoff."""

    retry_config = RetryConfig(
        max_retries, base_delay, max_delay, exponential_base, jitter
    )

    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs) -> Any:
            last_error = None

            for attempt in range(retry_config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e

                    # Don't retry on final attempt
                    if attempt == retry_config.max_retries:
                        break

                    # Check if error is retryable
                    if not is_retryable_error(e):
                        break

                    # Calculate delay
                    retry_after = extract_retry_after(e)
                    if retry_after:
                        delay = min(retry_after, retry_config.max_delay)
                    else:
                        delay = retry_config.calculate_delay(attempt + 1)

                    logger.debug(
                        f"Attempt {attempt + 1}/{retry_config.max_retries} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    await asyncio.sleep(delay)

            # Re-raise the last error
            raise last_error

        return wrapper

    return decorator


class RetryableHTTPClient:
    """HTTP client with built-in retry logic."""

    def __init__(
        self, client: httpx.Client, retry_config: Optional[RetryConfig] = None
    ):
        self.client = client
        self.retry_config = retry_config or RetryConfig()

    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Make HTTP request with retry logic."""

        @retry_with_backoff(
            max_retries=self.retry_config.max_retries,
            base_delay=self.retry_config.base_delay,
            max_delay=self.retry_config.max_delay,
            exponential_base=self.retry_config.exponential_base,
            jitter=self.retry_config.jitter,
        )
        def _request():
            return self.client.request(method, url, **kwargs)

        return _request()


class AsyncRetryableHTTPClient:
    """Async HTTP client with built-in retry logic."""

    def __init__(
        self, client: httpx.AsyncClient, retry_config: Optional[RetryConfig] = None
    ):
        self.client = client
        self.retry_config = retry_config or RetryConfig()

    async def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Make async HTTP request with retry logic."""

        @async_retry_with_backoff(
            max_retries=self.retry_config.max_retries,
            base_delay=self.retry_config.base_delay,
            max_delay=self.retry_config.max_delay,
            exponential_base=self.retry_config.exponential_base,
            jitter=self.retry_config.jitter,
        )
        async def _request():
            return await self.client.request(method, url, **kwargs)

        return await _request()
