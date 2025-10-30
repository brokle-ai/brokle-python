"""
Async utilities for Brokle SDK.

Simple async utilities extracted and cleaned from the old integration framework.
"""

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def safe_async_call(
    coro: Awaitable[T], timeout: Optional[float] = None, default: Optional[T] = None
) -> Optional[T]:
    """
    Safely execute an async call with timeout and error handling.

    Args:
        coro: Coroutine to execute
        timeout: Optional timeout in seconds
        default: Default value to return on error

    Returns:
        Result or default value
    """
    try:
        if timeout:
            return await asyncio.wait_for(coro, timeout=timeout)
        else:
            return await coro
    except asyncio.TimeoutError:
        logger.warning(f"Async call timed out after {timeout}s")
        return default
    except Exception as e:
        logger.debug(f"Async call failed: {e}")
        return default


def is_async_callable(obj: Any) -> bool:
    """
    Check if an object is an async callable.

    Args:
        obj: Object to check

    Returns:
        True if object is async callable
    """
    return asyncio.iscoroutinefunction(obj) or (
        hasattr(obj, "__call__") and asyncio.iscoroutinefunction(obj.__call__)
    )


async def run_async_in_sync(coro: Awaitable[T]) -> T:
    """
    Run async code in a sync context safely.

    Args:
        coro: Coroutine to run

    Returns:
        Result of coroutine
    """
    try:
        # Check if we're already in an event loop
        loop = asyncio.get_running_loop()
        # If we are, we can't use asyncio.run()
        return await coro
    except RuntimeError:
        # No event loop running, safe to use asyncio.run()
        return asyncio.run(coro)


def sync_wrapper(async_func: Callable[..., Awaitable[T]]) -> Callable[..., T]:
    """
    Create a sync wrapper for an async function.

    Args:
        async_func: Async function to wrap

    Returns:
        Sync version of the function
    """

    def wrapper(*args, **kwargs) -> T:
        coro = async_func(*args, **kwargs)
        return asyncio.run(coro)

    return wrapper


async def gather_with_limit(
    *coroutines: Awaitable[T], limit: int = 10, return_exceptions: bool = False
) -> list:
    """
    Execute coroutines with concurrency limit.

    Args:
        *coroutines: Coroutines to execute
        limit: Maximum concurrent coroutines
        return_exceptions: Whether to return exceptions instead of raising

    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(limit)

    async def _limited_coro(coro):
        async with semaphore:
            return await coro

    limited_coroutines = [_limited_coro(coro) for coro in coroutines]
    return await asyncio.gather(
        *limited_coroutines, return_exceptions=return_exceptions
    )


async def async_retry(
    coro_func: Callable[..., Awaitable[T]],
    *args,
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    **kwargs,
) -> T:
    """
    Retry an async function with exponential backoff.

    Args:
        coro_func: Async function to retry
        *args: Arguments for the function
        max_attempts: Maximum retry attempts
        delay: Initial delay between retries
        backoff: Backoff multiplier
        **kwargs: Keyword arguments for the function

    Returns:
        Result of successful call

    Raises:
        Last exception if all attempts fail
    """
    last_exception = None
    current_delay = delay

    for attempt in range(max_attempts):
        try:
            return await coro_func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < max_attempts - 1:
                logger.debug(f"Async retry attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(current_delay)
                current_delay *= backoff

    raise last_exception
