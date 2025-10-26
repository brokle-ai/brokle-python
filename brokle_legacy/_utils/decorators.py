"""
Decorator utilities for Brokle SDK.

Simple decorator utilities extracted and cleaned from the old integration framework.
"""

import functools
import logging
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def safe_wrapper(func: F) -> F:
    """
    Create a safe wrapper that catches and logs exceptions.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function that handles exceptions safely
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.debug(f"Safe wrapper caught exception in {func.__name__}: {e}")
            return None

    return wrapper


def async_safe_wrapper(func: F) -> F:
    """
    Create a safe wrapper for async functions.

    Args:
        func: Async function to wrap

    Returns:
        Wrapped async function that handles exceptions safely
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.debug(f"Async safe wrapper caught exception in {func.__name__}: {e}")
            return None

    return wrapper


def retry_decorator(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    Decorator for retrying function calls with exponential backoff.

    Args:
        max_attempts: Maximum retry attempts
        delay: Initial delay between retries
        backoff: Backoff multiplier
        exceptions: Tuple of exceptions to catch and retry

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time

            last_exception = None
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.debug(f"Retry attempt {attempt + 1} failed: {e}")
                        time.sleep(current_delay)
                        current_delay *= backoff

            raise last_exception

        return wrapper

    return decorator


def conditional_decorator(condition: bool):
    """
    Apply decorator only if condition is True.

    Args:
        condition: Whether to apply the decorator

    Returns:
        Decorator function or identity function
    """

    def decorator(func_or_decorator):
        if condition:
            return func_or_decorator
        else:
            # If condition is False, return identity function
            def identity(func):
                return func

            return identity

    return decorator


def memoize_simple(func: F) -> F:
    """
    Simple memoization decorator for functions with hashable arguments.

    Args:
        func: Function to memoize

    Returns:
        Memoized function
    """
    cache = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create cache key from args and sorted kwargs
        key = (args, tuple(sorted(kwargs.items())))

        if key not in cache:
            cache[key] = func(*args, **kwargs)

        return cache[key]

    wrapper.cache_clear = cache.clear
    wrapper.cache_info = lambda: f"Cache size: {len(cache)}"

    return wrapper


def deprecated(reason: str = "This function is deprecated"):
    """
    Mark a function as deprecated.

    Args:
        reason: Deprecation reason

    Returns:
        Decorated function that warns when called
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import warnings

            warnings.warn(
                f"{func.__name__} is deprecated: {reason}",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator
