"""
Sync Utilities

Utilities for running async code in synchronous contexts.
"""

import asyncio
import concurrent.futures
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run async coroutine synchronously.

    This function runs an async coroutine in a synchronous context.
    It raises RuntimeError if called inside a running event loop,
    directing users to use AsyncBrokle instead.

    Args:
        coro: Async coroutine to run

    Returns:
        Result of the coroutine

    Raises:
        RuntimeError: If called inside an async event loop

    Example:
        >>> async def fetch_data():
        ...     return 'data'
        >>> result = run_sync(fetch_data())
        >>> print(result)
        'data'
    """
    try:
        asyncio.get_running_loop()
        raise RuntimeError(
            "Brokle cannot be used inside an async event loop. "
            "Use AsyncBrokle instead."
        )
    except RuntimeError as e:
        if "Brokle cannot be used" in str(e):
            raise

    return asyncio.run(coro)


def run_sync_safely(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run async coroutine with event loop escape hatch.

    Safe for Jupyter notebooks and nested event loop scenarios.
    Falls back to nest_asyncio if available, otherwise creates new thread.

    This function should be used when you need to run async code from a
    synchronous context but might already be inside an event loop (e.g.,
    Jupyter notebooks, async frameworks).

    Args:
        coro: Async coroutine to run

    Returns:
        Result of the coroutine

    Example:
        >>> async def fetch_data():
        ...     return 'data'
        >>> # Works even inside Jupyter or async context
        >>> result = run_sync_safely(fetch_data())
        >>> print(result)
        'data'
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No event loop - safe to use asyncio.run()
        return asyncio.run(coro)

    # Inside event loop - need escape hatch

    # Try nest_asyncio first (common in Jupyter)
    try:
        import nest_asyncio

        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    except ImportError:
        pass

    # Fallback: run in separate thread with new event loop
    def _run_in_new_loop() -> T:
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_in_new_loop)
        return future.result()
