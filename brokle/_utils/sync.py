"""
Sync Utilities

Utilities for running async code in synchronous contexts.
"""

import asyncio
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
