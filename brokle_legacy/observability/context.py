"""
Thread-local context helpers for observability.

This module keeps track of the active Brokle client along with the current
trace/observation stacks so Pattern 1 (wrappers) and Pattern 2 (decorators)
can automatically build the proper hierarchy without requiring callers to pass
identifiers around manually.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Dict, List, Optional

from ..client import Brokle, get_client as get_global_client


class ObservabilityContext:
    """Per-thread observability context."""

    __slots__ = ("client", "trace_stack", "observation_stack", "session_id")

    def __init__(self) -> None:
        self.client: Optional[Brokle] = None
        self.trace_stack: List[str] = []
        self.observation_stack: List[str] = []
        self.session_id: Optional[str] = None


_context = threading.local()


def _get_or_create_context() -> ObservabilityContext:
    """Return the thread-local context, creating it on first access."""
    ctx = getattr(_context, "value", None)
    if ctx is None:
        ctx = ObservabilityContext()
        _context.value = ctx
    return ctx


def get_context() -> ObservabilityContext:
    """Expose the raw ObservabilityContext for advanced usage/testing."""
    return _get_or_create_context()


def clear_context() -> None:
    """Reset the thread-local context (useful in tests)."""
    _context.value = ObservabilityContext()


def set_client(client: Optional[Brokle]) -> None:
    """Manually set the Brokle client for the active thread."""
    _get_or_create_context().client = client


def get_client(
    *,
    api_key: Optional[str] = None,
    host: Optional[str] = None,
    environment: Optional[str] = None,
    **kwargs,
) -> Brokle:
    """
    Get or lazily create a Brokle client bound to the current thread.

    If explicit configuration is supplied the client is instantiated directly
    with those parameters. Otherwise we fall back to the global singleton from
    ``brokle.client.get_client()`` which reads configuration from the environment.
    """
    ctx = _get_or_create_context()
    if ctx.client is not None:
        return ctx.client

    explicit_kwargs: Dict[str, object] = {}
    if api_key is not None:
        explicit_kwargs["api_key"] = api_key
    if host is not None:
        explicit_kwargs["host"] = host
    if environment is not None:
        explicit_kwargs["environment"] = environment

    # Pass through additional keyword arguments when explicitly provided.
    explicit_kwargs.update({k: v for k, v in kwargs.items() if v is not None})

    if explicit_kwargs:
        client = Brokle(**explicit_kwargs)
    else:
        client = get_global_client()

    ctx.client = client
    return client


def get_client_context() -> Optional[Brokle]:
    """Return the client currently stored in the thread-local context, if any."""
    return _get_or_create_context().client


@contextmanager
def client_context(client: Brokle):
    """Temporarily set the active Brokle client within a ``with`` block."""
    previous = get_client_context()
    set_client(client)
    try:
        yield client
    finally:
        if previous is not None:
            set_client(previous)
        else:
            clear_context()


def get_context_info() -> Dict[str, object]:
    """Return a lightweight snapshot describing the current context."""
    ctx = _get_or_create_context()
    client = ctx.client
    api_key_display: Optional[str] = None
    if client and client.config.api_key:
        key = client.config.api_key
        api_key_display = f"{key[:10]}..." if len(key) > 10 else key
    return {
        "has_client": client is not None,
        "environment": client.config.environment if client else None,
        "host": client.config.host if client else None,
        "api_key": api_key_display,
        "session_id": ctx.session_id,
        "trace_depth": len(ctx.trace_stack),
        "observation_depth": len(ctx.observation_stack),
    }


def get_current_trace_id() -> Optional[str]:
    """Return the active trace identifier, if any."""
    stack = _get_or_create_context().trace_stack
    return stack[-1] if stack else None


def push_trace(trace_id: str) -> None:
    """Push a trace identifier onto the stack."""
    _get_or_create_context().trace_stack.append(trace_id)


def pop_trace() -> Optional[str]:
    """Pop the most recent trace identifier."""
    stack = _get_or_create_context().trace_stack
    return stack.pop() if stack else None


def get_current_observation_id() -> Optional[str]:
    """Return the active observation identifier, if any."""
    stack = _get_or_create_context().observation_stack
    return stack[-1] if stack else None


def push_observation(observation_id: str) -> None:
    """Push an observation identifier onto the stack."""
    _get_or_create_context().observation_stack.append(observation_id)


def pop_observation() -> Optional[str]:
    """Pop the most recent observation identifier."""
    stack = _get_or_create_context().observation_stack
    return stack.pop() if stack else None


def set_session_id(session_id: Optional[str]) -> None:
    """Associate a session identifier with the current context."""
    _get_or_create_context().session_id = session_id


def get_session_id() -> Optional[str]:
    """Return the session identifier stored in the current context."""
    return _get_or_create_context().session_id


__all__ = [
    "ObservabilityContext",
    "clear_context",
    "client_context",
    "get_client",
    "get_client_context",
    "get_context",
    "get_context_info",
    "get_current_observation_id",
    "get_current_trace_id",
    "get_session_id",
    "pop_observation",
    "pop_trace",
    "push_observation",
    "push_trace",
    "set_client",
    "set_session_id",
]
