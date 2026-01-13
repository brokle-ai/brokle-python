"""
Integration Hook Registry System

Provides a centralized system for registering and executing hooks
at various points in the integration lifecycle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
from opentelemetry.trace import Span

logger = logging.getLogger(__name__)


class HookType(Enum):
    """Hook type enumeration."""
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    STREAM = "stream"


@dataclass
class HookContext:
    """Base hook context passed to hook handlers."""
    integration_name: str
    operation: str
    span: Optional[Span] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RequestHookContext(HookContext):
    """Request hook context - before an LLM call."""
    request: Dict[str, Any] = field(default_factory=dict)
    model: Optional[str] = None


@dataclass
class ResponseHookContext(HookContext):
    """Response hook context - after an LLM call."""
    response: Any = None
    request: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    usage: Optional[Dict[str, int]] = None


@dataclass
class ErrorHookContext(HookContext):
    """Error hook context - when an error occurs."""
    error: Exception = field(default_factory=Exception)
    request: Optional[Dict[str, Any]] = None


@dataclass
class StreamHookContext(HookContext):
    """Streaming hook context - for streaming operations."""
    chunk: Any = None
    is_first: bool = False
    is_last: bool = False


# Type aliases for hook handlers
RequestHook = Callable[[RequestHookContext], None]
ResponseHook = Callable[[ResponseHookContext], None]
ErrorHook = Callable[[ErrorHookContext], None]
StreamHook = Callable[[StreamHookContext], None]
HookHandler = Union[RequestHook, ResponseHook, ErrorHook, StreamHook]


@dataclass
class RegisteredHook:
    """Registered hook entry."""
    id: str
    type: HookType
    handler: HookHandler
    priority: int = 100
    integration_filter: Optional[Union[str, List[str]]] = None


@dataclass
class HookRegistrationOptions:
    """Hook registration options."""
    priority: int = 100
    integration_filter: Optional[Union[str, List[str]]] = None


class HookRegistry:
    """
    Hook Registry - manages integration lifecycle hooks.

    Provides a centralized system for registering and executing hooks
    at various points in the integration lifecycle.

    Example:
        >>> registry = HookRegistry()
        >>> def on_request(ctx: RequestHookContext):
        ...     print(f"Request to {ctx.model}")
        >>> hook_id = registry.on_request(on_request)
        >>> # Later...
        >>> registry.unregister(hook_id)
    """

    def __init__(self) -> None:
        self._hooks: Dict[HookType, List[RegisteredHook]] = {
            HookType.REQUEST: [],
            HookType.RESPONSE: [],
            HookType.ERROR: [],
            HookType.STREAM: [],
        }
        self._hook_counter = 0

    def on_request(
        self,
        handler: RequestHook,
        options: Optional[HookRegistrationOptions] = None
    ) -> str:
        """Register a request hook."""
        return self._register(HookType.REQUEST, handler, options)

    def on_response(
        self,
        handler: ResponseHook,
        options: Optional[HookRegistrationOptions] = None
    ) -> str:
        """Register a response hook."""
        return self._register(HookType.RESPONSE, handler, options)

    def on_error(
        self,
        handler: ErrorHook,
        options: Optional[HookRegistrationOptions] = None
    ) -> str:
        """Register an error hook."""
        return self._register(HookType.ERROR, handler, options)

    def on_stream(
        self,
        handler: StreamHook,
        options: Optional[HookRegistrationOptions] = None
    ) -> str:
        """Register a stream hook."""
        return self._register(HookType.STREAM, handler, options)

    def _register(
        self,
        hook_type: HookType,
        handler: HookHandler,
        options: Optional[HookRegistrationOptions] = None
    ) -> str:
        """Register a hook."""
        self._hook_counter += 1
        hook_id = f"hook_{self._hook_counter}"

        opts = options or HookRegistrationOptions()

        hook = RegisteredHook(
            id=hook_id,
            type=hook_type,
            handler=handler,
            priority=opts.priority,
            integration_filter=opts.integration_filter,
        )

        hooks = self._hooks[hook_type]
        hooks.append(hook)
        hooks.sort(key=lambda h: h.priority)

        return hook_id

    def unregister(self, hook_id: str) -> bool:
        """Unregister a hook by ID."""
        for hook_type, hooks in self._hooks.items():
            for i, hook in enumerate(hooks):
                if hook.id == hook_id:
                    hooks.pop(i)
                    return True
        return False

    def execute_request_hooks(self, ctx: RequestHookContext) -> None:
        """Execute request hooks synchronously."""
        self._execute(HookType.REQUEST, ctx)

    def execute_response_hooks(self, ctx: ResponseHookContext) -> None:
        """Execute response hooks synchronously."""
        self._execute(HookType.RESPONSE, ctx)

    def execute_error_hooks(self, ctx: ErrorHookContext) -> None:
        """Execute error hooks synchronously."""
        self._execute(HookType.ERROR, ctx)

    def execute_stream_hooks(self, ctx: StreamHookContext) -> None:
        """Execute stream hooks synchronously."""
        self._execute(HookType.STREAM, ctx)

    async def execute_request_hooks_async(self, ctx: RequestHookContext) -> None:
        """Execute request hooks asynchronously."""
        await self._execute_async(HookType.REQUEST, ctx)

    async def execute_response_hooks_async(self, ctx: ResponseHookContext) -> None:
        """Execute response hooks asynchronously."""
        await self._execute_async(HookType.RESPONSE, ctx)

    async def execute_error_hooks_async(self, ctx: ErrorHookContext) -> None:
        """Execute error hooks asynchronously."""
        await self._execute_async(HookType.ERROR, ctx)

    async def execute_stream_hooks_async(self, ctx: StreamHookContext) -> None:
        """Execute stream hooks asynchronously."""
        await self._execute_async(HookType.STREAM, ctx)

    def _execute(self, hook_type: HookType, ctx: HookContext) -> None:
        """Execute hooks of a specific type synchronously."""
        hooks = self._hooks.get(hook_type, [])

        for hook in hooks:
            # Check integration filter
            if hook.integration_filter:
                filters = (
                    hook.integration_filter
                    if isinstance(hook.integration_filter, list)
                    else [hook.integration_filter]
                )
                if ctx.integration_name not in filters:
                    continue

            try:
                hook.handler(ctx)  # type: ignore
            except Exception as e:
                logger.error(f"Hook {hook.id} failed: {e}")

    async def _execute_async(self, hook_type: HookType, ctx: HookContext) -> None:
        """Execute hooks of a specific type asynchronously."""
        import asyncio

        hooks = self._hooks.get(hook_type, [])

        for hook in hooks:
            # Check integration filter
            if hook.integration_filter:
                filters = (
                    hook.integration_filter
                    if isinstance(hook.integration_filter, list)
                    else [hook.integration_filter]
                )
                if ctx.integration_name not in filters:
                    continue

            try:
                result = hook.handler(ctx)  # type: ignore
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Hook {hook.id} failed: {e}")

    def clear(self) -> None:
        """Clear all hooks."""
        for hook_type in self._hooks:
            self._hooks[hook_type] = []

    def get_hook_count(self, hook_type: Optional[HookType] = None) -> int:
        """Get count of registered hooks."""
        if hook_type:
            return len(self._hooks.get(hook_type, []))
        return sum(len(hooks) for hooks in self._hooks.values())


# Global hook registry instance
_global_registry: Optional[HookRegistry] = None


def get_hook_registry() -> HookRegistry:
    """Get the global hook registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = HookRegistry()
    return _global_registry


def reset_hook_registry() -> None:
    """Reset the global hook registry (mainly for testing)."""
    global _global_registry
    _global_registry = None
