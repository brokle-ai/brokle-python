"""
Decorator-based observability helpers (Pattern 2).

The ``@observe`` decorator instruments arbitrary callables by creating either
root traces or child spans, automatically capturing inputs/outputs, and
propagating errors as structured telemetry events.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar

from .observability.context import get_client, get_current_trace_id
from .observability.span import ObservationClient
from .types.observability import ObservationLevel, SpanType

F = TypeVar("F", bound=Callable[..., Any])


def observe(
    *,
    name: Optional[str] = None,
    type: SpanType = SpanType.SPAN,
    capture_input: bool = True,
    capture_output: bool = True,
    as_trace: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> Callable[[F], F]:
    """
    Instrument a callable with automatic trace/span creation.

    Args:
        name: Optional override for the generated span/trace name.
        type: Span type to emit when not operating as a trace.
        capture_input: Whether to capture args/kwargs as structured input.
        capture_output: Whether to capture the return value as structured output.
        as_trace: Force creation of a root trace even if one already exists.
        metadata: Static metadata to attach to the span or trace.
    """

    def decorator(func: F) -> F:
        func_name = name or func.__name__

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                client = _safe_get_client()
                if client is None:
                    return await func(*args, **kwargs)

                current_trace_id = get_current_trace_id()
                if as_trace or current_trace_id is None:
                    return await _execute_as_trace_async(
                        func,
                        func_name,
                        client,
                        args,
                        kwargs,
                        capture_input,
                        capture_output,
                        metadata,
                    )

                return await _execute_as_span_async(
                    func,
                    func_name,
                    client,
                    current_trace_id,
                    type,
                    args,
                    kwargs,
                    capture_input,
                    capture_output,
                    metadata,
                )

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            client = _safe_get_client()
            if client is None:
                return func(*args, **kwargs)

            current_trace_id = get_current_trace_id()
            if as_trace or current_trace_id is None:
                return _execute_as_trace(
                    func,
                    func_name,
                    client,
                    args,
                    kwargs,
                    capture_input,
                    capture_output,
                    metadata,
                )

            return _execute_as_span(
                func,
                func_name,
                client,
                current_trace_id,
                type,
                args,
                kwargs,
                capture_input,
                capture_output,
                metadata,
            )

        return sync_wrapper  # type: ignore[return-value]

    return decorator


def observe_llm(
    *, name: Optional[str] = None, capture_input: bool = True, capture_output: bool = True
) -> Callable[[F], F]:
    """Convenience decorator for LLM spans."""
    return observe(
        name=name,
        type=SpanType.LLM,
        capture_input=capture_input,
        capture_output=capture_output,
    )


def observe_retrieval(
    *,
    name: Optional[str] = None,
    capture_input: bool = True,
    capture_output: bool = True,
) -> Callable[[F], F]:
    """Convenience decorator for retrieval/tooling spans."""
    return observe(
        name=name,
        type=SpanType.RETRIEVAL,
        capture_input=capture_input,
        capture_output=capture_output,
    )


def trace_workflow(
    *, name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
) -> Callable[[F], F]:
    """Decorator that always emits a root trace."""
    return observe(name=name, as_trace=True, metadata=metadata)


# ---- Helpers -----------------------------------------------------------------


def _safe_get_client():
    """Fetch a Brokle client, suppressing configuration errors."""
    try:
        return get_client()
    except Exception:
        return None


def _capture_function_input(func: Callable[..., Any], args: tuple, kwargs: dict) -> Dict[str, Any]:
    """Best-effort capture of function call arguments."""
    try:
        signature = inspect.signature(func)
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except Exception:
        return {"args": args, "kwargs": kwargs}


def _execute_as_trace(
    func: Callable[..., Any],
    name: str,
    client,
    args: tuple,
    kwargs: dict,
    capture_input: bool,
    capture_output: bool,
    metadata: Optional[Dict[str, Any]],
) -> Any:
    trace = client.trace(name=name, metadata=metadata or {})

    if capture_input:
        trace.trace.input = _capture_function_input(func, args, kwargs)

    try:
        result = func(*args, **kwargs)
    except Exception as exc:
        trace.trace.metadata.setdefault("error", str(exc))
        trace.trace.metadata.setdefault("error_type", type(exc).__name__)
        trace.end()
        raise
    else:
        if capture_output:
            trace.trace.output = {"result": result}
        trace.end()
        return result


def _execute_as_span(
    func: Callable[..., Any],
    name: str,
    client,
    trace_id: str,
    obs_type: SpanType,
    args: tuple,
    kwargs: dict,
    capture_input: bool,
    capture_output: bool,
    metadata: Optional[Dict[str, Any]],
) -> Any:
    obs = ObservationClient(
        client=client,
        trace_id=trace_id,
        type=obs_type,
        name=name,
        metadata=metadata or {},
    )

    if capture_input:
        obs.span.input = _capture_function_input(func, args, kwargs)

    try:
        result = func(*args, **kwargs)
    except Exception as exc:
        obs.span.level = ObservationLevel.ERROR
        obs.span.status_message = str(exc)
        obs.end()
        raise
    else:
        if capture_output:
            obs.span.output = {"result": result}
        obs.end()
        return result


async def _execute_as_trace_async(
    func: Callable[..., Awaitable[Any]],
    name: str,
    client,
    args: tuple,
    kwargs: dict,
    capture_input: bool,
    capture_output: bool,
    metadata: Optional[Dict[str, Any]],
) -> Any:
    trace = client.trace(name=name, metadata=metadata or {})

    if capture_input:
        trace.trace.input = _capture_function_input(func, args, kwargs)

    try:
        result = await func(*args, **kwargs)
    except Exception as exc:
        trace.trace.metadata.setdefault("error", str(exc))
        trace.trace.metadata.setdefault("error_type", type(exc).__name__)
        trace.end()
        raise
    else:
        if capture_output:
            trace.trace.output = {"result": result}
        trace.end()
        return result


async def _execute_as_span_async(
    func: Callable[..., Awaitable[Any]],
    name: str,
    client,
    trace_id: str,
    obs_type: SpanType,
    args: tuple,
    kwargs: dict,
    capture_input: bool,
    capture_output: bool,
    metadata: Optional[Dict[str, Any]],
) -> Any:
    obs = ObservationClient(
        client=client,
        trace_id=trace_id,
        type=obs_type,
        name=name,
        metadata=metadata or {},
    )

    if capture_input:
        obs.span.input = _capture_function_input(func, args, kwargs)

    try:
        result = await func(*args, **kwargs)
    except Exception as exc:
        obs.span.level = ObservationLevel.ERROR
        obs.span.status_message = str(exc)
        obs.end()
        raise
    else:
        if capture_output:
            obs.span.output = {"result": result}
        obs.end()
        return result


__all__ = ["observe", "observe_llm", "observe_retrieval", "trace_workflow"]
