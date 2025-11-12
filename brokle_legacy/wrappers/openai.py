"""
Utilities for instrumenting OpenAI clients (Pattern 1).

This module exposes a ``wrap_openai`` helper that monkey-patches the relevant
methods on an OpenAI client so every call emits Brokle traces/spans.
"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

try:
    from openai import AsyncOpenAI as _AsyncOpenAI
    from openai import OpenAI as _OpenAI

    HAS_OPENAI = True
except ImportError:
    _OpenAI = None  # type: ignore[assignment]
    _AsyncOpenAI = None  # type: ignore[assignment]
    HAS_OPENAI = False

from .._version import __version__
from ..exceptions import ProviderError
from ..observability import (
    get_client,
    get_current_span_id,
    get_current_trace_id,
)
from ..observability.span import ObservationClient
from ..observability.trace import TraceClient
from ..types.observability import ObservationLevel, SpanType

logger = logging.getLogger(__name__)

OpenAIClientT = TypeVar("OpenAIClientT")


def wrap_openai(client: OpenAIClientT) -> OpenAIClientT:
    """
    Wrap an OpenAI client so operations emit Brokle telemetry automatically.

    Args:
        client: Instance of ``openai.OpenAI`` or ``openai.AsyncOpenAI``.
    """
    if not HAS_OPENAI or _OpenAI is None or _AsyncOpenAI is None:
        raise ProviderError(
            "OpenAI SDK not installed. Install with `pip install openai>=1.0.0`."
        )

    if not isinstance(client, (_OpenAI, _AsyncOpenAI)):
        raise ProviderError(
            f"wrap_openai() expects OpenAI/AsyncOpenAI client, received {type(client).__name__}"
        )

    if getattr(client, "_brokle_instrumented", False):
        logger.debug("OpenAI client already instrumented; skipping wrap.")
        return client

    instrumentor = _OpenAIInstrumentor(client)
    instrumentor.instrument()

    setattr(client, "_brokle_instrumented", True)
    setattr(client, "_brokle_wrapper_version", __version__)
    logger.info("Instrumented OpenAI client with Brokle telemetry.")
    return client


class _OpenAIInstrumentor:
    """Encapsulates instrumentation logic for OpenAI clients."""

    def __init__(self, client: OpenAIClientT) -> None:
        self._client = client
        self._brokle_client = None  # Lazily resolved

    def instrument(self) -> None:
        """Apply monkey patches to relevant OpenAI methods."""
        if hasattr(self._client, "chat") and hasattr(self._client.chat, "completions"):
            original = self._client.chat.completions.create
            patched = self._wrap_method(original, "chat.completions.create")
            self._client.chat.completions.create = patched  # type: ignore[assignment]

        if hasattr(self._client, "completions"):
            original = self._client.completions.create
            patched = self._wrap_method(original, "completions.create")
            self._client.completions.create = patched  # type: ignore[assignment]

        if hasattr(self._client, "embeddings"):
            original = self._client.embeddings.create
            patched = self._wrap_method(original, "embeddings.create", obs_type=SpanType.EMBEDDING)
            self._client.embeddings.create = patched  # type: ignore[assignment]

    # --------------------------------------------------------------------- #
    # Wrapping helpers

    def _wrap_method(
        self,
        method: Callable[..., Any],
        method_name: str,
        *,
        obs_type: SpanType = SpanType.LLM,
    ) -> Callable[..., Any]:
        """Wrap a sync or async OpenAI client method."""

        if inspect.iscoroutinefunction(method):

            @functools.wraps(method)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await self._execute(method, method_name, obs_type, args, kwargs, is_async=True)

            return async_wrapper

        @functools.wraps(method)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._execute(method, method_name, obs_type, args, kwargs, is_async=False)

        return sync_wrapper

    # --------------------------------------------------------------------- #
    # Execution flow

    async def _execute_async_call(
        self,
        method: Callable[..., Any],
        method_name: str,
        obs_type: SpanType,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        try:
            trace_client, created_trace = self._ensure_trace(method_name, kwargs)
            span = self._start_span(
                trace_client.trace.id if trace_client else get_current_trace_id(),
                obs_type,
                method_name,
                args,
                kwargs,
            )
        except Exception as setup_error:  # pragma: no cover - defensive
            logger.debug("Failed to instrument OpenAI async call: %s", setup_error)
            return await method(*args, **kwargs)

        try:
            response = await method(*args, **kwargs)
        except Exception as exc:
            self._record_error(span, trace_client if created_trace else None, exc)
            raise
        else:
            self._record_success(span, response)
            if created_trace and trace_client:
                trace_client.end()
            return response

    def _execute(
        self,
        method: Callable[..., Any],
        method_name: str,
        obs_type: SpanType,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        *,
        is_async: bool,
    ) -> Any:
        if is_async:
            # Delegate to async path
            return self._execute_async_call(method, method_name, obs_type, args, kwargs)

        try:
            trace_client, created_trace = self._ensure_trace(method_name, kwargs)
            span = self._start_span(
                trace_client.trace.id if trace_client else get_current_trace_id(),
                obs_type,
                method_name,
                args,
                kwargs,
            )
        except Exception as setup_error:  # pragma: no cover - defensive
            logger.debug("Failed to instrument OpenAI call: %s", setup_error)
            return method(*args, **kwargs)

        try:
            response = method(*args, **kwargs)
        except Exception as exc:
            self._record_error(span, trace_client if created_trace else None, exc)
            raise
        else:
            self._record_success(span, response)
            if created_trace and trace_client:
                trace_client.end()
            return response

    # --------------------------------------------------------------------- #
    # Telemetry helpers

    def _ensure_trace(self, method_name: str, kwargs: Dict[str, Any]) -> Tuple[Optional[TraceClient], bool]:
        """
        Ensure a trace exists for the current call.

        Returns the TraceClient (if we created it) and a flag indicating whether
        a new trace was created.
        """
        brokle_client = self._resolve_brokle_client()
        existing_trace_id = get_current_trace_id()
        if existing_trace_id:
            return None, False

        trace_name = f"openai.{method_name}"
        metadata = {"provider": "openai"}
        model = kwargs.get("model")
        if model:
            metadata["model"] = str(model)

        trace_client = brokle_client.trace(name=trace_name, metadata=metadata)
        return trace_client, True

    def _start_span(
        self,
        trace_id: Optional[str],
        obs_type: SpanType,
        method_name: str,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> ObservationClient:
        if trace_id is None:
            # Defensive: ensure we have a trace
            trace_client, _ = self._ensure_trace(method_name, kwargs)
            trace_id = trace_client.trace.id if trace_client else get_current_trace_id()

        if trace_id is None:
            # As a last resort, skip instrumentation
            raise ProviderError("Unable to determine trace context for OpenAI call.")

        parent_id = get_current_span_id()

        span = ObservationClient(
            client=self._resolve_brokle_client(),
            trace_id=trace_id,
            type=obs_type,
            name=f"openai.{method_name}",
            parent_span_id=parent_id,
        )

        request_payload = _summarize_request(args, kwargs)
        if request_payload:
            span.span.input = request_payload

        model = kwargs.get("model")
        if model:
            span.span.model = str(model)

        return span

    def _record_success(self, span: SpanClient, response: Any) -> None:
        payload, usage = _summarize_response(response)
        if payload:
            span.span.output = payload
        if usage:
            span.span.usage_details = usage
        span.end()

    def _record_error(
        self,
        span: SpanClient,
        trace_client: Optional[TraceClient],
        exc: Exception,
    ) -> None:
        span.span.level = ObservationLevel.ERROR
        span.span.status_message = str(exc)
        span.end()

        if trace_client is not None:
            trace_client.trace.metadata.setdefault("error", str(exc))
            trace_client.trace.metadata.setdefault("error_type", type(exc).__name__)
            trace_client.end()

    def _resolve_brokle_client(self):
        if self._brokle_client is None:
            self._brokle_client = get_client()
        return self._brokle_client


def _summarize_request(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Collect relevant request attributes for telemetry."""
    payload: Dict[str, Any] = {}

    if "messages" in kwargs:
        payload["messages"] = kwargs["messages"]
    elif args:
        payload["messages"] = args[0]

    for key in ("model", "temperature", "max_tokens", "frequency_penalty", "presence_penalty"):
        if key in kwargs:
            payload[key] = kwargs[key]

    if "input" in kwargs:
        payload["input"] = kwargs["input"]
    if "prompt" in kwargs:
        payload["prompt"] = kwargs["prompt"]

    return payload


def _summarize_response(response: Any) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Extract output content and token usage from an OpenAI response."""
    output: Dict[str, Any] = {}
    usage: Dict[str, int] = {}

    try:
        choice = None
        choices = getattr(response, "choices", None)
        if choices and len(choices) > 0:
            choice = choices[0]

        if choice is not None:
            # Chat completion style
            message = getattr(choice, "message", None)
            if message is not None and hasattr(message, "content"):
                output["content"] = message.content
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason:
                output["finish_reason"] = finish_reason

        raw_usage = getattr(response, "usage", None)
        if raw_usage:
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = getattr(raw_usage, key, None)
                if isinstance(value, int):
                    usage[key] = value

    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Failed to summarize OpenAI response: %s", exc)

    return output, usage


__all__ = ["wrap_openai"]
