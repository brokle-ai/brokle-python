"""
Utilities for instrumenting Anthropic clients (Pattern 1).
"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

try:
    from anthropic import Anthropic as _Anthropic
    from anthropic import AsyncAnthropic as _AsyncAnthropic

    HAS_ANTHROPIC = True
except ImportError:
    _Anthropic = None  # type: ignore[assignment]
    _AsyncAnthropic = None  # type: ignore[assignment]
    HAS_ANTHROPIC = False

from .._version import __version__
from ..exceptions import ProviderError
from ..observability import get_client, get_current_span_id, get_current_trace_id
from ..observability.span import ObservationClient
from ..observability.trace import TraceClient
from ..types.observability import ObservationLevel, SpanType

logger = logging.getLogger(__name__)

AnthropicClientT = TypeVar("AnthropicClientT")


def wrap_anthropic(client: AnthropicClientT) -> AnthropicClientT:
    """Wrap an Anthropic client so messages emit Brokle telemetry."""
    if not HAS_ANTHROPIC or _Anthropic is None or _AsyncAnthropic is None:
        raise ProviderError(
            "Anthropic SDK not installed. Install with `pip install anthropic>=0.5.0`."
        )

    if not isinstance(client, (_Anthropic, _AsyncAnthropic)):
        raise ProviderError(
            f"wrap_anthropic() expects Anthropic/AsyncAnthropic client, received {type(client).__name__}"
        )

    if getattr(client, "_brokle_instrumented", False):
        logger.debug("Anthropic client already instrumented; skipping wrap.")
        return client

    instrumentor = _AnthropicInstrumentor(client)
    instrumentor.instrument()

    setattr(client, "_brokle_instrumented", True)
    setattr(client, "_brokle_wrapper_version", __version__)
    logger.info("Instrumented Anthropic client with Brokle telemetry.")
    return client


class _AnthropicInstrumentor:
    """Encapsulates instrumentation logic for Anthropic clients."""

    def __init__(self, client: AnthropicClientT) -> None:
        self._client = client
        self._brokle_client = None

    def instrument(self) -> None:
        if hasattr(self._client, "messages"):
            original = self._client.messages.create
            patched = self._wrap_method(original, "messages.create")
            self._client.messages.create = patched  # type: ignore[assignment]

    def _wrap_method(
        self,
        method: Callable[..., Any],
        method_name: str,
        *,
        obs_type: SpanType = SpanType.LLM,
    ) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(method):

            @functools.wraps(method)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await self._execute(method, method_name, obs_type, args, kwargs, is_async=True)

            return async_wrapper

        @functools.wraps(method)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._execute(method, method_name, obs_type, args, kwargs, is_async=False)

        return sync_wrapper

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
            logger.debug("Failed to instrument Anthropic async call: %s", setup_error)
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
            logger.debug("Failed to instrument Anthropic call: %s", setup_error)
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

    def _ensure_trace(self, method_name: str, kwargs: Dict[str, Any]) -> Tuple[Optional[TraceClient], bool]:
        brokle_client = self._resolve_brokle_client()
        existing_trace_id = get_current_trace_id()
        if existing_trace_id:
            return None, False

        trace_name = f"anthropic.{method_name}"
        metadata = {"provider": "anthropic"}
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
            trace_client, _ = self._ensure_trace(method_name, kwargs)
            trace_id = trace_client.trace.id if trace_client else get_current_trace_id()

        if trace_id is None:
            raise ProviderError("Unable to determine trace context for Anthropic call.")

        parent_id = get_current_span_id()

        span = ObservationClient(
            client=self._resolve_brokle_client(),
            trace_id=trace_id,
            type=obs_type,
            name=f"anthropic.{method_name}",
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
    payload: Dict[str, Any] = {}

    if "messages" in kwargs:
        payload["messages"] = kwargs["messages"]
    elif args:
        payload["messages"] = args[0]

    for key in ("model", "max_tokens", "temperature", "top_p"):
        if key in kwargs:
            payload[key] = kwargs[key]

    if "input" in kwargs:
        payload["input"] = kwargs["input"]

    return payload


def _summarize_response(response: Any) -> Tuple[Dict[str, Any], Dict[str, int]]:
    output: Dict[str, Any] = {}
    usage: Dict[str, int] = {}

    try:
        content = getattr(response, "content", None)
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict):
                output["content"] = first.get("text") or first.get("content")
            elif hasattr(first, "text"):
                output["content"] = first.text

        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason:
            output["finish_reason"] = stop_reason

        raw_usage = getattr(response, "usage", None)
        if raw_usage:
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                value = getattr(raw_usage, key, None)
                if isinstance(value, int):
                    usage_key = "prompt_tokens" if key == "input_tokens" else "completion_tokens" if key == "output_tokens" else key
                    usage[usage_key] = value
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Failed to summarize Anthropic response: %s", exc)

    return output, usage


__all__ = ["wrap_anthropic"]
