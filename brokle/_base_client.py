"""
Base Brokle Client

Provides the base class with shared initialization, configuration,
and OpenTelemetry setup used by both Brokle and AsyncBrokle.
"""

import atexit
import json
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional
from uuid import UUID

from opentelemetry import trace

if TYPE_CHECKING:
    from .prompts import Prompt
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Span, TracerProvider
from opentelemetry.sdk.trace.sampling import ALWAYS_ON, TraceIdRatioBased
from opentelemetry.trace import SpanKind, Status, StatusCode, Tracer

from .config import BrokleConfig
from .exporter import create_exporter_for_config
from .logs import BrokleLoggerProvider
from .metrics import BrokleMeterProvider, GenAIMetrics, create_genai_metrics
from .processor import BrokleSpanProcessor
from .prompts import CacheOptions, PromptConfig
from .types import Attrs, LLMProvider, OperationType, SchemaURLs, SpanType

class BaseBrokleClient:
    """
    Base Brokle client with shared initialization and OpenTelemetry setup.

    This class contains the core functionality shared by both Brokle
    (sync) and AsyncBrokle (async). It handles:
    - Configuration management
    - OpenTelemetry tracer/meter/logger providers
    - HTTP client initialization
    - Span creation methods
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:8080",
        environment: str = "default",
        debug: bool = False,
        tracing_enabled: bool = True,
        metrics_enabled: bool = True,
        release: Optional[str] = None,
        version: Optional[str] = None,
        sample_rate: float = 1.0,
        mask: Optional[callable] = None,
        flush_at: int = 100,
        flush_interval: float = 5.0,
        timeout: int = 30,
        config: Optional[BrokleConfig] = None,
        **kwargs,
    ):
        """
        Initialize Brokle client.

        Args:
            api_key: Brokle API key (required, must start with 'bk_')
            base_url: Brokle API base URL
            environment: Environment tag (e.g., 'production', 'staging')
            debug: Enable debug logging
            tracing_enabled: Enable/disable tracing (if False, all calls are no-ops)
            metrics_enabled: Enable/disable metrics collection (if False, no metrics recorded)
            release: Release identifier for deployment tracking (e.g., 'v2.1.24', 'abc123')
            version: Trace-level version for A/B testing experiments (e.g., 'experiment-A', 'control')
            sample_rate: Sampling rate for traces (0.0 to 1.0)
            mask: Optional function to mask sensitive data
            flush_at: Maximum batch size before flush (1-1000)
            flush_interval: Maximum delay in seconds before flush (0.1-60.0)
            timeout: HTTP timeout in seconds
            config: Pre-built BrokleConfig object (if provided, other params are ignored)
            **kwargs: Additional configuration options

        Raises:
            ValueError: If configuration is invalid
        """
        if config is not None:
            self.config = config
        else:
            self.config = BrokleConfig(
                api_key=api_key or "",  # Will be validated by BrokleConfig
                base_url=base_url,
                environment=environment,
                debug=debug,
                tracing_enabled=tracing_enabled,
                metrics_enabled=metrics_enabled,
                release=release,
                version=version,
                sample_rate=sample_rate,
                mask=mask,
                flush_at=flush_at,
                flush_interval=flush_interval,
                timeout=timeout,
                **kwargs,
            )

        self._prompt_config = PromptConfig()
        self._prompts_manager = None
        self._datasets_manager = None
        self._scores_manager = None
        self._experiments_manager = None

        self._meter_provider: Optional[BrokleMeterProvider] = None
        self._metrics: Optional[GenAIMetrics] = None
        self._logger_provider: Optional[BrokleLoggerProvider] = None

        resource = Resource.create({}, schema_url=SchemaURLs.DEFAULT)

        resource_attrs = {}
        if self.config.release:
            resource_attrs[Attrs.BROKLE_RELEASE] = self.config.release
        if self.config.version:
            resource_attrs[Attrs.BROKLE_VERSION] = self.config.version

        if resource_attrs:
            resource = resource.merge(Resource.create(resource_attrs))

        if not self.config.tracing_enabled:
            self._tracer = trace.get_tracer(__name__)
            self._provider = None
            self._processor = None
        else:
            if self.config.sample_rate < 1.0:
                sampler = TraceIdRatioBased(self.config.sample_rate)
            else:
                sampler = ALWAYS_ON

            self._provider = TracerProvider(resource=resource, sampler=sampler)

            exporter = create_exporter_for_config(self.config)
            self._processor = BrokleSpanProcessor(
                span_exporter=exporter,
                config=self.config,
            )
            self._provider.add_span_processor(self._processor)

            self._tracer = self._provider.get_tracer(
                instrumenting_module_name="brokle",
                instrumenting_library_version=self._get_sdk_version(),
                schema_url=SchemaURLs.DEFAULT,
            )

        if self.config.metrics_enabled:
            try:
                self._meter_provider = BrokleMeterProvider(
                    config=self.config,
                    resource=resource,
                )
                meter = self._meter_provider.get_meter()
                self._metrics = create_genai_metrics(meter)
            except ImportError as e:
                import logging

                logging.getLogger(__name__).warning(
                    f"Metrics disabled: {e}. Install opentelemetry-exporter-otlp-proto-http."
                )

        if self.config.logs_enabled:
            try:
                self._logger_provider = BrokleLoggerProvider(
                    config=self.config,
                    resource=resource,
                )
            except ImportError as e:
                import logging

                logging.getLogger(__name__).warning(
                    f"Logs disabled: {e}. Install opentelemetry-exporter-otlp-proto-http."
                )

        # Register cleanup on process exit
        atexit.register(self._cleanup)

    @staticmethod
    def _extract_project_id(api_key: Optional[str]) -> str:
        """
        Extract project ID from API key.

        For now, we use the API key itself as the project identifier.
        The backend will validate this during authentication.

        Args:
            api_key: Brokle API key

        Returns:
            Project identifier string
        """
        if not api_key:
            return "unknown"
        return api_key[:20]

    @staticmethod
    def _get_sdk_version() -> str:
        """Get SDK version."""
        try:
            from . import __version__

            return __version__
        except (ImportError, AttributeError):
            return "0.1.0-dev"

    def _cleanup(self):
        """Cleanup handler called on process exit."""
        if self._processor:
            self.flush()
            self._processor.shutdown()
        if self._meter_provider:
            self._meter_provider.shutdown()
        if self._logger_provider:
            self._logger_provider.shutdown()

    @contextmanager
    def start_as_current_span(
        self,
        name: str,
        as_type: Optional[str] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        prompt: Optional["Prompt"] = None,
        **kwargs,
    ) -> Iterator[Span]:
        """
        Create a span using context manager (OpenTelemetry standard pattern).

        This is the recommended way to create spans as it automatically handles
        span lifecycle and context propagation.

        Args:
            name: Span name
            as_type: Span type for categorization (span, generation, tool, agent, chain, etc.)
            kind: Span kind (INTERNAL, CLIENT, SERVER, PRODUCER, CONSUMER)
            attributes: Initial span attributes
            version: Version identifier for A/B testing and experiment tracking
            input: Input data (LLM messages or generic data)
                   - LLM format: [{"role": "user", "content": "..."}]
                   - Generic format: {"query": "...", "count": 5} or any value
            output: Output data (LLM messages or generic data)
            prompt: Prompt to link to this span (fallback prompts are not linked)
            **kwargs: Additional arguments passed to tracer.start_as_current_span()

        Yields:
            Span instance

        Example:
            >>> # With prompt linking
            >>> prompt = client.prompts.get_sync("greeting")
            >>> with client.start_as_current_span("process", prompt=prompt) as span:
            ...     result = do_work()
            ...     span.set_attribute(Attrs.OUTPUT_VALUE, json.dumps(result))
            >>>
            >>> # LLM messages
            >>> with client.start_as_current_span("llm-trace",
            ...     input=[{"role": "user", "content": "Hello"}]) as span:
            ...     pass
        """
        attrs = attributes.copy() if attributes else {}

        if version:
            attrs[Attrs.BROKLE_VERSION] = version

        if as_type:
            attrs[Attrs.BROKLE_SPAN_TYPE] = as_type
        elif Attrs.BROKLE_SPAN_TYPE not in attrs:
            attrs[Attrs.BROKLE_SPAN_TYPE] = SpanType.SPAN

        # Handle input (auto-detect LLM messages vs generic data)
        if input is not None:
            if _is_llm_messages_format(input):
                # LLM messages → use OTLP GenAI standard
                attrs[Attrs.GEN_AI_INPUT_MESSAGES] = json.dumps(input)
            else:
                # Generic data → use OpenInference pattern
                input_str, mime_type = _serialize_with_mime(input)
                attrs[Attrs.INPUT_VALUE] = input_str
                attrs[Attrs.INPUT_MIME_TYPE] = mime_type

        # Handle output (auto-detect LLM messages vs generic data)
        if output is not None:
            if _is_llm_messages_format(output):
                # LLM messages → use OTLP GenAI standard
                attrs[Attrs.GEN_AI_OUTPUT_MESSAGES] = json.dumps(output)
            else:
                # Generic data → use OpenInference pattern
                output_str, mime_type = _serialize_with_mime(output)
                attrs[Attrs.OUTPUT_VALUE] = output_str
                attrs[Attrs.OUTPUT_MIME_TYPE] = mime_type

        # Link prompt if provided and NOT a fallback
        if prompt and not prompt.is_fallback:
            attrs[Attrs.BROKLE_PROMPT_NAME] = prompt.name
            attrs[Attrs.BROKLE_PROMPT_VERSION] = prompt.version
            if prompt.id and prompt.id != "fallback":
                attrs[Attrs.BROKLE_PROMPT_ID] = prompt.id

        with self._tracer.start_as_current_span(
            name=name,
            kind=kind,
            attributes=attrs,
            **kwargs,
        ) as span:
            yield span

    @contextmanager
    def start_as_current_generation(
        self,
        name: str,
        model: str,
        provider: str,
        input_messages: Optional[List[Dict[str, Any]]] = None,
        model_parameters: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None,
        prompt: Optional["Prompt"] = None,
        **kwargs,
    ) -> Iterator[Span]:
        """
        Create an LLM generation span (OTEL 1.28+ compliant).

        This method creates a span with GenAI semantic attributes following
        OpenTelemetry 1.28+ GenAI conventions.

        Args:
            name: Operation name (e.g., "chat", "completion")
            model: Model identifier (e.g., "gpt-4", "claude-3-opus")
            provider: Provider name (e.g., "openai", "anthropic")
            input_messages: Input messages in OTEL format
            model_parameters: Model parameters (temperature, max_tokens, etc.)
            version: Version identifier for A/B testing and experiment tracking
            prompt: Prompt to link to this generation (fallback prompts are not linked)
            **kwargs: Additional span attributes

        Yields:
            Span instance

        Example:
            >>> prompt = client.prompts.get_sync("movie-critic")
            >>> with client.start_as_current_generation(
            ...     name="chat",
            ...     model="gpt-4",
            ...     provider="openai",
            ...     input_messages=[{"role": "user", "content": "Hello"}],
            ...     prompt=prompt,
            ... ) as gen:
            ...     # Make LLM call
            ...     gen.set_attribute(Attrs.GEN_AI_OUTPUT_MESSAGES, [...])
        """
        attrs = {
            Attrs.BROKLE_SPAN_TYPE: SpanType.GENERATION,
            Attrs.GEN_AI_PROVIDER_NAME: provider,
            Attrs.GEN_AI_OPERATION_NAME: name,
            Attrs.GEN_AI_REQUEST_MODEL: model,
        }

        if input_messages:
            attrs[Attrs.GEN_AI_INPUT_MESSAGES] = json.dumps(input_messages)

        if model_parameters:
            for key, value in model_parameters.items():
                if key == "temperature":
                    attrs[Attrs.GEN_AI_REQUEST_TEMPERATURE] = value
                elif key == "max_tokens":
                    attrs[Attrs.GEN_AI_REQUEST_MAX_TOKENS] = value
                elif key == "top_p":
                    attrs[Attrs.GEN_AI_REQUEST_TOP_P] = value
                elif key == "frequency_penalty":
                    attrs[Attrs.GEN_AI_REQUEST_FREQUENCY_PENALTY] = value
                elif key == "presence_penalty":
                    attrs[Attrs.GEN_AI_REQUEST_PRESENCE_PENALTY] = value

        if version:
            attrs[Attrs.BROKLE_VERSION] = version

        # Link prompt if provided and NOT a fallback
        if prompt and not prompt.is_fallback:
            attrs[Attrs.BROKLE_PROMPT_NAME] = prompt.name
            attrs[Attrs.BROKLE_PROMPT_VERSION] = prompt.version
            if prompt.id and prompt.id != "fallback":
                attrs[Attrs.BROKLE_PROMPT_ID] = prompt.id

        attrs.update(kwargs)
        span_name = f"{name} {model}"

        with self._tracer.start_as_current_span(
            name=span_name,
            kind=SpanKind.CLIENT,  # LLM calls are CLIENT spans
            attributes=attrs,
        ) as span:
            yield span

    @contextmanager
    def start_as_current_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None,
    ) -> Iterator[Span]:
        """
        Create a point-in-time event span.

        Events are instantaneous spans (e.g., logging, metrics).

        Args:
            name: Event name
            attributes: Event attributes
            version: Version identifier for A/B testing and experiment tracking

        Yields:
            Span instance

        Example:
            >>> with client.start_as_current_event("user-login", version="1.0") as event:
            ...     event.set_attribute("user_id", "user-123")
        """
        attrs = attributes.copy() if attributes else {}
        attrs[Attrs.BROKLE_SPAN_TYPE] = SpanType.EVENT

        if version:
            attrs[Attrs.BROKLE_VERSION] = version

        with self._tracer.start_as_current_span(
            name=name,
            kind=SpanKind.INTERNAL,
            attributes=attrs,
        ) as span:
            yield span

    def flush(self, timeout_seconds: int = 30) -> bool:
        """
        Force flush all pending spans, metrics, and logs.

        Blocks until all pending data is exported or timeout is reached.
        This is important for short-lived applications (scripts, serverless).

        Args:
            timeout_seconds: Timeout in seconds

        Returns:
            True if successful, False otherwise

        Example:
            >>> client.flush()  # Ensure all data is sent before exit
        """
        success = True
        timeout_millis = timeout_seconds * 1000

        if self._processor:
            success = self._processor.force_flush(timeout_millis) and success

        if self._meter_provider:
            success = self._meter_provider.force_flush(timeout_millis) and success

        if self._logger_provider:
            success = self._logger_provider.force_flush(timeout_millis) and success

        return success

    def shutdown(self, timeout_seconds: int = 30) -> bool:
        """
        Shutdown the client and flush all pending spans, metrics, and logs.

        Args:
            timeout_seconds: Timeout in seconds

        Returns:
            True if successful, False on error

        Example:
            >>> client.shutdown()
        """
        success = True
        timeout_millis = timeout_seconds * 1000

        if self._provider:
            try:
                self._provider.shutdown()
            except Exception:
                success = False

        if self._meter_provider:
            if not self._meter_provider.shutdown(timeout_millis):
                success = False

        if self._logger_provider:
            if not self._logger_provider.shutdown(timeout_millis):
                success = False

        return success

    def get_metrics(self) -> Optional[GenAIMetrics]:
        """
        Get the GenAI metrics instance for recording custom metrics.

        Returns:
            GenAIMetrics instance if metrics are enabled, None otherwise

        Example:
            >>> metrics = client.get_metrics()
            >>> if metrics:
            ...     metrics.record_tokens(input_tokens=100, output_tokens=50, model="gpt-4")
            ...     metrics.record_duration(duration_ms=1500, model="gpt-4")
        """
        return self._metrics

    def link_prompt(self, prompt: "Prompt") -> bool:
        """
        Link a prompt to the current active span.

        This method sets prompt attributes on the currently active span,
        allowing you to track which prompt version was used in a generation.
        Fallback prompts are NOT linked (returns False without setting attributes).

        Args:
            prompt: Prompt instance to link

        Returns:
            True if prompt was successfully linked, False otherwise
            (no active span, or prompt is a fallback)

        Example:
            >>> prompt = await client.prompts.get("greeting", label="production")
            >>> with client.start_as_current_span("my-operation") as span:
            ...     client.link_prompt(prompt)  # Links prompt to current span
            ...     # ... do work ...

            >>> # With @observe decorator
            >>> @observe()
            ... def my_function():
            ...     prompt = client.prompts.get_sync("assistant")
            ...     client.link_prompt(prompt)  # Links to decorated function's span
            ...     return call_llm(prompt)
        """
        span = trace.get_current_span()
        if not span or not span.is_recording():
            return False

        if prompt.is_fallback:
            return False

        span.set_attribute(Attrs.BROKLE_PROMPT_NAME, prompt.name)
        span.set_attribute(Attrs.BROKLE_PROMPT_VERSION, prompt.version)

        if prompt.id and prompt.id != "fallback":
            span.set_attribute(Attrs.BROKLE_PROMPT_ID, prompt.id)

        return True

    def update_current_span(
        self,
        prompt: Optional["Prompt"] = None,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> bool:
        """
        Update the current active span with additional attributes.

        This method allows updating the current span with prompt linking,
        output data, or custom metadata. Useful for dynamic updates inside
        decorated functions.

        Args:
            prompt: Prompt to link (fallback prompts are not linked)
            output: Output data to record
            metadata: Additional metadata
            **kwargs: Additional span attributes

        Returns:
            True if span was updated, False if no active span

        Example:
            >>> @observe(as_type="generation")
            ... def nested_generation():
            ...     prompt = brokle.prompts.get_sync("movie-critic")
            ...     brokle.update_current_span(prompt=prompt)
            ...     # ... do LLM call ...
            ...     brokle.update_current_span(output="LLM response")
        """
        span = trace.get_current_span()
        if not span or not span.is_recording():
            return False

        if prompt and not prompt.is_fallback:
            span.set_attribute(Attrs.BROKLE_PROMPT_NAME, prompt.name)
            span.set_attribute(Attrs.BROKLE_PROMPT_VERSION, prompt.version)
            if prompt.id and prompt.id != "fallback":
                span.set_attribute(Attrs.BROKLE_PROMPT_ID, prompt.id)

        if output is not None:
            if _is_llm_messages_format(output):
                span.set_attribute(Attrs.GEN_AI_OUTPUT_MESSAGES, json.dumps(output))
            else:
                output_str, mime_type = _serialize_with_mime(output)
                span.set_attribute(Attrs.OUTPUT_VALUE, output_str)
                span.set_attribute(Attrs.OUTPUT_MIME_TYPE, mime_type)

        if metadata:
            span.set_attribute(Attrs.METADATA, json.dumps(metadata))

        for key, value in kwargs.items():
            span.set_attribute(key, value)

        return True

    # Alias for convenience
    update_current_generation = update_current_span


def _serialize_with_mime(value: Any) -> tuple[str, str]:
    """
    Serialize value to string with MIME type detection.

    Handles edge cases: None, bytes, non-serializable objects, circular references.

    Args:
        value: Value to serialize

    Returns:
        Tuple of (serialized_string, mime_type)

    Examples:
        >>> _serialize_with_mime({"key": "value"})
        ('{"key":"value"}', 'application/json')
        >>> _serialize_with_mime("hello")
        ('hello', 'text/plain')
    """
    try:
        if value is None:
            return "null", "application/json"

        if isinstance(value, (dict, list)):
            return json.dumps(value, default=str), "application/json"

        if isinstance(value, str):
            return value, "text/plain"

        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace"), "text/plain"

        if hasattr(value, "model_dump"):
            return json.dumps(value.model_dump(exclude_none=True)), "application/json"

        if hasattr(value, "__dataclass_fields__"):
            import dataclasses
            return json.dumps(dataclasses.asdict(value)), "application/json"

        return str(value), "text/plain"

    except Exception as e:
        # Serialization failed - return error message
        return f"<serialization failed: {type(value).__name__}: {str(e)}>", "text/plain"


def _is_llm_messages_format(data: Any) -> bool:
    """Check if data is in LLM ChatML messages format."""
    return (
        isinstance(data, list)
        and len(data) > 0
        and all(isinstance(m, dict) and "role" in m for m in data)
    )
