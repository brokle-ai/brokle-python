"""
Main Brokle OpenTelemetry client.

Provides high-level API for creating traces, spans, and LLM observations
using OpenTelemetry as the underlying telemetry framework.
"""

import atexit
import json
from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Iterator
from uuid import UUID

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider, Span
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased, ALWAYS_ON
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Tracer, SpanKind, Status, StatusCode

from .config import BrokleConfig
from .exporter import create_exporter_for_config
from .processor import BrokleSpanProcessor
from .types import Attrs, ObservationType, LLMProvider, OperationType


# Global singleton instance
_global_client: Optional["Brokle"] = None


class Brokle:
    """
    Main Brokle client for OpenTelemetry-based observability.

    This client initializes OpenTelemetry with Brokle-specific configuration
    and provides high-level methods for creating traces and observations.

    Example:
        >>> from brokle import Brokle
        >>> client = Brokle(api_key="bk_your_secret")
        >>> with client.start_as_current_span("my-operation") as span:
        ...     span.set_attribute("output", "Hello, world!")
        >>> client.flush()
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:8080",
        environment: str = "default",
        debug: bool = False,
        tracing_enabled: bool = True,
        release: Optional[str] = None,
        sample_rate: float = 1.0,
        mask: Optional[callable] = None,
        flush_at: int = 100,
        flush_interval: float = 5.0,
        timeout: int = 30,
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
            release: Release version/hash for grouping analytics
            sample_rate: Sampling rate for traces (0.0 to 1.0)
            mask: Optional function to mask sensitive data
            flush_at: Maximum batch size before flush (1-1000)
            flush_interval: Maximum delay in seconds before flush (0.1-60.0)
            timeout: HTTP timeout in seconds
            **kwargs: Additional configuration options

        Raises:
            ValueError: If configuration is invalid
        """
        # Create configuration
        self.config = BrokleConfig(
            api_key=api_key or "",  # Will be validated by BrokleConfig
            base_url=base_url,
            environment=environment,
            debug=debug,
            tracing_enabled=tracing_enabled,
            release=release,
            sample_rate=sample_rate,
            mask=mask,
            flush_at=flush_at,
            flush_interval=flush_interval,
            timeout=timeout,
            **kwargs,
        )

        # If tracing is disabled, we still initialize but with no-op behavior
        if not self.config.tracing_enabled:
            self._tracer = trace.get_tracer(__name__)
            self._provider = None
            self._processor = None
            return

        # Create Resource with project-level attributes
        # These attributes are automatically included in all spans
        resource = Resource.create(
            {
                Attrs.BROKLE_PROJECT_ID: self._extract_project_id(api_key),
                Attrs.BROKLE_ENVIRONMENT: environment,
                "service.name": "brokle-sdk",
                "service.version": self._get_sdk_version(),
            }
        )

        # Add release if provided
        if release:
            resource = resource.merge(
                Resource.create({Attrs.BROKLE_RELEASE: release})
            )

        # Create sampler based on sample_rate
        # Uses OpenTelemetry's TraceIdRatioBased sampler for trace-level sampling
        # This ensures entire traces are sampled together (not individual spans)
        if self.config.sample_rate < 1.0:
            sampler = TraceIdRatioBased(self.config.sample_rate)
        else:
            sampler = ALWAYS_ON  # 100% sampling (default)

        # Create TracerProvider with Resource and Sampler
        self._provider = TracerProvider(
            resource=resource,
            sampler=sampler,  # Trace-level sampling (deterministic by trace_id)
        )

        # Create exporter based on configuration
        exporter = create_exporter_for_config(self.config)

        # Create Brokle span processor with batching
        self._processor = BrokleSpanProcessor(
            span_exporter=exporter,
            config=self.config,
        )

        # Add processor to provider
        self._provider.add_span_processor(self._processor)

        # Get tracer from provider
        self._tracer = self._provider.get_tracer(
            instrumenting_module_name="brokle",
            instrumenting_library_version=self._get_sdk_version(),
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
        # Hash or extract project ID from API key
        # For now, use a portion of the key as identifier
        return api_key[:20]  # Placeholder - backend determines actual project

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

    @contextmanager
    def start_as_current_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Iterator[Span]:
        """
        Create a span using context manager (OpenTelemetry standard pattern).

        This is the recommended way to create spans as it automatically handles
        span lifecycle and context propagation.

        Args:
            name: Span name
            kind: Span kind (INTERNAL, CLIENT, SERVER, PRODUCER, CONSUMER)
            attributes: Initial span attributes
            **kwargs: Additional arguments passed to tracer.start_as_current_span()

        Yields:
            Span instance

        Example:
            >>> with client.start_as_current_span("my-operation") as span:
            ...     span.set_attribute("key", "value")
            ...     # Span automatically ends when context exits
        """
        # Build attributes
        attrs = attributes or {}

        # Set observation type for Brokle backend
        if Attrs.BROKLE_OBSERVATION_TYPE not in attrs:
            attrs[Attrs.BROKLE_OBSERVATION_TYPE] = ObservationType.SPAN

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
            **kwargs: Additional span attributes

        Yields:
            Span instance

        Example:
            >>> with client.start_as_current_generation(
            ...     name="chat",
            ...     model="gpt-4",
            ...     provider="openai",
            ...     input_messages=[{"role": "user", "content": "Hello"}],
            ... ) as gen:
            ...     # Make LLM call
            ...     gen.set_attribute(Attrs.GEN_AI_OUTPUT_MESSAGES, [...])
        """
        # Build OTEL GenAI attributes
        attrs = {
            Attrs.BROKLE_OBSERVATION_TYPE: ObservationType.GENERATION,
            Attrs.GEN_AI_PROVIDER_NAME: provider,
            Attrs.GEN_AI_OPERATION_NAME: name,
            Attrs.GEN_AI_REQUEST_MODEL: model,
        }

        # Add input messages if provided
        if input_messages:
            attrs[Attrs.GEN_AI_INPUT_MESSAGES] = json.dumps(input_messages)

        # Add model parameters
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

        # Merge additional kwargs
        attrs.update(kwargs)

        # Span name follows OTEL pattern: "{operation} {model}"
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
    ) -> Iterator[Span]:
        """
        Create a point-in-time event span.

        Events are instantaneous observations (e.g., logging, metrics).

        Args:
            name: Event name
            attributes: Event attributes

        Yields:
            Span instance

        Example:
            >>> with client.start_as_current_event("user-login") as event:
            ...     event.set_attribute("user_id", "user-123")
        """
        attrs = attributes or {}
        attrs[Attrs.BROKLE_OBSERVATION_TYPE] = ObservationType.EVENT

        with self._tracer.start_as_current_span(
            name=name,
            kind=SpanKind.INTERNAL,
            attributes=attrs,
        ) as span:
            yield span

    def flush(self, timeout_seconds: int = 30) -> bool:
        """
        Force flush all pending spans.

        Blocks until all pending spans are exported or timeout is reached.
        This is important for short-lived applications (scripts, serverless).

        Args:
            timeout_seconds: Timeout in seconds

        Returns:
            True if successful, False otherwise

        Example:
            >>> client.flush()  # Ensure all data is sent before exit
        """
        if not self._processor:
            return True

        timeout_millis = timeout_seconds * 1000
        return self._processor.force_flush(timeout_millis)

    def shutdown(self, timeout_seconds: int = 30) -> bool:
        """
        Shutdown the client and flush all pending spans.

        Args:
            timeout_seconds: Timeout in seconds

        Returns:
            True if successful

        Example:
            >>> client.shutdown()
        """
        if not self._provider:
            return True

        timeout_millis = timeout_seconds * 1000
        return self._provider.shutdown()

    def close(self):
        """
        Close the client (alias for shutdown).

        Example:
            >>> with Brokle(api_key="...") as client:
            ...     # Use client
            ...     pass  # Automatically closed
        """
        self.shutdown()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        return f"Brokle(environment='{self.config.environment}', tracing_enabled={self.config.tracing_enabled})"


def get_client(**overrides) -> Brokle:
    """
    Get or create global singleton Brokle client.

    Configuration is read from environment variables on first call.
    Subsequent calls return the same instance.

    Args:
        **overrides: Override specific configuration values

    Returns:
        Singleton Brokle instance

    Raises:
        ValueError: If BROKLE_API_KEY environment variable is missing

    Example:
        >>> from brokle import get_client
        >>> client = get_client()  # Reads from BROKLE_* env vars
        >>> # All calls return same instance
        >>> client2 = get_client()
        >>> assert client is client2
    """
    global _global_client

    if _global_client is None:
        # Create config from environment variables
        config = BrokleConfig.from_env(**overrides)

        # Create client with config parameters
        _global_client = Brokle(
            api_key=config.api_key,
            base_url=config.base_url,
            environment=config.environment,
            debug=config.debug,
            tracing_enabled=config.tracing_enabled,
            release=config.release,
            sample_rate=config.sample_rate,
            mask=config.mask,
            flush_at=config.flush_at,
            flush_interval=config.flush_interval,
            timeout=config.timeout,
            use_protobuf=config.use_protobuf,
            compression=config.compression,
            cache_enabled=config.cache_enabled,
            routing_enabled=config.routing_enabled,
        )

    return _global_client


def reset_client():
    """
    Reset global singleton client.

    Useful for testing. Should not be used in production code.

    Example:
        >>> reset_client()
        >>> client = get_client()  # Creates new instance
    """
    global _global_client
    if _global_client:
        _global_client.close()
    _global_client = None
