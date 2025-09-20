"""
Brokle client implementation with OpenTelemetry integration.

This module implements the core Brokle client
but with Brokle-specific enhancements for AI platform features.
"""

import asyncio
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Union, cast
from datetime import datetime
from hashlib import sha256
from time import time_ns

import httpx
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.util._decorator import _AgnosticContextManager, _agnosticcontextmanager

from ..config import Config
from ..auth import AuthManager
from .attributes import BrokleOtelSpanAttributes
from .span import BrokleSpan, BrokleGeneration
from .exporters import BrokleSpanExporter

logger = logging.getLogger(__name__)

# Environment variable constants
BROKLE_API_KEY = "BROKLE_API_KEY"
BROKLE_PROJECT_ID = "BROKLE_PROJECT_ID"
BROKLE_HOST = "BROKLE_HOST"
BROKLE_ENVIRONMENT = "BROKLE_ENVIRONMENT"
BROKLE_OTEL_ENABLED = "BROKLE_OTEL_ENABLED"
BROKLE_DEBUG = "BROKLE_DEBUG"


class Brokle:
    """
    Main Brokle client with OTEL integration.

    This client provides:
    - Automatic OTEL span creation and management
    - Brokle-specific AI platform features (routing, caching, optimization)
    - Clean API with Brokle enhancements
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        host: Optional[str] = None,
        environment: Optional[str] = None,
        otel_enabled: Optional[bool] = None,
        debug: Optional[bool] = None,
        config: Optional[Config] = None,
        _internal_register: bool = True,
        **kwargs
    ):
        """Initialize Brokle client with OTEL integration.

        Args:
            api_key: Brokle API key. Falls back to BROKLE_API_KEY env var.
            project_id: Brokle project ID. Falls back to BROKLE_PROJECT_ID env var.
            host: Brokle API host. Falls back to BROKLE_HOST env var.
            environment: Environment name. Falls back to BROKLE_ENVIRONMENT env var.
            otel_enabled: Enable OpenTelemetry. Falls back to BROKLE_OTEL_ENABLED env var.
            debug: Enable debug logging. Falls back to BROKLE_DEBUG env var.
            config: Pre-configured Config object (overrides env vars and parameters).
            _internal_register: INTERNAL USE ONLY - controls singleton registration.
            **kwargs: Additional configuration parameters.
        """
        # Track initialization state for resource cleanup
        self._fully_initialized = False

        try:
            if config is not None:
                # Use provided config object directly
                self.config = config
            else:
                # Create config with parameter and environment variable fallback
                api_key = api_key or os.environ.get(BROKLE_API_KEY)
                project_id = project_id or os.environ.get(BROKLE_PROJECT_ID)
                host = host or os.environ.get(BROKLE_HOST, "http://localhost:8000")
                environment = environment or os.environ.get(BROKLE_ENVIRONMENT, "default")

                # Handle boolean environment variables
                if otel_enabled is None:
                    otel_env = os.environ.get(BROKLE_OTEL_ENABLED)
                    otel_enabled = otel_env.lower() in ('true', '1', 'yes') if otel_env else True

                if debug is None:
                    debug_env = os.environ.get(BROKLE_DEBUG)
                    debug = debug_env.lower() in ('true', '1', 'yes') if debug_env else False

                # Validate required parameters
                if api_key is None:
                    logger.warning(
                        "Authentication error: Brokle client initialized without api_key. "
                        "Provide an api_key parameter or set BROKLE_API_KEY environment variable."
                    )
                    # Continue with disabled tracing similar
                    api_key = "ak_fake"  # Must start with "ak_" to pass validation
                    otel_enabled = False

                if project_id is None:
                    logger.warning(
                        "Configuration error: Brokle client initialized without project_id. "
                        "Provide a project_id parameter or set BROKLE_PROJECT_ID environment variable."
                    )
                    project_id = "fake"
                    otel_enabled = False

                # Create config with all parameters
                config_params = {
                    'api_key': api_key,
                    'project_id': project_id,
                    'host': host,
                    'environment': environment,
                    'otel_enabled': otel_enabled,
                    'debug': debug,
                    **kwargs
                }

                self.config = Config(**config_params)

            # Initialize auth manager
            self.auth_manager = AuthManager(self.config)

            # Initialize HTTP client
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
            )

            # Initialize OTEL components
            self._tracer_provider: Optional[TracerProvider] = None
            self._tracer: Optional[trace.Tracer] = None
            self._span_exporter: Optional[BrokleSpanExporter] = None
            self._initialized = False
            self._lock = threading.Lock()

            # Initialize OTEL if enabled
            if self.config.otel_enabled:
                self._initialize_otel()

            # Mark as fully initialized before registration
            self._fully_initialized = True

            # Auto-register in singleton cache (Langfuse pattern) - only if requested
            if _internal_register and self.config.api_key != "ak_fake":
                with _instances_lock:
                    if self.config.api_key:
                        _instances[self.config.api_key] = self
                    elif "__default__" not in _instances:
                        _instances["__default__"] = self

        except Exception:
            # CRITICAL: Force cleanup of partial resources
            self._cleanup_resources(force=True)
            raise

    def _initialize_otel(self) -> None:
        """Initialize OpenTelemetry components."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            try:
                # Create tracer provider
                self._tracer_provider = TracerProvider()

                # Create custom Brokle span exporter
                self._span_exporter = BrokleSpanExporter(
                    endpoint=self.config.host,
                    api_key=self.config.api_key,
                    project_id=self.config.project_id
                )

                # Add span processor
                span_processor = BatchSpanProcessor(
                    self._span_exporter,
                    max_queue_size=2048,
                    max_export_batch_size=512,
                    export_timeout_millis=30000,
                )
                self._tracer_provider.add_span_processor(span_processor)

                # Set global tracer provider
                trace.set_tracer_provider(self._tracer_provider)

                # Get tracer
                self._tracer = trace.get_tracer(
                    "brokle-python-sdk",
                    version="0.1.0",
                )

                self._initialized = True
                logger.info("Brokle OTEL integration initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize Brokle OTEL integration: {e}")
                self._initialized = False

    @property
    def tracer(self) -> trace.Tracer:
        """Get the OTEL tracer."""
        if not self._tracer:
            self._initialize_otel()
        return self._tracer

    def span(
        self,
        name: str,
        *,
        trace_id: Optional[str] = None,
        parent_observation_id: Optional[str] = None,
        level: str = "DEFAULT",
        status_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> BrokleSpan:
        """Create a new Brokle span."""
        if not self._initialized:
            self._initialize_otel()

        return BrokleSpan(
            client=self,
            name=name,
            trace_id=trace_id,
            parent_observation_id=parent_observation_id,
            level=level,
            status_message=status_message,
            metadata=metadata or {},
            tags=tags or [],
            **kwargs
        )

    def generation(
        self,
        name: str,
        *,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        model_parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BrokleGeneration:
        """Create a new Brokle generation span for LLM calls."""
        if not self._initialized:
            self._initialize_otel()

        return BrokleGeneration(
            client=self,
            name=name,
            model=model,
            provider=provider,
            model_parameters=model_parameters or {},
            **kwargs
        )

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Make HTTP request to Brokle API."""
        url = f"{self.config.host.rstrip('/')}{endpoint}"

        # Get auth headers
        headers = await self.auth_manager.get_headers()
        headers.update({
            "Content-Type": "application/json",
            "User-Agent": f"brokle-python-sdk/0.1.0",
        })

        try:
            response = await self._http_client.request(
                method=method,
                url=url,
                json=data,
                headers=headers,
                **kwargs
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise

    def flush(self) -> None:
        """Flush all pending spans to Brokle backend."""
        if self._tracer_provider and hasattr(self._tracer_provider, 'force_flush'):
            self._tracer_provider.force_flush(timeout_millis=30000)

    async def _async_cleanup(self) -> None:
        """Idempotent async cleanup - safe to call multiple times."""
        # Close HTTP client idempotently
        if hasattr(self, '_http_client') and self._http_client is not None:
            try:
                await self._http_client.aclose()
            except Exception:
                pass  # Defensive - already closed or error
            finally:
                self._http_client = None

        # Shutdown OTEL resources defensively
        if hasattr(self, '_tracer_provider') and self._tracer_provider is not None:
            try:
                self._tracer_provider.force_flush()
                if hasattr(self._tracer_provider, 'shutdown'):
                    self._tracer_provider.shutdown()
            except Exception:
                pass  # Some providers may not have shutdown
            finally:
                self._tracer_provider = None

        if hasattr(self, '_span_exporter') and self._span_exporter is not None:
            try:
                if hasattr(self._span_exporter, 'shutdown'):
                    self._span_exporter.shutdown()
            except Exception:
                pass  # Defensive guard
            finally:
                self._span_exporter = None

    def _cleanup_resources(self, force: bool = False) -> None:
        """Cleanup resources with optional force for partial initialization."""
        if not force and not getattr(self, '_fully_initialized', False):
            return  # Skip cleanup unless forced

        if not (hasattr(self, '_http_client') and self._http_client):
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop - safe to use asyncio.run
            asyncio.run(self._async_cleanup())
        else:
            # Inside running loop - schedule as task
            loop.create_task(self._async_cleanup())

    async def shutdown(self) -> None:
        """Shutdown the client and cleanup resources."""
        await self._async_cleanup()


# Resource management for multiple clients
_instances: Dict[str, Brokle] = {}
_instances_lock = threading.Lock()


def get_client(*, api_key: Optional[str] = None, **kwargs) -> Brokle:
    """Get or create a Brokle client instance.

    Returns an existing Brokle client or creates a new one if none exists. In multi-project setups,
    providing an api_key is required. Multi-project support.

    Behavior:
    - Single project: Returns existing client or creates new one
    - Multi-project: Requires api_key to return specific client
    - No api_key in multi-project: Returns disabled client to prevent data leakage

    The function uses a singleton pattern per api_key to conserve resources and maintain state.

    Args:
        api_key (Optional[str]): Project identifier
            - With key: Returns client for that project
            - Without key: Returns single client or disabled client if multiple exist
        **kwargs: Additional configuration parameters

    Returns:
        Brokle: Client instance in one of three states:
            1. Client for specified api_key
            2. Default client for single-project setup
            3. Disabled client when multiple projects exist without key

    Security:
        Disables tracing when multiple projects exist without explicit key to prevent
        cross-project data leakage.

    Example:
        ```python
        # Recommended patterns:

        # 1. Direct instantiation (explicit configuration) - auto-registers for @observe
        client = Brokle(api_key="ak_project_a", project_id="proj_123")

        # 2. Environment variables (requires BROKLE_API_KEY, etc. set)
        client = get_client()  # Creates or returns existing client

        # Multi-project usage:
        client_a = get_client(api_key="ak_project_a")  # Returns project A's client
        client_b = get_client(api_key="ak_project_b")  # Returns project B's client
        ```
    """
    # Resolve API key
    if not api_key:
        api_key = os.environ.get(BROKLE_API_KEY)

    cache_key = api_key if api_key else "__default__"

    # First check: fast path
    with _instances_lock:
        if cache_key in _instances:
            return _instances[cache_key]

        if not api_key and len(_instances) == 1:
            return next(iter(_instances.values()))

    # Create instance outside lock to avoid deadlock
    new_instance = None
    try:
        new_instance = Brokle(api_key=api_key, _internal_register=False, **kwargs)

        # Second check with race protection
        with _instances_lock:
            if cache_key in _instances:
                # Someone beat us - cleanup our instance safely
                if new_instance._fully_initialized:
                    new_instance._cleanup_resources()
                return _instances[cache_key]

            # We won - register our instance
            _instances[cache_key] = new_instance

        return new_instance

    except Exception:
        # Cleanup failed instance if it was created
        if new_instance:
            new_instance._cleanup_resources(force=True)
        raise
