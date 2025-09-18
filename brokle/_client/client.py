"""
Brokle client implementation with OpenTelemetry integration.

This module implements the core Brokle client
but with Brokle-specific enhancements for AI platform features.
"""

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

from ..config import Config, get_config
from ..auth import AuthManager
from .attributes import BrokleOtelSpanAttributes
from .span import BrokleSpan, BrokleGeneration
from .exporters import BrokleSpanExporter

logger = logging.getLogger(__name__)


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
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        config: Optional[Config] = None,
        **kwargs
    ):
        """Initialize Brokle client with OTEL integration."""
        # Use provided config or get default
        self.config = config or get_config()

        # Override config with provided parameters
        if public_key:
            self.config.public_key = public_key
        if secret_key:
            self.config.secret_key = secret_key
        if host:
            self.config.host = host

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
                    api_key=self.config.secret_key,
                    public_key=self.config.public_key
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

    async def shutdown(self) -> None:
        """Shutdown the client and cleanup resources."""
        if self._tracer_provider and hasattr(self._tracer_provider, 'shutdown'):
            self._tracer_provider.shutdown()

        if self._http_client:
            await self._http_client.aclose()


# Global client instance
_client: Optional[Brokle] = None
_client_lock = threading.Lock()


def get_client(
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    host: Optional[str] = None,
    **kwargs
) -> Brokle:
    """
    Get or create a singleton Brokle client.

    This function returns a singleton instance of the Brokle client,
    creating it if it doesn't exist.
    """
    global _client

    if _client is None:
        with _client_lock:
            if _client is None:
                _client = Brokle(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host,
                    **kwargs
                )

    return _client