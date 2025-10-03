"""
Main client for Brokle SDK.

This module provides the main client interface with OpenAI-compatible design
and clean resource management.

Key Features:
- Clean OpenAI-compatible interface
- Proper resource management with context managers
- Sync and async client variants
- Brokle extensions (routing, caching, tags)
- Integrated background task processing
"""

from typing import Any, Callable, Dict, List, Optional

import httpx

from ._task_manager.processor import BackgroundProcessor, get_background_processor
from .config import Config
from .exceptions import NetworkError
from .http.base import HTTPBase


class Brokle(HTTPBase):
    """
    Sync Brokle client with OpenAI-compatible interface.

    Usage:
        with Brokle(api_key="bk_...", host="http://localhost:8080") as client:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello!"}],
                routing_strategy="cost_optimized"  # Brokle extension
            )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        environment: Optional[str] = None,
        timeout: Optional[float] = None,
        background_processor: Optional[BackgroundProcessor] = None,
        **kwargs,
    ):
        """
        Initialize sync Brokle client.

        Args:
            api_key: Brokle API key
            host: Brokle host URL
            environment: Environment name
            timeout: Request timeout in seconds
            background_processor: Optional background processor for telemetry (will create default if None)
            **kwargs: Additional configuration
        """
        super().__init__(
            api_key=api_key,
            host=host,
            environment=environment,
            timeout=timeout,
            **kwargs,
        )

        # Initialize HTTP client
        self._client: Optional[httpx.Client] = None

        # Handle disabled state
        if getattr(self, '_disabled', False):
            self._background_processor = None
            self._owns_processor = False
        else:
            # Initialize background processor for telemetry
            if background_processor is not None:
                self._background_processor = background_processor
                self._owns_processor = False  # We don't own it, don't shut it down
            else:
                # Create default processor using our config
                self._background_processor = get_background_processor(config=self.config)
                self._owns_processor = True  # We created it, we should shut it down

        # Initialize resources (always, even when disabled)
        from .resources.chat import ChatResource
        from .resources.embeddings import EmbeddingsResource
        from .resources.models import ModelsResource

        self.chat = ChatResource(self)
        self.embeddings = EmbeddingsResource(self)
        self.models = ModelsResource(self)

    def span(self, name: str, **kwargs):
        """Create a span for observability."""
        from .observability.spans import create_span

        return create_span(name=name, **kwargs)

    @property
    def is_disabled(self) -> bool:
        """Check if client is operating in disabled mode."""
        return getattr(self, '_disabled', False)

    def submit_telemetry(
        self, data: Dict[str, Any], event_type: str = "event_create"
    ) -> None:
        """
        Submit telemetry data for background processing.

        Args:
            data: Telemetry data to submit
            event_type: Event type (defaults to "event_create")
        """
        if self.is_disabled or not self._background_processor:
            return  # Skip telemetry when disabled
        self._background_processor.submit_telemetry(data, event_type=event_type)

    def submit_batch_event(self, event_type: str, payload: Dict[str, Any]) -> str:
        """
        Submit a batch event with proper event envelope.

        This is the preferred method for submitting structured telemetry events.

        Args:
            event_type: Type of event (trace_create, observation_create, etc.)
            payload: Event payload data

        Returns:
            Event ID (ULID) for tracking

        Example:
            >>> event_id = client.submit_batch_event(
            ...     "trace_create",
            ...     {"name": "my-trace", "user_id": "123"}
            ... )
        """
        if self.is_disabled or not self._background_processor:
            return ""  # Return empty ID when disabled

        from .types.telemetry import TelemetryEvent
        from ._utils.ulid import generate_event_id
        import time

        event = TelemetryEvent(
            event_id=generate_event_id(),
            event_type=event_type,
            payload=payload,
            timestamp=int(time.time())
        )

        self._background_processor.submit_batch_event(event)
        return event.event_id

    def submit_analytics(self, data: Dict[str, Any]) -> None:
        """
        Submit analytics data for background processing.

        Args:
            data: Analytics data to submit
        """
        if self.is_disabled or not self._background_processor:
            return  # Skip analytics when disabled
        self._background_processor.submit_analytics(data)

    def submit_evaluation(self, data: Dict[str, Any]) -> None:
        """
        Submit evaluation data for background processing.

        Args:
            data: Evaluation data to submit
        """
        if self.is_disabled or not self._background_processor:
            return  # Skip evaluation when disabled
        self._background_processor.submit_evaluation(data)

    def get_processor_metrics(self) -> Dict[str, Any]:
        """
        Get background processor metrics.

        Returns:
            Dictionary containing processor metrics
        """
        if self.is_disabled or not self._background_processor:
            return {}  # Return empty metrics when disabled
        return self._background_processor.get_metrics()

    def is_processor_healthy(self) -> bool:
        """
        Check if background processor is healthy.

        Returns:
            True if processor is healthy
        """
        if self.is_disabled or not self._background_processor:
            return False  # Processor not healthy when disabled
        return self._background_processor.is_healthy()

    def flush_processor(self, timeout: Optional[float] = None) -> bool:
        """
        Flush pending processor items and wait for completion.

        Args:
            timeout: Maximum time to wait (None = wait indefinitely)

        Returns:
            True if all items processed, False if timeout reached
        """
        if self.is_disabled or not self._background_processor:
            return True  # Nothing to flush when disabled
        return self._background_processor.flush(timeout=timeout)

    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                timeout=httpx.Timeout(self.config.timeout),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            )
        return self._client

    def request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make HTTP request to Brokle backend with retry logic.

        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Request kwargs

        Returns:
            Response data

        Raises:
            NetworkError: For connection errors
        """
        # Return empty dict if client is disabled (graceful degradation)
        if self.is_disabled:
            return {}

        import time

        from ._utils.retry import is_retryable_error, retry_with_backoff

        start_time = time.time()
        url = self._prepare_url(endpoint)
        kwargs = self._prepare_request_kwargs(**kwargs)

        @retry_with_backoff(
            max_retries=self.config.max_retries, base_delay=1.0, max_delay=30.0
        )
        def _make_request():
            client = self._get_client()
            response = client.request(method, url, **kwargs)
            result = self._process_response(response)
            return result, response.status_code

        try:
            result, status_code = _make_request()

            # Submit telemetry data in background
            telemetry_data = {
                "method": method,
                "endpoint": endpoint,
                "status_code": status_code,  # Real status code from response
                "latency_ms": int((time.time() - start_time) * 1000),
                "success": True,
                "timestamp": time.time(),
                "environment": self.config.environment,
            }
            self.submit_telemetry(telemetry_data)

            return result
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError) as e:
            # Submit error telemetry
            telemetry_data = {
                "method": method,
                "endpoint": endpoint,
                "status_code": (
                    getattr(e.response, "status_code", 0)
                    if hasattr(e, "response")
                    else 0
                ),
                "latency_ms": int((time.time() - start_time) * 1000),
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": time.time(),
                "environment": self.config.environment,
            }
            self.submit_telemetry(telemetry_data)

            # Re-raise with appropriate exception type
            if isinstance(e, httpx.ConnectError):
                raise NetworkError(f"Failed to connect to Brokle backend: {e}")
            elif isinstance(e, httpx.TimeoutException):
                raise NetworkError(f"Request timeout: {e}")
            else:
                raise NetworkError(f"HTTP error: {e}")

    def close(self) -> None:
        """Close HTTP client and cleanup resources."""
        # Flush processor before closing (give it 5 seconds)
        if hasattr(self, "_background_processor"):
            try:
                self._background_processor.flush(timeout=5.0)
            except Exception:
                pass  # Don't let processor errors prevent cleanup

            # Shutdown processor if we own it
            if hasattr(self, "_owns_processor") and self._owns_processor:
                try:
                    self._background_processor.shutdown()
                except Exception:
                    pass  # Don't let processor errors prevent cleanup

        # Close HTTP client
        if hasattr(self, "_client") and self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()

    def __del__(self):
        """Cleanup on deletion (fallback)."""
        self.close()


class AsyncBrokle(HTTPBase):
    """
    Async Brokle client with OpenAI-compatible interface.

    Usage:
        client = AsyncBrokle(api_key="bk_...")
        try:
            response = await client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello!"}],
                routing_strategy="cost_optimized"
            )
        finally:
            await client.close()
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        environment: Optional[str] = None,
        timeout: Optional[float] = None,
        background_processor: Optional[BackgroundProcessor] = None,
        **kwargs,
    ):
        """
        Initialize async Brokle client.

        Args:
            api_key: Brokle API key
            host: Brokle host URL
            environment: Environment name
            timeout: Request timeout in seconds
            background_processor: Optional background processor for telemetry (will create default if None)
            **kwargs: Additional configuration
        """
        super().__init__(
            api_key=api_key,
            host=host,
            environment=environment,
            timeout=timeout,
            **kwargs,
        )

        # Handle disabled state
        if getattr(self, '_disabled', False):
            self._client = None
            self._background_processor = None
            self._owns_processor = False
        else:
            # Initialize persistent HTTP client (performance optimization)
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            )

            # Initialize background processor for telemetry
            if background_processor is not None:
                self._background_processor = background_processor
                self._owns_processor = False  # We don't own it, don't shut it down
            else:
                # Create default processor using our config
                self._background_processor = get_background_processor(config=self.config)
                self._owns_processor = True  # We created it, we should shut it down

        # Initialize async resources (always, even when disabled)
        from .resources.chat import AsyncChatResource
        from .resources.embeddings import AsyncEmbeddingsResource
        from .resources.models import AsyncModelsResource

        self.chat = AsyncChatResource(self)
        self.embeddings = AsyncEmbeddingsResource(self)
        self.models = AsyncModelsResource(self)

    @property
    def is_disabled(self) -> bool:
        """Check if client is operating in disabled mode."""
        return getattr(self, '_disabled', False)

    def submit_telemetry(
        self, data: Dict[str, Any], event_type: str = "event_create"
    ) -> None:
        """
        Submit telemetry data for background processing.

        Args:
            data: Telemetry data to submit
            event_type: Event type (defaults to "event_create")
        """
        if self.is_disabled or not self._background_processor:
            return  # Skip telemetry when disabled
        self._background_processor.submit_telemetry(data, event_type=event_type)

    def submit_batch_event(self, event_type: str, payload: Dict[str, Any]) -> str:
        """
        Submit a batch event with proper event envelope.

        This is the preferred method for submitting structured telemetry events.

        Args:
            event_type: Type of event (trace_create, observation_create, etc.)
            payload: Event payload data

        Returns:
            Event ID (ULID) for tracking

        Example:
            >>> event_id = client.submit_batch_event(
            ...     "trace_create",
            ...     {"name": "my-trace", "user_id": "123"}
            ... )
        """
        if self.is_disabled or not self._background_processor:
            return ""  # Return empty ID when disabled

        from .types.telemetry import TelemetryEvent
        from ._utils.ulid import generate_event_id
        import time

        event = TelemetryEvent(
            event_id=generate_event_id(),
            event_type=event_type,
            payload=payload,
            timestamp=int(time.time())
        )

        self._background_processor.submit_batch_event(event)
        return event.event_id

    def submit_analytics(self, data: Dict[str, Any]) -> None:
        """
        Submit analytics data for background processing.

        Args:
            data: Analytics data to submit
        """
        if self.is_disabled or not self._background_processor:
            return  # Skip analytics when disabled
        self._background_processor.submit_analytics(data)

    def submit_evaluation(self, data: Dict[str, Any]) -> None:
        """
        Submit evaluation data for background processing.

        Args:
            data: Evaluation data to submit
        """
        if self.is_disabled or not self._background_processor:
            return  # Skip evaluation when disabled
        self._background_processor.submit_evaluation(data)

    def get_processor_metrics(self) -> Dict[str, Any]:
        """
        Get background processor metrics.

        Returns:
            Dictionary containing processor metrics
        """
        if self.is_disabled or not self._background_processor:
            return {}  # Return empty metrics when disabled
        return self._background_processor.get_metrics()

    def is_processor_healthy(self) -> bool:
        """
        Check if background processor is healthy.

        Returns:
            True if processor is healthy
        """
        if self.is_disabled or not self._background_processor:
            return False  # Processor not healthy when disabled
        return self._background_processor.is_healthy()

    def flush_processor(self, timeout: Optional[float] = None) -> bool:
        """
        Flush pending processor items and wait for completion.

        Args:
            timeout: Maximum time to wait (None = wait indefinitely)

        Returns:
            True if all items processed, False if timeout reached
        """
        if self.is_disabled or not self._background_processor:
            return True  # Nothing to flush when disabled
        return self._background_processor.flush(timeout=timeout)

    async def request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make async HTTP request to Brokle backend with retry logic.

        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Request kwargs

        Returns:
            Response data

        Raises:
            NetworkError: For connection errors
        """
        # Return empty dict if client is disabled (graceful degradation)
        if self.is_disabled:
            return {}

        import time

        from ._utils.retry import async_retry_with_backoff

        start_time = time.time()
        url = self._prepare_url(endpoint)
        kwargs = self._prepare_request_kwargs(**kwargs)

        @async_retry_with_backoff(
            max_retries=self.config.max_retries, base_delay=1.0, max_delay=30.0
        )
        async def _make_request():
            response = await self._client.request(method, url, **kwargs)
            result = self._process_response(response)
            return result, response.status_code

        try:
            result, status_code = await _make_request()

            # Submit telemetry data in background
            telemetry_data = {
                "method": method,
                "endpoint": endpoint,
                "status_code": status_code,  # Real status code from response
                "latency_ms": int((time.time() - start_time) * 1000),
                "success": True,
                "timestamp": time.time(),
                "environment": self.config.environment,
            }
            self.submit_telemetry(telemetry_data)

            return result
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError) as e:
            # Submit error telemetry
            telemetry_data = {
                "method": method,
                "endpoint": endpoint,
                "status_code": (
                    getattr(e.response, "status_code", 0)
                    if hasattr(e, "response")
                    else 0
                ),
                "latency_ms": int((time.time() - start_time) * 1000),
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": time.time(),
                "environment": self.config.environment,
            }
            self.submit_telemetry(telemetry_data)

            # Re-raise with appropriate exception type
            if isinstance(e, httpx.ConnectError):
                raise NetworkError(f"Failed to connect to Brokle backend: {e}")
            elif isinstance(e, httpx.TimeoutException):
                raise NetworkError(f"Request timeout: {e}")
            else:
                raise NetworkError(f"HTTP error: {e}")

    async def close(self) -> None:
        """Close async HTTP client and cleanup resources."""
        # Flush processor before closing (give it 5 seconds)
        if hasattr(self, "_background_processor"):
            try:
                self._background_processor.flush(timeout=5.0)
            except Exception:
                pass  # Don't let processor errors prevent cleanup

            # Shutdown processor if we own it
            if hasattr(self, "_owns_processor") and self._owns_processor:
                try:
                    self._background_processor.shutdown()
                except Exception:
                    pass  # Don't let processor errors prevent cleanup

        # Close HTTP client
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.close()


# Global singleton for clean architecture
_client_singleton: Optional[Brokle] = None


def get_client(background_processor: Optional[BackgroundProcessor] = None) -> Brokle:
    """
    Get or create a singleton Brokle client instance from environment variables.

    This is the clean API for Pattern 1/2/3 integration:
    - Pattern 1 (Wrappers): Use this for observability context
    - Pattern 2 (Decorator): Use this for telemetry
    - Pattern 3 (Native): Use Brokle() for explicit config, get_client() for env config

    Configuration is read from environment variables:
    - BROKLE_API_KEY
    - BROKLE_HOST
    - BROKLE_ENVIRONMENT
    - BROKLE_OTEL_ENABLED
    - etc.

    Args:
        background_processor: Optional background processor for telemetry (only used when creating new singleton)

    Returns:
        Singleton Brokle client instance

    Example:
        ```python
        # For explicit configuration, use Brokle() directly
        client = Brokle(api_key="bk_your_secret")

        # For environment-based configuration, use get_client()
        client = get_client()  # Reads from BROKLE_* env vars

        # Sharing a background processor across multiple clients
        processor = get_background_processor(config=config)
        client = get_client(background_processor=processor)
        ```
    """
    global _client_singleton

    if _client_singleton is None:
        # Create singleton from environment variables
        _client_singleton = Brokle(background_processor=background_processor)

    return _client_singleton


# Export public API
__all__ = ["Brokle", "AsyncBrokle", "get_client"]
