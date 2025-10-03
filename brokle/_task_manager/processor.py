"""
Background processing for Brokle SDK.

Handles batch submission of telemetry events to the unified /v1/telemetry/batch endpoint.
"""

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Optional

import backoff
import httpx

from ..config import Config
from ..types.telemetry import (
    TelemetryBatchRequest,
    TelemetryBatchResponse,
    TelemetryEvent,
    TelemetryEventType,
    DeduplicationConfig,
)
from .._utils.ulid import generate_event_id

logger = logging.getLogger(__name__)


class BackgroundProcessor:
    """Background processor for telemetry and other async tasks."""

    def __init__(self, config: Config):
        self.config = config
        self._queue: Queue = Queue(maxsize=10000)
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="brokle-bg"
        )
        self._shutdown = False
        self._worker_thread: Optional[threading.Thread] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._lock = threading.Lock()

        # Metrics tracking
        self._metrics = {
            "items_processed": 0,
            "items_failed": 0,
            "batches_processed": 0,
            "start_time": time.time(),
            "last_error": None,
            "last_error_time": None,
        }

        # Start worker thread
        self._start_worker()

    def _start_worker(self) -> None:
        """Start background worker thread."""
        if self._worker_thread and self._worker_thread.is_alive():
            return

        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="brokle-bg-worker", daemon=True
        )
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        """Background worker loop."""
        logger.info("Background processor started")

        while not self._shutdown:
            try:
                # Process items in batches using batch_max_size configuration
                batch = []
                batch_size = self.config.batch_max_size

                # Collect batch items
                for _ in range(batch_size):
                    try:
                        item = self._queue.get(timeout=1.0)
                        batch.append(item)
                    except Empty:
                        break

                if batch:
                    self._process_batch(batch)

                    # Mark items as done
                    for _ in batch:
                        self._queue.task_done()

            except Exception as e:
                logger.error(f"Background processor error: {e}")
                time.sleep(1)

        logger.info("Background processor stopped")

    def _process_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Process a batch of items."""
        try:
            # Group items by type
            telemetry_items = []
            other_items = []

            for item in batch:
                if item.get("type") == "telemetry":
                    telemetry_items.append(item)
                else:
                    other_items.append(item)

            # Process telemetry items
            if telemetry_items:
                self._process_telemetry_batch(telemetry_items)

            # Process other items
            for item in other_items:
                self._process_item(item)

            # Update metrics
            with self._lock:
                self._metrics["items_processed"] += len(batch)
                self._metrics["batches_processed"] += 1

        except Exception as e:
            # Update error metrics
            with self._lock:
                self._metrics["items_failed"] += len(batch)
                self._metrics["last_error"] = str(e)
                self._metrics["last_error_time"] = time.time()
            raise

    def _process_telemetry_batch(self, items: List[Dict[str, Any]]) -> None:
        """Process telemetry items in batch using unified batch API."""
        if not self.config.telemetry_enabled:
            return

        try:
            # Transform items into TelemetryEvent objects
            events = []
            for item in items:
                if "data" in item:
                    # Extract event type (default to EVENT_CREATE for legacy data)
                    event_type = item.get("event_type", TelemetryEventType.EVENT_CREATE)

                    # Use pre-generated event_id if available, otherwise generate new one
                    event_id = item.get("event_id", generate_event_id())

                    # Create telemetry event
                    event = TelemetryEvent(
                        event_id=event_id,
                        event_type=event_type,
                        payload=item["data"],
                        timestamp=int(item.get("timestamp", time.time()))
                    )
                    events.append(event)

            if events:
                # Submit batch to background thread
                self._executor.submit(self._submit_batch_events, events)

        except Exception as e:
            logger.error(f"Failed to process telemetry batch: {e}")

    def _process_item(self, item: Dict[str, Any]) -> None:
        """Process individual item."""
        try:
            item_type = item.get("type")

            if item_type == "analytics":
                self._executor.submit(self._submit_analytics, item.get("data"))
            elif item_type == "evaluation":
                self._executor.submit(self._submit_evaluation, item.get("data"))
            else:
                logger.warning(f"Unknown item type: {item_type}")

        except Exception as e:
            logger.error(f"Failed to process item: {e}")

    @backoff.on_exception(
        backoff.expo,
        (httpx.HTTPError, httpx.TimeoutException),
        max_tries=3,
        max_time=30,
    )
    def _submit_batch_events(self, events: List[TelemetryEvent]) -> None:
        """Submit telemetry events to unified batch API."""
        try:
            # Run async submission in thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                loop.run_until_complete(self._async_submit_batch_events(events))
            finally:
                loop.close()

        except Exception as e:
            logger.error(f"Failed to submit batch events: {e}")

    async def _async_submit_batch_events(
        self, events: List[TelemetryEvent]
    ) -> None:
        """Async batch event submission to /v1/telemetry/batch."""
        if not self._client:
            self._client = httpx.AsyncClient(
                timeout=self.config.timeout, headers=self.config.get_headers()
            )

        try:
            # Build batch request
            batch_request = TelemetryBatchRequest(
                events=events,
                environment=self.config.environment,
                deduplication=DeduplicationConfig(
                    enabled=self.config.batch_enable_deduplication,
                    ttl=self.config.batch_deduplication_ttl,
                    use_redis_cache=self.config.batch_use_redis_cache,
                    fail_on_duplicate=self.config.batch_fail_on_duplicate,
                )
            )

            # Submit to unified batch endpoint
            response = await self._client.post(
                f"{self.config.host}/v1/telemetry/batch",
                json=batch_request.model_dump(mode="json", exclude_none=True),
            )

            if response.status_code != 200:
                logger.error(
                    f"Batch submission failed: {response.status_code} - {response.text}"
                )
            else:
                # Parse response and handle partial failures
                batch_response = TelemetryBatchResponse(**response.json())
                self._handle_batch_response(batch_response)
                logger.debug(
                    f"Submitted batch: {batch_response.processed_events} processed, "
                    f"{batch_response.duplicate_events} duplicates, "
                    f"{batch_response.failed_events} failed"
                )

        except Exception as e:
            logger.error(f"Batch submission error: {e}")
            raise

    def _handle_batch_response(self, response: TelemetryBatchResponse) -> None:
        """Handle batch response with partial failures."""
        # Log errors for failed events
        if response.errors:
            for error in response.errors:
                logger.warning(
                    f"Event {error.event_id} failed: {error.error}"
                    + (f" - {error.details}" if error.details else "")
                )

        # Log duplicate events (info level, not an error)
        if response.duplicate_event_ids:
            logger.info(
                f"Skipped {len(response.duplicate_event_ids)} duplicate events"
            )

        # Update metrics with failures
        if response.failed_events > 0:
            with self._lock:
                self._metrics["items_failed"] += response.failed_events
                if response.errors:
                    self._metrics["last_error"] = response.errors[0].error
                    self._metrics["last_error_time"] = time.time()

    def _submit_analytics(self, data: Dict[str, Any]) -> None:
        """Submit analytics data."""
        # Placeholder for analytics submission
        logger.debug("Analytics submission not implemented yet")

    def _submit_evaluation(self, data: Dict[str, Any]) -> None:
        """Submit evaluation data."""
        # Placeholder for evaluation submission
        logger.debug("Evaluation submission not implemented yet")

    def submit_telemetry(
        self, data: Dict[str, Any], event_type: str = TelemetryEventType.EVENT_CREATE
    ) -> None:
        """
        Submit telemetry data for background processing.

        Args:
            data: Telemetry payload data
            event_type: Type of event (defaults to EVENT_CREATE)
        """
        if self._shutdown:
            return

        item = {
            "type": "telemetry",
            "data": data,
            "event_type": event_type,
            "timestamp": time.time()
        }

        try:
            self._queue.put_nowait(item)
        except Exception as e:
            logger.error(f"Failed to queue telemetry: {e}")

    def submit_batch_event(self, event: TelemetryEvent) -> None:
        """
        Submit a pre-formed TelemetryEvent for background processing.

        This is the preferred method for submitting structured events.

        Args:
            event: TelemetryEvent with event_id, type, and payload
        """
        if self._shutdown:
            return

        item = {
            "type": "telemetry",
            "data": event.payload,
            "event_type": event.event_type,
            "timestamp": event.timestamp or int(time.time()),
            "event_id": event.event_id,  # Pre-generated ULID
        }

        try:
            self._queue.put_nowait(item)
        except Exception as e:
            logger.error(f"Failed to queue batch event: {e}")

    def submit_analytics(self, data: Dict[str, Any]) -> None:
        """Submit analytics data for background processing."""
        if self._shutdown:
            return

        item = {"type": "analytics", "data": data, "timestamp": time.time()}

        try:
            self._queue.put_nowait(item)
        except Exception as e:
            logger.error(f"Failed to queue analytics: {e}")

    def submit_evaluation(self, data: Dict[str, Any]) -> None:
        """Submit evaluation data for background processing."""
        if self._shutdown:
            return

        item = {"type": "evaluation", "data": data, "timestamp": time.time()}

        try:
            self._queue.put_nowait(item)
        except Exception as e:
            logger.error(f"Failed to queue evaluation: {e}")

    def flush(self, timeout: Optional[float] = None) -> bool:
        """
        Flush pending items and wait for processing to complete.

        Args:
            timeout: Maximum time to wait in seconds (None = wait indefinitely)

        Returns:
            True if all items processed successfully, False if timeout reached
        """
        if self._shutdown:
            return True

        try:
            # For tests and simple cases, just check if queue is already empty
            if self._queue.empty():
                return True

            if timeout is None:
                # Wait indefinitely for all tasks to complete
                self._queue.join()
                return True
            else:
                # Wait with timeout using bounded wait loop
                import threading

                # Use threading event to implement timeout on join()
                completed = threading.Event()

                def wait_for_completion():
                    self._queue.join()
                    completed.set()

                # Start join in separate thread
                join_thread = threading.Thread(target=wait_for_completion)
                join_thread.daemon = True
                join_thread.start()

                # Wait for completion or timeout
                if completed.wait(timeout):
                    return True
                else:
                    return False

        except Exception as e:
            logger.error(f"Failed to flush background processor: {e}")
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current task manager metrics.

        Returns:
            Dict containing queue depth, processed counts, error stats, and timing info
        """
        with self._lock:
            current_time = time.time()
            uptime = current_time - self._metrics["start_time"]

            return {
                "queue_depth": self._queue.qsize(),
                "queue_max_size": self._queue.maxsize,
                "items_processed": self._metrics["items_processed"],
                "items_failed": self._metrics["items_failed"],
                "batches_processed": self._metrics["batches_processed"],
                "uptime_seconds": uptime,
                "processing_rate": (
                    self._metrics["items_processed"] / uptime if uptime > 0 else 0
                ),
                "error_rate": (
                    self._metrics["items_failed"]
                    / max(
                        1,
                        self._metrics["items_processed"]
                        + self._metrics["items_failed"],
                    )
                ),
                "worker_alive": self._worker_thread and self._worker_thread.is_alive(),
                "shutdown": self._shutdown,
                "last_error": self._metrics["last_error"],
                "last_error_time": self._metrics["last_error_time"],
            }

    def is_healthy(self) -> bool:
        """
        Check if the background processor is healthy.

        Returns:
            True if processor is healthy, False if there are critical issues
        """
        if self._shutdown:
            return False

        # Check if worker thread is alive
        if not (self._worker_thread and self._worker_thread.is_alive()):
            return False

        with self._lock:
            # Check if queue is not overwhelmed (less than 90% full)
            queue_usage = self._queue.qsize() / self._queue.maxsize
            if queue_usage > 0.9:
                return False

            # Check if error rate is not too high (less than 10%)
            total_items = (
                self._metrics["items_processed"] + self._metrics["items_failed"]
            )
            if total_items > 10:  # Only check after some processing
                error_rate = self._metrics["items_failed"] / total_items
                if error_rate > 0.1:
                    return False

            # Check if there was a recent critical error (within last 5 minutes)
            if (
                self._metrics["last_error_time"]
                and time.time() - self._metrics["last_error_time"] < 300
            ):
                return False

        return True

    def shutdown(self) -> None:
        """Shutdown background processor."""
        self._shutdown = True

        # Flush remaining items
        try:
            flushed = self.flush(timeout=5)
            if not flushed:
                logger.warning("Background processor shutdown with pending items")
        except Exception:
            pass

        # Shutdown executor
        if self._executor:
            self._executor.shutdown(wait=True)

        # Close HTTP client
        if self._client:
            asyncio.run(self._client.aclose())

        logger.info("Background processor shutdown complete")


# Global background processor instance
_background_processor: Optional[BackgroundProcessor] = None
_lock = threading.Lock()


def get_background_processor(
    config: Optional[Config] = None,
    config_factory: Optional[Callable[[], Config]] = None,
) -> BackgroundProcessor:
    """
    Get or create background processor.

    Args:
        config: Configuration to use
        config_factory: Factory function that returns a Config

    Returns:
        BackgroundProcessor singleton instance

    Raises:
        ValueError: If neither config nor config_factory is provided
    """
    global _background_processor

    with _lock:
        if _background_processor is None:
            if config is None and config_factory is None:
                raise ValueError("Either config or config_factory must be provided")

            if config is None:
                config = config_factory()

            _background_processor = BackgroundProcessor(config)

        return _background_processor


def reset_background_processor() -> None:
    """Reset background processor (for testing)."""
    global _background_processor

    with _lock:
        if _background_processor:
            _background_processor.shutdown()
        _background_processor = None
