"""
Background processing for Brokle SDK.
"""

import asyncio
import logging
import threading
from queue import Queue, Empty
from typing import Any, Dict, Optional, Callable, List
from concurrent.futures import ThreadPoolExecutor
import time

import httpx
import backoff

from ..config import Config

logger = logging.getLogger(__name__)


class BackgroundProcessor:
    """Background processor for telemetry and other async tasks."""
    
    def __init__(self, config: Config):
        self.config = config
        self._queue: Queue = Queue(maxsize=10000)
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="brokle-bg")
        self._shutdown = False
        self._worker_thread: Optional[threading.Thread] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._lock = threading.Lock()
        
        # Start worker thread
        self._start_worker()
    
    def _start_worker(self) -> None:
        """Start background worker thread."""
        if self._worker_thread and self._worker_thread.is_alive():
            return
        
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="brokle-bg-worker",
            daemon=True
        )
        self._worker_thread.start()
    
    def _worker_loop(self) -> None:
        """Background worker loop."""
        logger.info("Background processor started")
        
        while not self._shutdown:
            try:
                # Process items in batches
                batch = []
                batch_size = min(self.config.telemetry_batch_size, 100)
                
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
        # Group items by type
        telemetry_items = []
        other_items = []
        
        for item in batch:
            if item.get('type') == 'telemetry':
                telemetry_items.append(item)
            else:
                other_items.append(item)
        
        # Process telemetry items
        if telemetry_items:
            self._process_telemetry_batch(telemetry_items)
        
        # Process other items
        for item in other_items:
            self._process_item(item)
    
    def _process_telemetry_batch(self, items: List[Dict[str, Any]]) -> None:
        """Process telemetry items in batch."""
        if not self.config.telemetry_enabled:
            return
        
        try:
            # Prepare telemetry data
            telemetry_data = []
            for item in items:
                if 'data' in item:
                    telemetry_data.append(item['data'])
            
            if telemetry_data:
                # Submit to background thread
                self._executor.submit(self._submit_telemetry, telemetry_data)
                
        except Exception as e:
            logger.error(f"Failed to process telemetry batch: {e}")
    
    def _process_item(self, item: Dict[str, Any]) -> None:
        """Process individual item."""
        try:
            item_type = item.get('type')
            
            if item_type == 'analytics':
                self._executor.submit(self._submit_analytics, item.get('data'))
            elif item_type == 'evaluation':
                self._executor.submit(self._submit_evaluation, item.get('data'))
            else:
                logger.warning(f"Unknown item type: {item_type}")
                
        except Exception as e:
            logger.error(f"Failed to process item: {e}")
    
    @backoff.on_exception(
        backoff.expo,
        (httpx.HTTPError, httpx.TimeoutException),
        max_tries=3,
        max_time=30
    )
    def _submit_telemetry(self, telemetry_data: List[Dict[str, Any]]) -> None:
        """Submit telemetry data to Brokle."""
        try:
            # Run async submission in thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                loop.run_until_complete(self._async_submit_telemetry(telemetry_data))
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Failed to submit telemetry: {e}")
    
    async def _async_submit_telemetry(self, telemetry_data: List[Dict[str, Any]]) -> None:
        """Async telemetry submission."""
        if not self._client:
            self._client = httpx.AsyncClient(
                timeout=self.config.timeout,
                headers=self.config.get_headers()
            )
        
        try:
            response = await self._client.post(
                f"{self.config.host}/api/v1/telemetry/bulk",
                json={
                    "telemetry": telemetry_data,
                    "batch_size": len(telemetry_data),
                    "timestamp": time.time()
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Telemetry submission failed: {response.status_code} - {response.text}")
            else:
                logger.debug(f"Submitted {len(telemetry_data)} telemetry items")
                
        except Exception as e:
            logger.error(f"Telemetry submission error: {e}")
            raise
    
    def _submit_analytics(self, data: Dict[str, Any]) -> None:
        """Submit analytics data."""
        # Placeholder for analytics submission
        logger.debug("Analytics submission not implemented yet")
    
    def _submit_evaluation(self, data: Dict[str, Any]) -> None:
        """Submit evaluation data."""
        # Placeholder for evaluation submission
        logger.debug("Evaluation submission not implemented yet")
    
    def submit_telemetry(self, data: Dict[str, Any]) -> None:
        """Submit telemetry data for background processing."""
        if self._shutdown:
            return
        
        item = {
            'type': 'telemetry',
            'data': data,
            'timestamp': time.time()
        }
        
        try:
            self._queue.put_nowait(item)
        except Exception as e:
            logger.error(f"Failed to queue telemetry: {e}")
    
    def submit_analytics(self, data: Dict[str, Any]) -> None:
        """Submit analytics data for background processing."""
        if self._shutdown:
            return
        
        item = {
            'type': 'analytics',
            'data': data,
            'timestamp': time.time()
        }
        
        try:
            self._queue.put_nowait(item)
        except Exception as e:
            logger.error(f"Failed to queue analytics: {e}")
    
    def submit_evaluation(self, data: Dict[str, Any]) -> None:
        """Submit evaluation data for background processing."""
        if self._shutdown:
            return
        
        item = {
            'type': 'evaluation',
            'data': data,
            'timestamp': time.time()
        }
        
        try:
            self._queue.put_nowait(item)
        except Exception as e:
            logger.error(f"Failed to queue evaluation: {e}")
    
    def flush(self, timeout: Optional[float] = None) -> None:
        """Flush pending items."""
        if self._shutdown:
            return
        
        try:
            self._queue.join()
            if timeout:
                time.sleep(min(timeout, 5))
        except Exception as e:
            logger.error(f"Failed to flush background processor: {e}")
    
    def shutdown(self) -> None:
        """Shutdown background processor."""
        self._shutdown = True
        
        # Flush remaining items
        try:
            self.flush(timeout=5)
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


def get_background_processor(config: Optional[Config] = None) -> BackgroundProcessor:
    """Get or create background processor."""
    global _background_processor
    
    with _lock:
        if _background_processor is None:
            if config is None:
                from ..client import get_client
                config = get_client().config.model_copy()
            _background_processor = BackgroundProcessor(config)

        return _background_processor


def reset_background_processor() -> None:
    """Reset background processor (for testing)."""
    global _background_processor
    
    with _lock:
        if _background_processor:
            _background_processor.shutdown()
        _background_processor = None
