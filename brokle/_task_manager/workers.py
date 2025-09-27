"""
Advanced worker system for background task processing.

Inspired by Optik's worker architecture with support for
multi-threading, auto-scaling, and health monitoring.
"""

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from .queue import Task, TaskPriority, TaskQueue, TaskResult, TaskStatus

logger = logging.getLogger(__name__)


class WorkerStatus(Enum):
    """Worker status enumeration."""

    STARTING = "starting"
    RUNNING = "running"
    IDLE = "idle"
    BUSY = "busy"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class WorkerMetrics:
    """Worker performance metrics."""

    worker_id: str
    status: WorkerStatus
    tasks_processed: int = 0
    tasks_failed: int = 0
    total_processing_time_ms: float = 0.0
    last_task_at: Optional[datetime] = None
    uptime_seconds: float = 0.0
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None

    @property
    def average_processing_time_ms(self) -> float:
        """Calculate average processing time."""
        if self.tasks_processed == 0:
            return 0.0
        return self.total_processing_time_ms / self.tasks_processed

    @property
    def success_rate(self) -> float:
        """Calculate task success rate."""
        total_tasks = self.tasks_processed + self.tasks_failed
        if total_tasks == 0:
            return 1.0
        return self.tasks_processed / total_tasks

    @property
    def tasks_per_second(self) -> float:
        """Calculate tasks processed per second."""
        if self.uptime_seconds == 0:
            return 0.0
        return (self.tasks_processed + self.tasks_failed) / self.uptime_seconds


class BaseWorker(ABC):
    """
    Base worker class for task processing.

    Provides common functionality for task execution, metrics,
    and lifecycle management.
    """

    def __init__(
        self,
        worker_id: str,
        queue: TaskQueue,
        task_handlers: Optional[Dict[str, Callable]] = None,
        max_task_time_seconds: float = 300.0,
        health_check_interval_seconds: float = 30.0,
    ):
        self.worker_id = worker_id
        self.queue = queue
        self.task_handlers = task_handlers or {}
        self.max_task_time_seconds = max_task_time_seconds
        self.health_check_interval_seconds = health_check_interval_seconds

        # State
        self.status = WorkerStatus.STOPPED
        self._shutdown = False
        self._start_time: Optional[datetime] = None
        self._current_task: Optional[Task] = None

        # Metrics
        self.metrics = WorkerMetrics(worker_id=worker_id, status=self.status)

        # Threading
        self._worker_thread: Optional[threading.Thread] = None
        self._health_check_thread: Optional[threading.Thread] = None

    def register_handler(self, task_type: str, handler: Callable) -> None:
        """Register a task handler."""
        self.task_handlers[task_type] = handler
        logger.debug(f"Worker {self.worker_id} registered handler for {task_type}")

    def start(self) -> None:
        """Start the worker."""
        if self.status != WorkerStatus.STOPPED:
            logger.warning(f"Worker {self.worker_id} already running")
            return

        self.status = WorkerStatus.STARTING
        self._shutdown = False
        self._start_time = datetime.now(timezone.utc)

        # Start worker thread
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name=f"brokle-worker-{self.worker_id}",
            daemon=True,
        )
        self._worker_thread.start()

        # Start health check thread
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            name=f"brokle-health-{self.worker_id}",
            daemon=True,
        )
        self._health_check_thread.start()

        logger.info(f"Worker {self.worker_id} started")

    def stop(self, timeout: float = 30.0) -> None:
        """Stop the worker gracefully."""
        if self.status == WorkerStatus.STOPPED:
            return

        self.status = WorkerStatus.STOPPING
        self._shutdown = True

        # Wait for current task to complete
        start_time = time.time()
        while self._current_task is not None and time.time() - start_time < timeout:
            time.sleep(0.1)

        # Wait for threads to finish
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)

        if self._health_check_thread and self._health_check_thread.is_alive():
            self._health_check_thread.join(timeout=1.0)

        self.status = WorkerStatus.STOPPED
        logger.info(f"Worker {self.worker_id} stopped")

    def _worker_loop(self) -> None:
        """Main worker loop."""
        self.status = WorkerStatus.RUNNING
        logger.info(f"Worker {self.worker_id} loop started")

        while not self._shutdown:
            try:
                # Get next task
                task = self.queue.dequeue(timeout=1.0)

                if task is None:
                    self.status = WorkerStatus.IDLE
                    continue

                # Process task
                self.status = WorkerStatus.BUSY
                self._current_task = task
                self._process_task(task)

            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}")
                self.status = WorkerStatus.ERROR
                time.sleep(1.0)
            finally:
                self._current_task = None

        logger.info(f"Worker {self.worker_id} loop ended")

    def _process_task(self, task: Task) -> None:
        """Process a single task."""
        start_time = time.time()

        try:
            # Find handler
            handler = self.task_handlers.get(task.task_type)
            if not handler:
                raise ValueError(
                    f"No handler registered for task type: {task.task_type}"
                )

            # Execute task with timeout
            result = self._execute_with_timeout(handler, task)

            # Mark as completed
            self.queue.complete_task(task.id, result)

            # Update metrics
            processing_time_ms = (time.time() - start_time) * 1000
            self.metrics.tasks_processed += 1
            self.metrics.total_processing_time_ms += processing_time_ms
            self.metrics.last_task_at = datetime.now(timezone.utc)

            logger.debug(
                f"Worker {self.worker_id} completed task {task.id} in {processing_time_ms:.1f}ms"
            )

        except Exception as e:
            # Mark as failed
            error_msg = f"Task execution failed: {str(e)}"
            self.queue.fail_task(task.id, error_msg)

            # Update metrics
            self.metrics.tasks_failed += 1

            logger.error(f"Worker {self.worker_id} failed task {task.id}: {e}")

    @abstractmethod
    def _execute_with_timeout(self, handler: Callable, task: Task) -> Any:
        """Execute task handler with timeout. Implemented by subclasses."""
        pass

    def _health_check_loop(self) -> None:
        """Health check loop."""
        while not self._shutdown:
            try:
                self._update_health_metrics()
                time.sleep(self.health_check_interval_seconds)
            except Exception as e:
                logger.error(f"Health check error for worker {self.worker_id}: {e}")

    def _update_health_metrics(self) -> None:
        """Update health and performance metrics."""
        if self._start_time:
            self.metrics.uptime_seconds = (
                datetime.now(timezone.utc) - self._start_time
            ).total_seconds()

        self.metrics.status = self.status

        # Optional: Add memory and CPU monitoring
        try:
            import psutil

            process = psutil.Process()
            self.metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            self.metrics.cpu_usage_percent = process.cpu_percent()
        except ImportError:
            pass  # psutil not available

    def get_metrics(self) -> WorkerMetrics:
        """Get current worker metrics."""
        self._update_health_metrics()
        return self.metrics


class ThreadWorker(BaseWorker):
    """
    Thread-based worker for synchronous task processing.

    Uses ThreadPoolExecutor for task execution with timeout support.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"task-{self.worker_id}"
        )

    def _execute_with_timeout(self, handler: Callable, task: Task) -> Any:
        """Execute handler in thread with timeout."""
        future = self._executor.submit(handler, task)

        try:
            return future.result(timeout=self.max_task_time_seconds)
        except TimeoutError:
            future.cancel()
            raise TimeoutError(
                f"Task {task.id} timed out after {self.max_task_time_seconds} seconds"
            )

    def stop(self, timeout: float = 30.0) -> None:
        """Stop worker and shutdown executor."""
        super().stop(timeout)
        if self._executor:
            self._executor.shutdown(wait=True, timeout=5.0)


class AsyncWorker(BaseWorker):
    """
    Async-based worker for asynchronous task processing.

    Uses asyncio for task execution with timeout support.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _worker_loop(self) -> None:
        """Async worker loop."""
        # Create new event loop for this thread
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._async_worker_loop())
        finally:
            self._loop.close()

    async def _async_worker_loop(self) -> None:
        """Async worker main loop."""
        self.status = WorkerStatus.RUNNING
        logger.info(f"Async worker {self.worker_id} loop started")

        while not self._shutdown:
            try:
                # Get next task (non-blocking)
                task = self.queue.dequeue(timeout=0.1)

                if task is None:
                    self.status = WorkerStatus.IDLE
                    await asyncio.sleep(0.1)
                    continue

                # Process task
                self.status = WorkerStatus.BUSY
                self._current_task = task
                await self._async_process_task(task)

            except Exception as e:
                logger.error(f"Async worker {self.worker_id} error: {e}")
                self.status = WorkerStatus.ERROR
                await asyncio.sleep(1.0)
            finally:
                self._current_task = None

        logger.info(f"Async worker {self.worker_id} loop ended")

    async def _async_process_task(self, task: Task) -> None:
        """Process task asynchronously."""
        start_time = time.time()

        try:
            # Find handler
            handler = self.task_handlers.get(task.task_type)
            if not handler:
                raise ValueError(
                    f"No handler registered for task type: {task.task_type}"
                )

            # Execute with timeout
            result = await self._execute_with_timeout(handler, task)

            # Mark as completed
            self.queue.complete_task(task.id, result)

            # Update metrics
            processing_time_ms = (time.time() - start_time) * 1000
            self.metrics.tasks_processed += 1
            self.metrics.total_processing_time_ms += processing_time_ms
            self.metrics.last_task_at = datetime.now(timezone.utc)

            logger.debug(
                f"Async worker {self.worker_id} completed task {task.id} in {processing_time_ms:.1f}ms"
            )

        except Exception as e:
            # Mark as failed
            error_msg = f"Async task execution failed: {str(e)}"
            self.queue.fail_task(task.id, error_msg)

            # Update metrics
            self.metrics.tasks_failed += 1

            logger.error(f"Async worker {self.worker_id} failed task {task.id}: {e}")

    async def _execute_with_timeout(self, handler: Callable, task: Task) -> Any:
        """Execute handler with async timeout."""
        if asyncio.iscoroutinefunction(handler):
            # Async handler
            return await asyncio.wait_for(
                handler(task), timeout=self.max_task_time_seconds
            )
        else:
            # Sync handler - run in executor
            loop = asyncio.get_event_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(None, handler, task),
                timeout=self.max_task_time_seconds,
            )


class WorkerPool:
    """
    Pool of workers with auto-scaling and load balancing.

    Provides enterprise-grade worker management with dynamic scaling.
    """

    def __init__(
        self,
        queue: TaskQueue,
        worker_class: type = ThreadWorker,
        min_workers: int = 1,
        max_workers: int = 10,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.2,
        scale_check_interval_seconds: float = 30.0,
        task_handlers: Optional[Dict[str, Callable]] = None,
    ):
        self.queue = queue
        self.worker_class = worker_class
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.scale_check_interval_seconds = scale_check_interval_seconds
        self.task_handlers = task_handlers or {}

        # Workers
        self.workers: Dict[str, BaseWorker] = {}
        self._worker_counter = 0

        # Scaling
        self._shutdown = False
        self._scaling_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the worker pool."""
        if self._scaling_thread and self._scaling_thread.is_alive():
            logger.warning("Worker pool already started")
            return

        # Start initial workers
        for _ in range(self.min_workers):
            self._add_worker()

        # Start scaling thread
        self._scaling_thread = threading.Thread(
            target=self._scaling_loop, name="brokle-scaling", daemon=True
        )
        self._scaling_thread.start()

        logger.info(f"Worker pool started with {len(self.workers)} workers")

    def stop(self, timeout: float = 30.0) -> None:
        """Stop the worker pool."""
        self._shutdown = True

        # Stop all workers
        for worker in self.workers.values():
            worker.stop(timeout=timeout / len(self.workers))

        # Wait for scaling thread
        if self._scaling_thread and self._scaling_thread.is_alive():
            self._scaling_thread.join(timeout=5.0)

        logger.info("Worker pool stopped")

    def register_handler(self, task_type: str, handler: Callable) -> None:
        """Register task handler with all workers."""
        self.task_handlers[task_type] = handler

        # Register with existing workers
        for worker in self.workers.values():
            worker.register_handler(task_type, handler)

        logger.info(
            f"Registered handler for {task_type} with {len(self.workers)} workers"
        )

    def _add_worker(self) -> str:
        """Add a new worker to the pool."""
        self._worker_counter += 1
        worker_id = f"worker-{self._worker_counter}"

        worker = self.worker_class(
            worker_id=worker_id,
            queue=self.queue,
            task_handlers=self.task_handlers.copy(),
        )

        worker.start()
        self.workers[worker_id] = worker

        logger.info(f"Added worker {worker_id} (total: {len(self.workers)})")
        return worker_id

    def _remove_worker(self) -> bool:
        """Remove an idle worker from the pool."""
        # Find idle worker
        for worker_id, worker in self.workers.items():
            if worker.status == WorkerStatus.IDLE:
                worker.stop()
                del self.workers[worker_id]
                logger.info(f"Removed worker {worker_id} (total: {len(self.workers)})")
                return True

        return False

    def _scaling_loop(self) -> None:
        """Auto-scaling loop."""
        logger.info("Worker pool auto-scaling started")

        while not self._shutdown:
            try:
                self._check_scaling()
                time.sleep(self.scale_check_interval_seconds)
            except Exception as e:
                logger.error(f"Scaling error: {e}")

        logger.info("Worker pool auto-scaling stopped")

    def _check_scaling(self) -> None:
        """Check if scaling is needed."""
        if len(self.workers) == 0:
            return

        # Calculate metrics
        queue_metrics = self.queue.get_metrics()
        total_queue_size = queue_metrics.get("queue_size", 0) + queue_metrics.get(
            "retry_queue_size", 0
        )

        busy_workers = sum(
            1 for w in self.workers.values() if w.status == WorkerStatus.BUSY
        )
        worker_utilization = busy_workers / len(self.workers) if self.workers else 0

        # Scale up conditions
        should_scale_up = (
            worker_utilization > self.scale_up_threshold
            and total_queue_size > 0
            and len(self.workers) < self.max_workers
        )

        # Scale down conditions
        should_scale_down = (
            worker_utilization < self.scale_down_threshold
            and len(self.workers) > self.min_workers
        )

        if should_scale_up:
            self._add_worker()
        elif should_scale_down:
            self._remove_worker()

    def get_pool_metrics(self) -> Dict[str, Any]:
        """Get metrics for the entire worker pool."""
        worker_metrics = [worker.get_metrics() for worker in self.workers.values()]

        # Aggregate metrics
        total_tasks_processed = sum(m.tasks_processed for m in worker_metrics)
        total_tasks_failed = sum(m.tasks_failed for m in worker_metrics)
        total_processing_time = sum(m.total_processing_time_ms for m in worker_metrics)

        status_counts = {}
        for metric in worker_metrics:
            status = metric.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        avg_processing_time = (
            total_processing_time / total_tasks_processed
            if total_tasks_processed > 0
            else 0.0
        )

        success_rate = (
            total_tasks_processed / (total_tasks_processed + total_tasks_failed)
            if (total_tasks_processed + total_tasks_failed) > 0
            else 1.0
        )

        return {
            "worker_count": len(self.workers),
            "status_counts": status_counts,
            "total_tasks_processed": total_tasks_processed,
            "total_tasks_failed": total_tasks_failed,
            "average_processing_time_ms": avg_processing_time,
            "success_rate": success_rate,
            "queue_metrics": self.queue.get_metrics(),
            "individual_workers": {
                worker_id: {
                    "status": metric.status.value,
                    "tasks_processed": metric.tasks_processed,
                    "tasks_failed": metric.tasks_failed,
                    "avg_processing_time_ms": metric.average_processing_time_ms,
                    "success_rate": metric.success_rate,
                    "uptime_seconds": metric.uptime_seconds,
                }
                for worker_id, metric in zip(self.workers.keys(), worker_metrics)
            },
        }
