"""
Advanced queue system for background task processing.

Inspired by Optik's robust task queue architecture with
priority, retry, and dead letter queue capabilities.
"""

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from queue import Empty, Full, PriorityQueue, Queue
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class TaskStatus(Enum):
    """Task status enumeration."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    DEAD_LETTER = "dead_letter"


@dataclass
class TaskResult:
    """Task execution result."""

    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None
    attempts: int = 0
    completed_at: Optional[datetime] = None


@dataclass
class Task:
    """
    Background task with priority, retry, and metadata support.

    Inspired by Optik's task structure but enhanced for AI workloads.
    """

    id: str
    task_type: str
    data: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL

    # Retry configuration
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    retry_backoff_multiplier: float = 2.0
    max_retry_delay_seconds: float = 60.0

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    scheduled_at: Optional[datetime] = None
    deadline: Optional[datetime] = None

    # Execution tracking
    attempts: int = 0
    last_attempt_at: Optional[datetime] = None
    last_error: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # Context
    user_id: Optional[str] = None
    organization_id: Optional[str] = None
    trace_id: Optional[str] = None

    def __post_init__(self):
        """Initialize task after creation."""
        if not self.id:
            self.id = str(uuid.uuid4())

        if self.scheduled_at is None:
            self.scheduled_at = self.created_at

    def __lt__(self, other) -> bool:
        """Priority queue comparison."""
        if not isinstance(other, Task):
            return NotImplemented

        # Higher priority number = higher priority
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value

        # Earlier scheduled time = higher priority
        return self.scheduled_at < other.scheduled_at

    def is_due(self) -> bool:
        """Check if task is due for execution."""
        return datetime.now(timezone.utc) >= self.scheduled_at

    def is_expired(self) -> bool:
        """Check if task has exceeded its deadline."""
        if self.deadline is None:
            return False
        return datetime.now(timezone.utc) > self.deadline

    def should_retry(self) -> bool:
        """Check if task should be retried."""
        return self.attempts < self.max_retries

    def calculate_retry_delay(self) -> float:
        """Calculate delay for next retry attempt."""
        if self.attempts == 0:
            return self.retry_delay_seconds

        delay = self.retry_delay_seconds * (
            self.retry_backoff_multiplier ** (self.attempts - 1)
        )
        return min(delay, self.max_retry_delay_seconds)

    def schedule_retry(self) -> None:
        """Schedule task for retry."""
        retry_delay = self.calculate_retry_delay()
        self.scheduled_at = datetime.now(timezone.utc) + timedelta(seconds=retry_delay)
        self.attempts += 1
        self.last_attempt_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization."""
        return {
            "id": self.id,
            "task_type": self.task_type,
            "data": self.data,
            "priority": self.priority.name,
            "max_retries": self.max_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
            "retry_backoff_multiplier": self.retry_backoff_multiplier,
            "max_retry_delay_seconds": self.max_retry_delay_seconds,
            "created_at": self.created_at.isoformat(),
            "scheduled_at": (
                self.scheduled_at.isoformat() if self.scheduled_at else None
            ),
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "attempts": self.attempts,
            "last_attempt_at": (
                self.last_attempt_at.isoformat() if self.last_attempt_at else None
            ),
            "last_error": self.last_error,
            "metadata": self.metadata,
            "tags": self.tags,
            "user_id": self.user_id,
            "organization_id": self.organization_id,
            "trace_id": self.trace_id,
        }


class TaskQueue:
    """
    Advanced task queue with priority, retry, and dead letter capabilities.

    Inspired by Optik's queue system but optimized for AI/ML workloads.
    """

    def __init__(
        self,
        name: str = "default",
        max_size: int = 10000,
        enable_dead_letter_queue: bool = True,
        dead_letter_ttl_hours: int = 24,
        enable_metrics: bool = True,
    ):
        self.name = name
        self.max_size = max_size
        self.enable_dead_letter_queue = enable_dead_letter_queue
        self.dead_letter_ttl_hours = dead_letter_ttl_hours
        self.enable_metrics = enable_metrics

        # Queues
        self._priority_queue = PriorityQueue(maxsize=max_size)
        self._retry_queue = PriorityQueue(maxsize=max_size // 4)
        self._dead_letter_queue: deque = deque(maxlen=max_size // 10)

        # Tracking
        self._processing_tasks: Dict[str, Task] = {}
        self._completed_tasks: Dict[str, TaskResult] = {}
        self._task_handlers: Dict[str, Callable] = {}

        # Metrics
        self._metrics = {
            "tasks_enqueued": 0,
            "tasks_processed": 0,
            "tasks_failed": 0,
            "tasks_retried": 0,
            "tasks_dead_lettered": 0,
            "total_processing_time_ms": 0.0,
            "queue_size": 0,
            "retry_queue_size": 0,
            "dead_letter_queue_size": 0,
        }

        # Configuration
        self._shutdown = False

    def register_handler(self, task_type: str, handler: Callable) -> None:
        """Register a handler for a specific task type."""
        self._task_handlers[task_type] = handler
        logger.debug(f"Registered handler for task type: {task_type}")

    def enqueue(
        self,
        task_type: str,
        data: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        delay_seconds: float = 0.0,
        deadline_seconds: Optional[float] = None,
        max_retries: int = 3,
        **kwargs,
    ) -> str:
        """
        Enqueue a new task.

        Args:
            task_type: Type of task to execute
            data: Task data payload
            priority: Task priority level
            delay_seconds: Delay before task becomes available
            deadline_seconds: Task deadline (from now)
            max_retries: Maximum retry attempts
            **kwargs: Additional task parameters

        Returns:
            Task ID
        """
        if self._shutdown:
            raise RuntimeError("Queue is shutting down")

        # Create task
        task = Task(
            id=str(uuid.uuid4()),
            task_type=task_type,
            data=data,
            priority=priority,
            max_retries=max_retries,
            **kwargs,
        )

        # Set scheduling
        if delay_seconds > 0:
            task.scheduled_at = datetime.now(timezone.utc) + timedelta(
                seconds=delay_seconds
            )

        if deadline_seconds:
            task.deadline = datetime.now(timezone.utc) + timedelta(
                seconds=deadline_seconds
            )

        # Enqueue task
        try:
            self._priority_queue.put_nowait(task)
            self._metrics["tasks_enqueued"] += 1
            self._metrics["queue_size"] = self._priority_queue.qsize()

            logger.debug(f"Enqueued task {task.id} of type {task_type}")
            return task.id

        except Full:
            logger.error(f"Task queue full, dropping task {task.id}")
            raise RuntimeError("Task queue is full")

    def dequeue(self, timeout: Optional[float] = None) -> Optional[Task]:
        """
        Dequeue next available task.

        Args:
            timeout: Maximum time to wait for task

        Returns:
            Next task or None if timeout
        """
        if self._shutdown:
            return None

        # Check retry queue first
        task = self._dequeue_from_retry_queue()
        if task:
            return task

        # Check main queue
        try:
            task = self._priority_queue.get(timeout=timeout or 0.1)

            # Check if task is due
            if not task.is_due():
                # Put back in queue if not due yet
                self._priority_queue.put_nowait(task)
                return None

            # Check if task is expired
            if task.is_expired():
                logger.warning(f"Task {task.id} expired, moving to dead letter queue")
                self._move_to_dead_letter(task, "Task expired")
                return None

            # Mark as processing
            self._processing_tasks[task.id] = task
            self._metrics["queue_size"] = self._priority_queue.qsize()

            return task

        except Empty:
            return None

    def _dequeue_from_retry_queue(self) -> Optional[Task]:
        """Dequeue task from retry queue if due."""
        try:
            # Peek at retry queue
            if self._retry_queue.empty():
                return None

            # Get next retry task
            task = self._retry_queue.get_nowait()

            if task.is_due():
                # Task is ready for retry
                self._processing_tasks[task.id] = task
                self._metrics["retry_queue_size"] = self._retry_queue.qsize()
                return task
            else:
                # Put back if not due yet
                self._retry_queue.put_nowait(task)
                return None

        except Empty:
            return None

    def complete_task(self, task_id: str, result: Any = None) -> None:
        """Mark task as completed successfully."""
        if task_id not in self._processing_tasks:
            logger.warning(f"Attempted to complete unknown task: {task_id}")
            return

        task = self._processing_tasks.pop(task_id)

        # Create result
        task_result = TaskResult(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            result=result,
            attempts=task.attempts + 1,
            completed_at=datetime.now(timezone.utc),
        )

        # Calculate execution time
        if task.last_attempt_at:
            execution_time = (
                datetime.now(timezone.utc) - task.last_attempt_at
            ).total_seconds() * 1000
            task_result.execution_time_ms = execution_time
            self._metrics["total_processing_time_ms"] += execution_time

        self._completed_tasks[task_id] = task_result
        self._metrics["tasks_processed"] += 1

        logger.debug(f"Task {task_id} completed successfully")

    def fail_task(self, task_id: str, error: str) -> None:
        """Mark task as failed and handle retry logic."""
        if task_id not in self._processing_tasks:
            logger.warning(f"Attempted to fail unknown task: {task_id}")
            return

        task = self._processing_tasks.pop(task_id)
        task.last_error = error
        task.last_attempt_at = datetime.now(timezone.utc)

        if task.should_retry():
            # Schedule for retry
            task.schedule_retry()

            try:
                self._retry_queue.put_nowait(task)
                self._metrics["tasks_retried"] += 1
                self._metrics["retry_queue_size"] = self._retry_queue.qsize()

                logger.info(
                    f"Task {task_id} scheduled for retry {task.attempts}/{task.max_retries}"
                )

            except Full:
                logger.error(
                    f"Retry queue full, moving task {task_id} to dead letter queue"
                )
                self._move_to_dead_letter(task, "Retry queue full")
        else:
            # Move to dead letter queue
            self._move_to_dead_letter(task, f"Max retries exceeded: {error}")

    def _move_to_dead_letter(self, task: Task, reason: str) -> None:
        """Move task to dead letter queue."""
        if not self.enable_dead_letter_queue:
            logger.error(f"Task {task.id} failed permanently: {reason}")
            return

        # Create dead letter entry
        dead_letter_entry = {
            "task": task.to_dict(),
            "reason": reason,
            "dead_lettered_at": datetime.now(timezone.utc).isoformat(),
            "ttl": datetime.now(timezone.utc)
            + timedelta(hours=self.dead_letter_ttl_hours),
        }

        self._dead_letter_queue.append(dead_letter_entry)
        self._metrics["tasks_dead_lettered"] += 1
        self._metrics["dead_letter_queue_size"] = len(self._dead_letter_queue)

        logger.warning(f"Task {task.id} moved to dead letter queue: {reason}")

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get current status of a task."""
        if task_id in self._processing_tasks:
            return TaskStatus.PROCESSING
        elif task_id in self._completed_tasks:
            return self._completed_tasks[task_id].status
        else:
            # Check queues
            return (
                TaskStatus.PENDING
            )  # Simplified - would need queue scanning for exact status

    def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """Get result of completed task."""
        return self._completed_tasks.get(task_id)

    def get_metrics(self) -> Dict[str, Any]:
        """Get queue metrics."""
        # Update real-time metrics
        self._metrics["queue_size"] = self._priority_queue.qsize()
        self._metrics["retry_queue_size"] = self._retry_queue.qsize()
        self._metrics["dead_letter_queue_size"] = len(self._dead_letter_queue)
        self._metrics["processing_tasks"] = len(self._processing_tasks)

        # Calculate averages
        if self._metrics["tasks_processed"] > 0:
            self._metrics["avg_processing_time_ms"] = (
                self._metrics["total_processing_time_ms"]
                / self._metrics["tasks_processed"]
            )

        return self._metrics.copy()

    def cleanup_completed_tasks(self, older_than_hours: int = 24) -> int:
        """Clean up old completed tasks."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=older_than_hours)

        to_remove = []
        for task_id, result in self._completed_tasks.items():
            if result.completed_at and result.completed_at < cutoff_time:
                to_remove.append(task_id)

        for task_id in to_remove:
            del self._completed_tasks[task_id]

        logger.info(f"Cleaned up {len(to_remove)} completed tasks")
        return len(to_remove)

    def cleanup_dead_letter_queue(self) -> int:
        """Clean up expired dead letter entries."""
        if not self.enable_dead_letter_queue:
            return 0

        now = datetime.now(timezone.utc)
        original_size = len(self._dead_letter_queue)

        # Remove expired entries
        self._dead_letter_queue = deque(
            [
                entry
                for entry in self._dead_letter_queue
                if entry.get("ttl")
                and datetime.fromisoformat(
                    entry["ttl"].replace("Z", "+00:00").replace("+00:00", "")
                )
                > now
            ],
            maxlen=self._dead_letter_queue.maxlen,
        )

        cleaned = original_size - len(self._dead_letter_queue)
        self._metrics["dead_letter_queue_size"] = len(self._dead_letter_queue)

        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} expired dead letter entries")

        return cleaned

    def get_dead_letter_entries(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get dead letter queue entries."""
        if not self.enable_dead_letter_queue:
            return []

        return list(self._dead_letter_queue)[-limit:]

    def flush(self, timeout: float = 30.0) -> None:
        """Wait for all queued tasks to be processed."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            if (
                self._priority_queue.empty()
                and self._retry_queue.empty()
                and not self._processing_tasks
            ):
                break
            time.sleep(0.1)

        logger.info("Queue flush completed")

    def shutdown(self) -> None:
        """Shutdown the queue."""
        self._shutdown = True
        logger.info(f"Queue {self.name} shutdown initiated")


class TaskQueueManager:
    """
    Manager for multiple task queues with routing and load balancing.

    Provides enterprise-grade queue management capabilities.
    """

    def __init__(self):
        self._queues: Dict[str, TaskQueue] = {}
        self._default_queue_name = "default"
        self._shutdown = False

    def create_queue(
        self,
        name: str,
        max_size: int = 10000,
        enable_dead_letter_queue: bool = True,
        **kwargs,
    ) -> TaskQueue:
        """Create a new task queue."""
        if name in self._queues:
            raise ValueError(f"Queue {name} already exists")

        queue = TaskQueue(
            name=name,
            max_size=max_size,
            enable_dead_letter_queue=enable_dead_letter_queue,
            **kwargs,
        )

        self._queues[name] = queue
        logger.info(f"Created queue: {name}")

        return queue

    def get_queue(self, name: str = None) -> TaskQueue:
        """Get a task queue by name."""
        queue_name = name or self._default_queue_name

        if queue_name not in self._queues:
            # Create default queue if it doesn't exist
            if queue_name == self._default_queue_name:
                return self.create_queue(queue_name)
            else:
                raise ValueError(f"Queue {queue_name} does not exist")

        return self._queues[queue_name]

    def enqueue_to_queue(
        self, queue_name: str, task_type: str, data: Dict[str, Any], **kwargs
    ) -> str:
        """Enqueue task to specific queue."""
        queue = self.get_queue(queue_name)
        return queue.enqueue(task_type, data, **kwargs)

    def enqueue(
        self, task_type: str, data: Dict[str, Any], queue_name: str = None, **kwargs
    ) -> str:
        """Enqueue task with automatic queue routing."""
        # Route to appropriate queue based on task type
        target_queue = self._route_task(task_type, queue_name)
        queue = self.get_queue(target_queue)
        return queue.enqueue(task_type, data, **kwargs)

    def _route_task(self, task_type: str, preferred_queue: str = None) -> str:
        """Route task to appropriate queue."""
        if preferred_queue:
            return preferred_queue

        # Route based on task type
        if task_type.startswith("analytics"):
            return "analytics"
        elif task_type.startswith("evaluation"):
            return "evaluation"
        elif task_type.startswith("telemetry"):
            return "telemetry"
        else:
            return self._default_queue_name

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all queues."""
        return {name: queue.get_metrics() for name, queue in self._queues.items()}

    def cleanup_all_queues(self) -> None:
        """Run cleanup on all queues."""
        for queue in self._queues.values():
            queue.cleanup_completed_tasks()
            queue.cleanup_dead_letter_queue()

    def shutdown_all(self) -> None:
        """Shutdown all queues."""
        self._shutdown = True

        for queue in self._queues.values():
            queue.shutdown()

        logger.info("All queues shut down")


# Global queue manager instance
_queue_manager: Optional[TaskQueueManager] = None


def get_queue_manager() -> TaskQueueManager:
    """Get global queue manager instance."""
    global _queue_manager

    if _queue_manager is None:
        _queue_manager = TaskQueueManager()

    return _queue_manager
