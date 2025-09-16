"""
Advanced background task processing for the Brokle SDK.

This module provides enterprise-grade background processing capabilities
inspired by Optik's architecture with advanced queue management,
worker pools, and comprehensive monitoring.
"""

# Legacy processor (backwards compatibility)
from .processor import get_background_processor, BackgroundProcessor

# Advanced queue system
from .queue import (
    Task,
    TaskResult,
    TaskStatus,
    TaskPriority,
    TaskQueue,
    TaskQueueManager,
    get_queue_manager
)

# Worker system
from .workers import (
    BaseWorker,
    ThreadWorker,
    AsyncWorker,
    WorkerPool,
    WorkerStatus,
    WorkerMetrics
)

__all__ = [
    # Legacy processor
    "get_background_processor",
    "BackgroundProcessor",

    # Advanced queue system
    "Task",
    "TaskResult",
    "TaskStatus",
    "TaskPriority",
    "TaskQueue",
    "TaskQueueManager",
    "get_queue_manager",

    # Worker system
    "BaseWorker",
    "ThreadWorker",
    "AsyncWorker",
    "WorkerPool",
    "WorkerStatus",
    "WorkerMetrics",
]