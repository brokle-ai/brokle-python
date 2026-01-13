"""
Annotations Module

Provides annotation queue management for human-in-the-loop (HITL) evaluation workflows.

Annotation queues allow human annotators to review and score AI outputs,
enabling quality assessment and feedback collection at scale.

Usage:
    >>> from brokle import Brokle
    >>>
    >>> client = Brokle(api_key="bk_...")
    >>>
    >>> # Add traces to annotation queue
    >>> result = client.annotations.add_traces(
    ...     queue_id="queue123",
    ...     trace_ids=["trace1", "trace2", "trace3"],
    ...     priority=5,
    ... )
    >>> print(f"Added {result['created']} items")
    >>>
    >>> # Add items with mixed types
    >>> client.annotations.add_items(
    ...     queue_id="queue123",
    ...     items=[
    ...         {"object_id": "trace1", "object_type": "trace"},
    ...         {"object_id": "span1", "object_type": "span", "priority": 10},
    ...     ]
    ... )
"""

from ._managers import AnnotationQueuesManager, AsyncAnnotationQueuesManager
from .exceptions import (
    AnnotationError,
    AssignmentError,
    ItemLockedError,
    ItemNotFoundError,
    NoItemsAvailableError,
    QueueNotFoundError,
)
from .types import (
    AddItemRequest,
    AnnotationQueue,
    AssignmentRole,
    ItemStatus,
    ObjectType,
    QueueAssignment,
    QueueItem,
    QueueSettings,
    QueueStats,
    QueueStatus,
    ScoreSubmission,
)

__all__ = [
    # Managers
    "AnnotationQueuesManager",
    "AsyncAnnotationQueuesManager",
    # Types
    "AnnotationQueue",
    "QueueItem",
    "QueueAssignment",
    "QueueSettings",
    "QueueStats",
    "QueueStatus",
    "ItemStatus",
    "ObjectType",
    "AssignmentRole",
    "AddItemRequest",
    "ScoreSubmission",
    # Exceptions
    "AnnotationError",
    "QueueNotFoundError",
    "ItemNotFoundError",
    "ItemLockedError",
    "NoItemsAvailableError",
    "AssignmentError",
]
