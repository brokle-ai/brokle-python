"""
Annotation Queue Types

Data types for annotation queue management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class QueueStatus(str, Enum):
    """Annotation queue status."""

    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


class ItemStatus(str, Enum):
    """Annotation queue item status."""

    PENDING = "pending"
    COMPLETED = "completed"
    SKIPPED = "skipped"


class ObjectType(str, Enum):
    """Type of object being annotated."""

    TRACE = "trace"
    SPAN = "span"


class AssignmentRole(str, Enum):
    """Role in annotation queue."""

    ADMIN = "admin"
    REVIEWER = "reviewer"
    ANNOTATOR = "annotator"


@dataclass
class QueueSettings:
    """Annotation queue settings."""

    lock_timeout_seconds: int = 300
    require_score_config: bool = False


@dataclass
class QueueStats:
    """Annotation queue statistics."""

    total_items: int = 0
    pending_items: int = 0
    completed_items: int = 0
    skipped_items: int = 0
    locked_items: int = 0


@dataclass
class AnnotationQueue:
    """Annotation queue data."""

    id: str
    project_id: str
    name: str
    status: QueueStatus
    description: Optional[str] = None
    score_config_ids: List[str] = field(default_factory=list)
    settings: Optional[QueueSettings] = None
    stats: Optional[QueueStats] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnnotationQueue":
        """Create from API response dictionary."""
        settings = None
        if data.get("settings"):
            settings = QueueSettings(
                lock_timeout_seconds=data["settings"].get("lock_timeout_seconds", 300),
                require_score_config=data["settings"].get("require_score_config", False),
            )

        stats = None
        if data.get("stats"):
            stats = QueueStats(
                total_items=data["stats"].get("total_items", 0),
                pending_items=data["stats"].get("pending_items", 0),
                completed_items=data["stats"].get("completed_items", 0),
                skipped_items=data["stats"].get("skipped_items", 0),
                locked_items=data["stats"].get("locked_items", 0),
            )

        return cls(
            id=data["id"],
            project_id=data["project_id"],
            name=data["name"],
            status=QueueStatus(data["status"]),
            description=data.get("description"),
            score_config_ids=data.get("score_config_ids", []),
            settings=settings,
            stats=stats,
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


@dataclass
class QueueItem:
    """Annotation queue item data."""

    id: str
    queue_id: str
    object_id: str
    object_type: ObjectType
    status: ItemStatus
    priority: int = 0
    locked_at: Optional[datetime] = None
    locked_by_user_id: Optional[str] = None
    annotator_user_id: Optional[str] = None
    completed_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueueItem":
        """Create from API response dictionary."""
        return cls(
            id=data["id"],
            queue_id=data["queue_id"],
            object_id=data["object_id"],
            object_type=ObjectType(data["object_type"]),
            status=ItemStatus(data["status"]),
            priority=data.get("priority", 0),
            locked_at=data.get("locked_at"),
            locked_by_user_id=data.get("locked_by_user_id"),
            annotator_user_id=data.get("annotator_user_id"),
            completed_at=data.get("completed_at"),
            metadata=data.get("metadata"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


@dataclass
class QueueAssignment:
    """Queue assignment data."""

    queue_id: str
    user_id: str
    role: AssignmentRole
    assigned_by_user_id: str
    created_at: Optional[datetime] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueueAssignment":
        """Create from API response dictionary."""
        return cls(
            queue_id=data["queue_id"],
            user_id=data["user_id"],
            role=AssignmentRole(data["role"]),
            assigned_by_user_id=data["assigned_by_user_id"],
            created_at=data.get("created_at"),
        )


@dataclass
class ScoreSubmission:
    """Score to submit when completing an item."""

    score_config_id: str
    value: float
    comment: Optional[str] = None


@dataclass
class AddItemRequest:
    """Request to add an item to a queue."""

    object_id: str
    object_type: ObjectType
    priority: int = 0
    metadata: Optional[Dict[str, Any]] = None
