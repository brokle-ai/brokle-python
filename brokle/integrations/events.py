"""
Enterprise event system for Brokle SDK.

Provides comprehensive event management with filtering, routing,
and real-time streaming capabilities.
"""

import asyncio
import json
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Set, Union, AsyncIterator
from weakref import WeakSet

from .webhooks import WebhookEventType, WebhookManager, send_webhook_event

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """Event priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class EventStatus(Enum):
    """Event processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class Event:
    """
    Event data structure with metadata and routing information.
    """
    id: str
    event_type: str
    data: Dict[str, Any]
    priority: EventPriority = EventPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Context
    user_id: Optional[str] = None
    organization_id: Optional[str] = None
    trace_id: Optional[str] = None
    session_id: Optional[str] = None

    # Routing
    source: str = "brokle-sdk"
    target: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    # Processing
    status: EventStatus = EventStatus.PENDING
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    # TTL
    expires_at: Optional[datetime] = None

    def __post_init__(self):
        """Initialize event after creation."""
        if not self.id:
            self.id = str(uuid.uuid4())

        # Set default TTL (24 hours)
        if self.expires_at is None:
            self.expires_at = self.created_at + timedelta(hours=24)

    def is_expired(self) -> bool:
        """Check if event has expired."""
        return self.expires_at is not None and datetime.utcnow() > self.expires_at

    def should_retry(self) -> bool:
        """Check if event should be retried."""
        return (
            self.status == EventStatus.FAILED and
            self.retry_count < self.max_retries and
            not self.is_expired()
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "data": self.data,
            "priority": self.priority.name,
            "created_at": self.created_at.isoformat(),
            "user_id": self.user_id,
            "organization_id": self.organization_id,
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "source": self.source,
            "target": self.target,
            "tags": self.tags,
            "status": self.status.value,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create event from dictionary."""
        # Parse datetime fields
        created_at = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
        processed_at = None
        if data.get("processed_at"):
            processed_at = datetime.fromisoformat(data["processed_at"].replace("Z", "+00:00"))
        expires_at = None
        if data.get("expires_at"):
            expires_at = datetime.fromisoformat(data["expires_at"].replace("Z", "+00:00"))

        return cls(
            id=data["id"],
            event_type=data["event_type"],
            data=data["data"],
            priority=EventPriority[data.get("priority", "NORMAL")],
            created_at=created_at,
            user_id=data.get("user_id"),
            organization_id=data.get("organization_id"),
            trace_id=data.get("trace_id"),
            session_id=data.get("session_id"),
            source=data.get("source", "brokle-sdk"),
            target=data.get("target"),
            tags=data.get("tags", []),
            status=EventStatus(data.get("status", "pending")),
            processed_at=processed_at,
            error_message=data.get("error_message"),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            expires_at=expires_at
        )


@dataclass
class EventFilter:
    """Filter configuration for event subscriptions."""
    event_types: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    user_ids: Optional[List[str]] = None
    organization_ids: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    priority_min: Optional[EventPriority] = None
    custom_filter: Optional[Callable[[Event], bool]] = None

    def matches(self, event: Event) -> bool:
        """Check if event matches this filter."""
        # Check event types
        if self.event_types and event.event_type not in self.event_types:
            return False

        # Check tags (event must have at least one matching tag)
        if self.tags and not any(tag in event.tags for tag in self.tags):
            return False

        # Check user IDs
        if self.user_ids and event.user_id not in self.user_ids:
            return False

        # Check organization IDs
        if self.organization_ids and event.organization_id not in self.organization_ids:
            return False

        # Check sources
        if self.sources and event.source not in self.sources:
            return False

        # Check priority
        if self.priority_min and event.priority.value < self.priority_min.value:
            return False

        # Check custom filter
        if self.custom_filter and not self.custom_filter(event):
            return False

        return True


class EventSubscription:
    """Event subscription with filtering and callback."""

    def __init__(
        self,
        subscription_id: str,
        callback: Callable[[Event], Any],
        event_filter: Optional[EventFilter] = None,
        max_queue_size: int = 1000,
        batch_size: int = 1,
        batch_timeout_seconds: float = 1.0
    ):
        self.subscription_id = subscription_id
        self.callback = callback
        self.filter = event_filter or EventFilter()
        self.max_queue_size = max_queue_size
        self.batch_size = batch_size
        self.batch_timeout_seconds = batch_timeout_seconds

        # Queue and processing
        self.event_queue = deque(maxlen=max_queue_size)
        self.is_active = True
        self.last_event_at: Optional[datetime] = None

        # Statistics
        self.events_received = 0
        self.events_processed = 0
        self.events_failed = 0
        self.created_at = datetime.utcnow()

    def add_event(self, event: Event) -> bool:
        """Add event to subscription queue if it matches filter."""
        if not self.is_active:
            return False

        if not self.filter.matches(event):
            return False

        try:
            self.event_queue.append(event)
            self.events_received += 1
            self.last_event_at = datetime.utcnow()
            return True
        except Exception as e:
            logger.error(f"Failed to add event to subscription {self.subscription_id}: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get subscription statistics."""
        uptime_seconds = (datetime.utcnow() - self.created_at).total_seconds()

        return {
            "subscription_id": self.subscription_id,
            "is_active": self.is_active,
            "queue_size": len(self.event_queue),
            "max_queue_size": self.max_queue_size,
            "events_received": self.events_received,
            "events_processed": self.events_processed,
            "events_failed": self.events_failed,
            "events_per_second": self.events_received / uptime_seconds if uptime_seconds > 0 else 0,
            "success_rate": (
                self.events_processed / (self.events_processed + self.events_failed)
                if (self.events_processed + self.events_failed) > 0 else 1.0
            ),
            "last_event_at": self.last_event_at.isoformat() if self.last_event_at else None,
            "uptime_seconds": uptime_seconds
        }


class EventBus:
    """
    Enterprise event bus with real-time streaming and routing.

    Provides publish-subscribe pattern with filtering, batching,
    and reliable delivery guarantees.
    """

    def __init__(
        self,
        max_event_history: int = 10000,
        cleanup_interval_seconds: float = 300.0,
        enable_webhooks: bool = True
    ):
        self.max_event_history = max_event_history
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self.enable_webhooks = enable_webhooks

        # Event storage
        self.event_history: deque = deque(maxlen=max_event_history)
        self.pending_events: List[Event] = []

        # Subscriptions
        self.subscriptions: Dict[str, EventSubscription] = {}
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)

        # Real-time subscribers
        self.realtime_subscribers: WeakSet = WeakSet()

        # Threading
        self._lock = threading.Lock()
        self._processing_thread: Optional[threading.Thread] = None
        self._cleanup_thread: Optional[threading.Thread] = None
        self._shutdown = False

        # Statistics
        self.stats = {
            "events_published": 0,
            "events_processed": 0,
            "events_failed": 0,
            "total_subscriptions": 0,
            "active_subscriptions": 0
        }

        # Start background processing
        self._start_background_threads()

    def _start_background_threads(self) -> None:
        """Start background processing threads."""
        # Event processing thread
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            name="brokle-event-processor",
            daemon=True
        )
        self._processing_thread.start()

        # Cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            name="brokle-event-cleanup",
            daemon=True
        )
        self._cleanup_thread.start()

    def publish(
        self,
        event_type: str,
        data: Dict[str, Any],
        priority: EventPriority = EventPriority.NORMAL,
        **kwargs
    ) -> str:
        """
        Publish an event to the bus.

        Args:
            event_type: Type of event
            data: Event data payload
            priority: Event priority level
            **kwargs: Additional event attributes

        Returns:
            Event ID
        """
        event = Event(
            id=str(uuid.uuid4()),
            event_type=event_type,
            data=data,
            priority=priority,
            **kwargs
        )

        with self._lock:
            self.pending_events.append(event)
            self.event_history.append(event)
            self.stats["events_published"] += 1

        logger.debug(f"Published event {event.id} of type {event_type}")

        # Send webhook if enabled
        if self.enable_webhooks:
            try:
                # Convert to webhook event type if possible
                webhook_event_type = None
                for webhook_type in WebhookEventType:
                    if webhook_type.value == event_type:
                        webhook_event_type = webhook_type
                        break

                if webhook_event_type:
                    send_webhook_event(
                        webhook_event_type,
                        data,
                        user_id=event.user_id,
                        organization_id=event.organization_id,
                        trace_id=event.trace_id
                    )
            except Exception as e:
                logger.error(f"Failed to send webhook for event {event.id}: {e}")

        return event.id

    def subscribe(
        self,
        subscription_id: str,
        callback: Callable[[Event], Any],
        event_filter: Optional[EventFilter] = None,
        **kwargs
    ) -> EventSubscription:
        """
        Subscribe to events with filtering.

        Args:
            subscription_id: Unique subscription ID
            callback: Callback function for events
            event_filter: Filter configuration
            **kwargs: Additional subscription options

        Returns:
            EventSubscription object
        """
        subscription = EventSubscription(
            subscription_id=subscription_id,
            callback=callback,
            event_filter=event_filter,
            **kwargs
        )

        with self._lock:
            self.subscriptions[subscription_id] = subscription
            self.stats["total_subscriptions"] += 1
            self.stats["active_subscriptions"] += 1

        logger.info(f"Created subscription {subscription_id}")
        return subscription

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events.

        Args:
            subscription_id: Subscription ID to remove

        Returns:
            True if successfully unsubscribed
        """
        with self._lock:
            if subscription_id in self.subscriptions:
                subscription = self.subscriptions[subscription_id]
                subscription.is_active = False
                del self.subscriptions[subscription_id]
                self.stats["active_subscriptions"] -= 1

                logger.info(f"Removed subscription {subscription_id}")
                return True

        return False

    def add_event_handler(self, event_type: str, handler: Callable[[Event], Any]) -> None:
        """Add event handler for specific event type."""
        with self._lock:
            self.event_handlers[event_type].append(handler)

        logger.debug(f"Added handler for event type {event_type}")

    def remove_event_handler(self, event_type: str, handler: Callable[[Event], Any]) -> bool:
        """Remove event handler for specific event type."""
        with self._lock:
            if event_type in self.event_handlers:
                try:
                    self.event_handlers[event_type].remove(handler)
                    return True
                except ValueError:
                    pass

        return False

    def get_events(
        self,
        event_filter: Optional[EventFilter] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Event]:
        """
        Get historical events with filtering.

        Args:
            event_filter: Filter configuration
            limit: Maximum number of events to return
            offset: Offset for pagination

        Returns:
            List of matching events
        """
        events = list(self.event_history)

        # Apply filter
        if event_filter:
            events = [event for event in events if event_filter.matches(event)]

        # Apply pagination
        start_idx = offset
        end_idx = offset + limit
        return events[start_idx:end_idx]

    def stream_events(
        self,
        event_filter: Optional[EventFilter] = None
    ) -> AsyncIterator[Event]:
        """
        Stream events in real-time.

        Args:
            event_filter: Filter configuration

        Yields:
            Events as they are published
        """
        # This would typically be implemented with async generators
        # For now, we'll provide a placeholder
        raise NotImplementedError("Real-time streaming not yet implemented")

    def _processing_loop(self) -> None:
        """Background event processing loop."""
        logger.info("Event processing loop started")

        while not self._shutdown:
            try:
                # Get pending events
                with self._lock:
                    if not self.pending_events:
                        time.sleep(0.1)
                        continue

                    # Process events in batches
                    batch = self.pending_events[:100]
                    self.pending_events = self.pending_events[100:]

                # Process batch
                for event in batch:
                    self._process_event(event)

            except Exception as e:
                logger.error(f"Event processing error: {e}")
                time.sleep(1.0)

        logger.info("Event processing loop ended")

    def _process_event(self, event: Event) -> None:
        """Process a single event."""
        try:
            event.status = EventStatus.PROCESSING

            # Send to subscriptions
            for subscription in self.subscriptions.values():
                if subscription.add_event(event):
                    self._execute_subscription_callback(subscription, event)

            # Send to event handlers
            handlers = self.event_handlers.get(event.event_type, [])
            for handler in handlers:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Event handler error for {event.event_type}: {e}")

            # Mark as processed
            event.status = EventStatus.PROCESSED
            event.processed_at = datetime.utcnow()
            self.stats["events_processed"] += 1

        except Exception as e:
            # Mark as failed
            event.status = EventStatus.FAILED
            event.error_message = str(e)
            self.stats["events_failed"] += 1

            logger.error(f"Failed to process event {event.id}: {e}")

    def _execute_subscription_callback(self, subscription: EventSubscription, event: Event) -> None:
        """Execute subscription callback safely."""
        try:
            # Check if callback is async
            if asyncio.iscoroutinefunction(subscription.callback):
                # Schedule async callback
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(subscription.callback(event))
                except RuntimeError:
                    # No event loop running
                    logger.warning(f"Cannot execute async callback for subscription {subscription.subscription_id}")
            else:
                # Execute sync callback
                subscription.callback(event)

            subscription.events_processed += 1

        except Exception as e:
            subscription.events_failed += 1
            logger.error(f"Subscription callback error for {subscription.subscription_id}: {e}")

    def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        logger.info("Event cleanup loop started")

        while not self._shutdown:
            try:
                time.sleep(self.cleanup_interval_seconds)
                self._cleanup_expired_events()

            except Exception as e:
                logger.error(f"Cleanup error: {e}")

        logger.info("Event cleanup loop ended")

    def _cleanup_expired_events(self) -> None:
        """Clean up expired events."""
        now = datetime.utcnow()
        expired_count = 0

        # Clean up event history
        with self._lock:
            # Filter out expired events
            valid_events = []
            for event in self.event_history:
                if not event.is_expired():
                    valid_events.append(event)
                else:
                    expired_count += 1

            # Update history
            self.event_history.clear()
            self.event_history.extend(valid_events)

        if expired_count > 0:
            logger.debug(f"Cleaned up {expired_count} expired events")

    def get_statistics(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        with self._lock:
            subscription_stats = [
                sub.get_statistics()
                for sub in self.subscriptions.values()
            ]

            return {
                "events_in_history": len(self.event_history),
                "pending_events": len(self.pending_events),
                "total_events_published": self.stats["events_published"],
                "total_events_processed": self.stats["events_processed"],
                "total_events_failed": self.stats["events_failed"],
                "success_rate": (
                    self.stats["events_processed"] /
                    (self.stats["events_processed"] + self.stats["events_failed"])
                    if (self.stats["events_processed"] + self.stats["events_failed"]) > 0 else 1.0
                ),
                "total_subscriptions": self.stats["total_subscriptions"],
                "active_subscriptions": self.stats["active_subscriptions"],
                "subscriptions": subscription_stats
            }

    def shutdown(self) -> None:
        """Shutdown event bus."""
        self._shutdown = True

        # Wait for threads to finish
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=5.0)

        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5.0)

        logger.info("Event bus shutdown complete")


# Global event bus instance
_event_bus: Optional[EventBus] = None
_bus_lock = threading.Lock()


def get_event_bus() -> EventBus:
    """Get global event bus instance."""
    global _event_bus

    with _bus_lock:
        if _event_bus is None:
            _event_bus = EventBus()

        return _event_bus


def publish_event(
    event_type: str,
    data: Dict[str, Any],
    priority: EventPriority = EventPriority.NORMAL,
    **kwargs
) -> str:
    """Publish an event to the global event bus."""
    bus = get_event_bus()
    return bus.publish(event_type, data, priority, **kwargs)


def subscribe_to_events(
    subscription_id: str,
    callback: Callable[[Event], Any],
    event_filter: Optional[EventFilter] = None,
    **kwargs
) -> EventSubscription:
    """Subscribe to events on the global event bus."""
    bus = get_event_bus()
    return bus.subscribe(subscription_id, callback, event_filter, **kwargs)


def get_event_statistics() -> Dict[str, Any]:
    """Get statistics for the global event bus."""
    bus = get_event_bus()
    return bus.get_statistics()