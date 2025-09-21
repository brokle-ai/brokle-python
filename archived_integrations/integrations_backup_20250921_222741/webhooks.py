"""
Enterprise-grade webhook system for Brokle SDK.

Provides reliable webhook delivery with retry logic, authentication,
and comprehensive event management.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
from urllib.parse import urlparse

import httpx
import backoff

logger = logging.getLogger(__name__)


class WebhookEventType(Enum):
    """Types of webhook events."""
    # Observability events
    TRACE_CREATED = "trace.created"
    TRACE_COMPLETED = "trace.completed"
    SPAN_CREATED = "span.created"
    SPAN_COMPLETED = "span.completed"

    # Evaluation events
    EVALUATION_STARTED = "evaluation.started"
    EVALUATION_COMPLETED = "evaluation.completed"
    EVALUATION_FAILED = "evaluation.failed"

    # Cost and usage events
    COST_THRESHOLD_EXCEEDED = "cost.threshold_exceeded"
    USAGE_LIMIT_REACHED = "usage.limit_reached"
    QUOTA_WARNING = "quota.warning"

    # Quality events
    QUALITY_SCORE_LOW = "quality.score_low"
    QUALITY_ALERT = "quality.alert"

    # System events
    API_ERROR = "api.error"
    RATE_LIMIT_EXCEEDED = "rate_limit.exceeded"
    SYSTEM_ALERT = "system.alert"

    # Custom events
    CUSTOM_EVENT = "custom.event"


class WebhookStatus(Enum):
    """Webhook delivery status."""
    PENDING = "pending"
    DELIVERING = "delivering"
    DELIVERED = "delivered"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class WebhookEvent:
    """Webhook event data structure."""
    id: str
    event_type: WebhookEventType
    data: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Metadata
    source: str = "brokle-sdk"
    version: str = "1.0"
    user_id: Optional[str] = None
    organization_id: Optional[str] = None
    trace_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "data": self.data,
            "created_at": self.created_at.isoformat(),
            "source": self.source,
            "version": self.version,
            "user_id": self.user_id,
            "organization_id": self.organization_id,
            "trace_id": self.trace_id
        }


@dataclass
class WebhookEndpoint:
    """Webhook endpoint configuration."""
    url: str
    events: List[WebhookEventType]
    secret: Optional[str] = None
    enabled: bool = True

    # Retry configuration
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    retry_backoff_multiplier: float = 2.0
    max_retry_delay_seconds: float = 60.0
    timeout_seconds: float = 30.0

    # Filtering
    filters: Dict[str, Any] = field(default_factory=dict)

    # Headers
    custom_headers: Dict[str, str] = field(default_factory=dict)

    # Metadata
    name: Optional[str] = None
    description: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Validate endpoint configuration."""
        parsed_url = urlparse(self.url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError(f"Invalid webhook URL: {self.url}")

        if parsed_url.scheme not in ["http", "https"]:
            raise ValueError(f"Webhook URL must use HTTP or HTTPS: {self.url}")

    def matches_event(self, event: WebhookEvent) -> bool:
        """Check if endpoint should receive this event."""
        if not self.enabled:
            return False

        # Check event type
        if event.event_type not in self.events:
            return False

        # Apply filters
        for filter_key, filter_value in self.filters.items():
            event_value = event.data.get(filter_key)

            if isinstance(filter_value, list):
                if event_value not in filter_value:
                    return False
            elif event_value != filter_value:
                return False

        return True


@dataclass
class WebhookDelivery:
    """Webhook delivery attempt record."""
    id: str
    endpoint_url: str
    event: WebhookEvent
    status: WebhookStatus = WebhookStatus.PENDING

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    delivered_at: Optional[datetime] = None
    next_attempt_at: Optional[datetime] = None

    # Attempts
    attempt_count: int = 0
    max_attempts: int = 3

    # Response data
    response_status_code: Optional[int] = None
    response_body: Optional[str] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        """Initialize delivery."""
        if not self.id:
            self.id = str(uuid.uuid4())

        if self.next_attempt_at is None:
            self.next_attempt_at = self.created_at

    def is_due_for_delivery(self) -> bool:
        """Check if delivery is due."""
        if self.status != WebhookStatus.PENDING:
            return False

        if self.next_attempt_at is None:
            return True

        return datetime.utcnow() >= self.next_attempt_at

    def should_retry(self) -> bool:
        """Check if delivery should be retried."""
        return (
            self.status == WebhookStatus.FAILED and
            self.attempt_count < self.max_attempts
        )

    def schedule_retry(self, delay_seconds: float) -> None:
        """Schedule next retry attempt."""
        self.next_attempt_at = datetime.utcnow() + timedelta(seconds=delay_seconds)
        self.status = WebhookStatus.PENDING


class WebhookManager:
    """
    Enterprise webhook management system.

    Provides reliable webhook delivery with retry logic, authentication,
    and comprehensive monitoring.
    """

    def __init__(
        self,
        max_pending_deliveries: int = 10000,
        delivery_timeout_hours: int = 24,
        cleanup_interval_hours: int = 6
    ):
        self.max_pending_deliveries = max_pending_deliveries
        self.delivery_timeout_hours = delivery_timeout_hours
        self.cleanup_interval_hours = cleanup_interval_hours

        # Storage
        self.endpoints: Dict[str, WebhookEndpoint] = {}
        self.pending_deliveries: List[WebhookDelivery] = []
        self.completed_deliveries: List[WebhookDelivery] = []

        # HTTP client
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=5)
        )

        # Event handlers
        self.event_handlers: Dict[WebhookEventType, List[Callable]] = {}

        # Background task control
        self._delivery_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown = False

    def register_endpoint(
        self,
        endpoint_id: str,
        url: str,
        events: List[WebhookEventType],
        **kwargs
    ) -> WebhookEndpoint:
        """Register a new webhook endpoint."""
        endpoint = WebhookEndpoint(
            url=url,
            events=events,
            **kwargs
        )

        self.endpoints[endpoint_id] = endpoint
        logger.info(f"Registered webhook endpoint {endpoint_id}: {url}")

        return endpoint

    def unregister_endpoint(self, endpoint_id: str) -> bool:
        """Unregister a webhook endpoint."""
        if endpoint_id in self.endpoints:
            del self.endpoints[endpoint_id]
            logger.info(f"Unregistered webhook endpoint {endpoint_id}")
            return True
        return False

    def update_endpoint(
        self,
        endpoint_id: str,
        **updates
    ) -> Optional[WebhookEndpoint]:
        """Update webhook endpoint configuration."""
        if endpoint_id not in self.endpoints:
            return None

        endpoint = self.endpoints[endpoint_id]

        for key, value in updates.items():
            if hasattr(endpoint, key):
                setattr(endpoint, key, value)

        logger.info(f"Updated webhook endpoint {endpoint_id}")
        return endpoint

    def send_event(
        self,
        event_type: WebhookEventType,
        data: Dict[str, Any],
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        trace_id: Optional[str] = None
    ) -> str:
        """Send webhook event to all matching endpoints."""
        event = WebhookEvent(
            id=str(uuid.uuid4()),
            event_type=event_type,
            data=data,
            user_id=user_id,
            organization_id=organization_id,
            trace_id=trace_id
        )

        # Find matching endpoints
        matching_endpoints = [
            (endpoint_id, endpoint)
            for endpoint_id, endpoint in self.endpoints.items()
            if endpoint.matches_event(event)
        ]

        if not matching_endpoints:
            logger.debug(f"No endpoints match event {event_type.value}")
            return event.id

        # Create deliveries
        for endpoint_id, endpoint in matching_endpoints:
            delivery = WebhookDelivery(
                id=str(uuid.uuid4()),
                endpoint_url=endpoint.url,
                event=event,
                max_attempts=endpoint.max_retries + 1
            )

            self.pending_deliveries.append(delivery)
            logger.debug(f"Queued webhook delivery {delivery.id} to {endpoint.url}")

        # Trigger delivery if not running
        if not self._delivery_task or self._delivery_task.done():
            self._start_delivery_task()

        return event.id

    def _start_delivery_task(self) -> None:
        """Start background delivery task."""
        if self._shutdown:
            return

        try:
            # Get current event loop
            loop = asyncio.get_running_loop()
            self._delivery_task = loop.create_task(self._delivery_loop())
        except RuntimeError:
            # No event loop running, skip
            logger.debug("No event loop running, skipping delivery task")

    async def _delivery_loop(self) -> None:
        """Background delivery loop."""
        logger.info("Webhook delivery loop started")

        while not self._shutdown and self.pending_deliveries:
            try:
                # Find deliveries due for delivery
                due_deliveries = [
                    delivery for delivery in self.pending_deliveries
                    if delivery.is_due_for_delivery()
                ]

                if not due_deliveries:
                    await asyncio.sleep(1.0)
                    continue

                # Process deliveries concurrently
                tasks = [
                    self._deliver_webhook(delivery)
                    for delivery in due_deliveries[:10]  # Limit concurrency
                ]

                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

                # Clean up completed deliveries
                self._cleanup_deliveries()

            except Exception as e:
                logger.error(f"Delivery loop error: {e}")
                await asyncio.sleep(5.0)

        logger.info("Webhook delivery loop ended")

    async def _deliver_webhook(self, delivery: WebhookDelivery) -> None:
        """Deliver a single webhook."""
        if delivery.status != WebhookStatus.PENDING:
            return

        delivery.status = WebhookStatus.DELIVERING
        delivery.attempt_count += 1

        # Find endpoint configuration
        endpoint = None
        for ep in self.endpoints.values():
            if ep.url == delivery.endpoint_url:
                endpoint = ep
                break

        if not endpoint:
            delivery.status = WebhookStatus.FAILED
            delivery.error_message = "Endpoint configuration not found"
            return

        try:
            # Prepare payload
            payload = delivery.event.to_dict()
            payload_json = json.dumps(payload, separators=(',', ':'))

            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "Brokle-Webhook/1.0",
                "X-Brokle-Event-Type": delivery.event.event_type.value,
                "X-Brokle-Event-Id": delivery.event.id,
                "X-Brokle-Delivery-Id": delivery.id,
                **endpoint.custom_headers
            }

            # Add signature if secret is configured
            if endpoint.secret:
                signature = self._generate_signature(payload_json, endpoint.secret)
                headers["X-Brokle-Signature"] = signature

            # Make HTTP request
            response = await self.http_client.post(
                endpoint.url,
                content=payload_json,
                headers=headers,
                timeout=endpoint.timeout_seconds
            )

            # Update delivery record
            delivery.response_status_code = response.status_code
            delivery.response_body = response.text[:1000]  # Limit response body size

            if 200 <= response.status_code < 300:
                delivery.status = WebhookStatus.DELIVERED
                delivery.delivered_at = datetime.utcnow()
                logger.debug(f"Webhook delivered successfully: {delivery.id}")
            else:
                delivery.status = WebhookStatus.FAILED
                delivery.error_message = f"HTTP {response.status_code}: {response.text[:200]}"

                if delivery.should_retry():
                    self._schedule_retry(delivery, endpoint)

                logger.warning(f"Webhook delivery failed: {delivery.id} - {delivery.error_message}")

        except Exception as e:
            delivery.status = WebhookStatus.FAILED
            delivery.error_message = str(e)

            if delivery.should_retry():
                self._schedule_retry(delivery, endpoint)

            logger.error(f"Webhook delivery error: {delivery.id} - {e}")

    def _schedule_retry(self, delivery: WebhookDelivery, endpoint: WebhookEndpoint) -> None:
        """Schedule delivery retry."""
        # Calculate retry delay
        delay = endpoint.retry_delay_seconds * (
            endpoint.retry_backoff_multiplier ** (delivery.attempt_count - 1)
        )
        delay = min(delay, endpoint.max_retry_delay_seconds)

        delivery.schedule_retry(delay)
        logger.info(f"Scheduled retry for delivery {delivery.id} in {delay:.1f}s")

    def _generate_signature(self, payload: str, secret: str) -> str:
        """Generate HMAC signature for webhook payload."""
        signature = hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"

    def _cleanup_deliveries(self) -> None:
        """Clean up old and completed deliveries."""
        now = datetime.utcnow()
        timeout_threshold = now - timedelta(hours=self.delivery_timeout_hours)

        # Move completed deliveries
        completed = []
        remaining = []

        for delivery in self.pending_deliveries:
            if delivery.status in [WebhookStatus.DELIVERED, WebhookStatus.FAILED]:
                if delivery.created_at < timeout_threshold or not delivery.should_retry():
                    completed.append(delivery)
                    continue
            elif delivery.created_at < timeout_threshold:
                delivery.status = WebhookStatus.EXPIRED
                completed.append(delivery)
                continue

            remaining.append(delivery)

        self.pending_deliveries = remaining
        self.completed_deliveries.extend(completed)

        # Limit completed deliveries history
        if len(self.completed_deliveries) > 1000:
            self.completed_deliveries = self.completed_deliveries[-1000:]

    def get_delivery_stats(self) -> Dict[str, Any]:
        """Get webhook delivery statistics."""
        total_deliveries = len(self.pending_deliveries) + len(self.completed_deliveries)

        status_counts = {}
        for delivery in self.pending_deliveries + self.completed_deliveries:
            status = delivery.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        success_rate = 0.0
        if total_deliveries > 0:
            delivered_count = status_counts.get("delivered", 0)
            success_rate = delivered_count / total_deliveries

        return {
            "total_deliveries": total_deliveries,
            "pending_deliveries": len(self.pending_deliveries),
            "completed_deliveries": len(self.completed_deliveries),
            "status_counts": status_counts,
            "success_rate": success_rate,
            "registered_endpoints": len(self.endpoints)
        }

    def get_endpoint_stats(self, endpoint_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific endpoint."""
        if endpoint_id not in self.endpoints:
            return None

        endpoint = self.endpoints[endpoint_id]

        # Count deliveries for this endpoint
        endpoint_deliveries = [
            d for d in self.pending_deliveries + self.completed_deliveries
            if d.endpoint_url == endpoint.url
        ]

        status_counts = {}
        for delivery in endpoint_deliveries:
            status = delivery.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        success_rate = 0.0
        if endpoint_deliveries:
            delivered_count = status_counts.get("delivered", 0)
            success_rate = delivered_count / len(endpoint_deliveries)

        return {
            "endpoint_id": endpoint_id,
            "url": endpoint.url,
            "enabled": endpoint.enabled,
            "events": [e.value for e in endpoint.events],
            "total_deliveries": len(endpoint_deliveries),
            "status_counts": status_counts,
            "success_rate": success_rate
        }

    async def shutdown(self) -> None:
        """Shutdown webhook manager."""
        self._shutdown = True

        # Cancel background tasks
        if self._delivery_task and not self._delivery_task.done():
            self._delivery_task.cancel()
            try:
                await self._delivery_task
            except asyncio.CancelledError:
                pass

        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Close HTTP client
        await self.http_client.aclose()

        logger.info("Webhook manager shutdown complete")


# Global webhook manager instance
_webhook_manager: Optional[WebhookManager] = None


def get_webhook_manager() -> WebhookManager:
    """Get global webhook manager instance."""
    global _webhook_manager

    if _webhook_manager is None:
        _webhook_manager = WebhookManager()

    return _webhook_manager


def register_webhook(
    endpoint_id: str,
    url: str,
    events: List[Union[str, WebhookEventType]],
    **kwargs
) -> WebhookEndpoint:
    """Register a webhook endpoint."""
    manager = get_webhook_manager()

    # Convert string events to enum
    event_enums = []
    for event in events:
        if isinstance(event, str):
            try:
                event_enums.append(WebhookEventType(event))
            except ValueError:
                logger.warning(f"Unknown webhook event type: {event}")
        else:
            event_enums.append(event)

    return manager.register_endpoint(endpoint_id, url, event_enums, **kwargs)


def send_webhook_event(
    event_type: Union[str, WebhookEventType],
    data: Dict[str, Any],
    **kwargs
) -> str:
    """Send a webhook event."""
    manager = get_webhook_manager()

    if isinstance(event_type, str):
        event_type = WebhookEventType(event_type)

    return manager.send_event(event_type, data, **kwargs)


def get_webhook_stats() -> Dict[str, Any]:
    """Get webhook delivery statistics."""
    manager = get_webhook_manager()
    return manager.get_delivery_stats()