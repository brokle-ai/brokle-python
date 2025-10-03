"""
Example: Using the Unified Telemetry Batch API

Demonstrates the new batch telemetry capabilities including:
- Structured event submission with ULIDs
- Event deduplication
- Partial failure handling
- Batch configuration
"""

import asyncio
import os

from brokle import Brokle, AsyncBrokle
from brokle.types.telemetry import (
    TelemetryEvent,
    TelemetryEventType,
    TelemetryBatchRequest,
    DeduplicationConfig,
)
from brokle._utils.ulid import generate_event_id


def example_1_legacy_telemetry():
    """Example 1: Legacy telemetry submission (still works)."""
    print("\n=== Example 1: Legacy Telemetry Submission ===")

    client = Brokle(api_key=os.getenv("BROKLE_API_KEY", "bk_test"))

    # Old way still works - automatically converted to batch format
    client.submit_telemetry({
        "method": "GET",
        "endpoint": "/api/users",
        "status_code": 200,
        "latency_ms": 45
    })

    print("✅ Legacy telemetry submitted (auto-converted to batch format)")
    print("   - Event type: event_create (default)")
    print("   - Event ID: auto-generated ULID")
    print("   - Endpoint: /v1/telemetry/batch")

    client.close()


def example_2_structured_events():
    """Example 2: New structured event submission with tracking."""
    print("\n=== Example 2: Structured Event Submission ===")

    client = Brokle(api_key=os.getenv("BROKLE_API_KEY", "bk_test"))

    # New preferred method - returns event ID for tracking
    trace_event_id = client.submit_batch_event(
        event_type="trace_create",
        payload={
            "name": "user-authentication-flow",
            "user_id": "user_123",
            "session_id": "session_456",
            "metadata": {
                "ip": "192.168.1.1",
                "user_agent": "Mozilla/5.0..."
            }
        }
    )

    print(f"✅ Trace created with ID: {trace_event_id}")
    print(f"   - Length: {len(trace_event_id)} characters (ULID)")
    print("   - Trackable across distributed systems")

    # Submit observation for the trace
    observation_event_id = client.submit_batch_event(
        event_type="observation_create",
        payload={
            "trace_id": trace_event_id,
            "type": "llm",
            "name": "OpenAI GPT-4 Call",
            "model": "gpt-4",
            "input": {"messages": [{"role": "user", "content": "Hello"}]},
            "metadata": {"temperature": 0.7}
        }
    )

    print(f"✅ Observation created with ID: {observation_event_id}")

    # Submit quality score
    score_event_id = client.submit_batch_event(
        event_type="quality_score_create",
        payload={
            "trace_id": trace_event_id,
            "observation_id": observation_event_id,
            "name": "response_quality",
            "value": 0.95,
            "data_type": "numeric"
        }
    )

    print(f"✅ Quality score created with ID: {score_event_id}")

    client.close()


def example_3_custom_batch_config():
    """Example 3: Custom batch configuration."""
    print("\n=== Example 3: Custom Batch Configuration ===")

    client = Brokle(
        api_key=os.getenv("BROKLE_API_KEY", "bk_test"),
        # Custom batch settings
        batch_max_size=200,              # Larger batches
        batch_flush_interval=2.0,        # Flush every 2 seconds
        batch_enable_deduplication=True, # Enable dedup
        batch_deduplication_ttl=7200,    # 2-hour cache
        batch_use_redis_cache=True,      # Use Redis
    )

    print("✅ Client configured with custom batch settings:")
    print(f"   - Max batch size: {client.config.batch_max_size}")
    print(f"   - Flush interval: {client.config.batch_flush_interval}s")
    print(f"   - Deduplication: {client.config.batch_enable_deduplication}")
    print(f"   - Dedup TTL: {client.config.batch_deduplication_ttl}s")

    # Submit multiple events
    for i in range(5):
        client.submit_batch_event(
            event_type="event_create",
            payload={"index": i, "data": f"event_{i}"}
        )

    print(f"✅ Submitted 5 events - will be batched together")

    # Force flush to see batching in action
    client.flush_processor(timeout=5.0)
    print("✅ Flushed processor - events sent to backend")

    client.close()


def example_4_deduplication():
    """Example 4: Event deduplication with ULID."""
    print("\n=== Example 4: Event Deduplication ===")

    client = Brokle(
        api_key=os.getenv("BROKLE_API_KEY", "bk_test"),
        batch_enable_deduplication=True,
        batch_deduplication_ttl=3600  # 1 hour
    )

    # Generate a specific event ID
    event_id = generate_event_id()
    print(f"Generated event ID: {event_id}")

    # Submit the same event multiple times with same ID
    from brokle.types.telemetry import TelemetryEvent

    event = TelemetryEvent(
        event_id=event_id,
        event_type=TelemetryEventType.TRACE_CREATE,
        payload={"name": "duplicate-test-trace"}
    )

    # First submission
    client._background_processor.submit_batch_event(event)
    print("✅ First submission - will be processed")

    # Second submission (duplicate)
    client._background_processor.submit_batch_event(event)
    print("⚠️  Second submission - will be deduplicated (same event_id)")

    # Flush and wait
    client.flush_processor(timeout=5.0)

    print("\nBackend response will show:")
    print("  - processed_events: 1")
    print("  - duplicate_events: 1")
    print("  - duplicate_event_ids: ['{event_id}']")

    client.close()


async def example_5_async_batch():
    """Example 5: Async batch submission."""
    print("\n=== Example 5: Async Batch Submission ===")

    async with AsyncBrokle(api_key=os.getenv("BROKLE_API_KEY", "bk_test")) as client:
        # Submit events asynchronously
        tasks = []
        for i in range(10):
            event_id = client.submit_batch_event(
                event_type="trace_create",
                payload={"name": f"async-trace-{i}"}
            )
            tasks.append(event_id)

        print(f"✅ Submitted {len(tasks)} events asynchronously")
        print(f"   - Event IDs: {tasks[:3]}... (showing first 3)")

        # Flush and wait
        client.flush_processor(timeout=5.0)
        print("✅ All events flushed to backend")


def example_6_error_handling():
    """Example 6: Handling partial failures."""
    print("\n=== Example 6: Partial Failure Handling ===")

    client = Brokle(api_key=os.getenv("BROKLE_API_KEY", "bk_test"))

    # Submit mix of valid and potentially invalid events
    valid_event = client.submit_batch_event(
        event_type="trace_create",
        payload={"name": "valid-trace", "user_id": "user_123"}
    )
    print(f"✅ Valid event: {valid_event}")

    # This might fail if validation is strict
    invalid_event = client.submit_batch_event(
        event_type="trace_create",
        payload={"invalid_field": "missing_name"}  # Missing required 'name'
    )
    print(f"⚠️  Potentially invalid event: {invalid_event}")

    # Flush and observe batch response
    client.flush_processor(timeout=5.0)

    # Check metrics to see failures
    metrics = client.get_processor_metrics()
    print("\nProcessor metrics:")
    print(f"  - Items processed: {metrics.get('items_processed', 0)}")
    print(f"  - Items failed: {metrics.get('items_failed', 0)}")
    print(f"  - Error rate: {metrics.get('error_rate', 0):.2%}")

    if metrics.get("items_failed", 0) > 0:
        print("\n⚠️  Some events failed - check logs for details")
        print("   Failed events are tracked separately and can be retried")

    client.close()


def example_7_monitoring():
    """Example 7: Monitoring batch telemetry."""
    print("\n=== Example 7: Batch Telemetry Monitoring ===")

    client = Brokle(api_key=os.getenv("BROKLE_API_KEY", "bk_test"))

    # Submit various events
    for i in range(20):
        client.submit_batch_event(
            event_type="event_create",
            payload={"index": i}
        )

    # Get detailed metrics
    metrics = client.get_processor_metrics()

    print("📊 Batch Processor Metrics:")
    print(f"   - Queue depth: {metrics.get('queue_depth', 0)}")
    print(f"   - Items processed: {metrics.get('items_processed', 0)}")
    print(f"   - Items failed: {metrics.get('items_failed', 0)}")
    print(f"   - Batches processed: {metrics.get('batches_processed', 0)}")
    print(f"   - Processing rate: {metrics.get('processing_rate', 0):.2f} items/sec")
    print(f"   - Error rate: {metrics.get('error_rate', 0):.2%}")
    print(f"   - Worker alive: {metrics.get('worker_alive', False)}")

    # Check health
    is_healthy = client.is_processor_healthy()
    print(f"\n🏥 Health status: {'✅ Healthy' if is_healthy else '❌ Unhealthy'}")

    client.close()


def main():
    """Run all examples."""
    print("=" * 60)
    print("Brokle SDK - Unified Telemetry Batch API Examples")
    print("=" * 60)

    # Run synchronous examples
    example_1_legacy_telemetry()
    example_2_structured_events()
    example_3_custom_batch_config()
    example_4_deduplication()
    example_6_error_handling()
    example_7_monitoring()

    # Run async example
    print("\n" + "=" * 60)
    asyncio.run(example_5_async_batch())

    print("\n" + "=" * 60)
    print("✅ All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
