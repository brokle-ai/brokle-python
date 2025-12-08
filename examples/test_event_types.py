"""
Test that event types match backend expectations.
"""

import logging
import os
import time

from brokle import Brokle

# Enable logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Set environment
os.environ["BROKLE_API_KEY"] = "bk_test"
os.environ["BROKLE_HOST"] = "http://localhost:8080"

print("\n========== EVENT TYPE COMPATIBILITY TEST ==========\n")

client = Brokle()

# Test all supported event types
event_types = [
    ("event", {"name": "generic-event", "data": "some_data"}),
    ("trace", {"name": "test-trace", "user_id": "user_123"}),
    ("span", {"name": "test-span", "type": "llm"}),
    ("quality_score", {"name": "test-score", "score": 0.95}),
]

print(f"Testing {len(event_types)} supported event types...\n")

for event_type, payload in event_types:
    try:
        event_id = client.submit_batch_event(event_type, payload)
        print(f"✅ {event_type:25} → Event ID: {event_id}")
    except Exception as e:
        print(f"❌ {event_type:25} → Error: {e}")

# Wait for background processing
print("\nWaiting for background processing...")
time.sleep(2)

# Flush
print("Flushing processor...")
client.flush_processor(timeout=5.0)

# Check metrics
metrics = client.get_processor_metrics()
print(f"\n📊 Final Metrics:")
print(f"   Events processed: {metrics.get('items_processed', 0)}")
print(f"   Events failed: {metrics.get('items_failed', 0)}")
print(f"   Batches sent: {metrics.get('batches_processed', 0)}")

if metrics.get("last_error"):
    print(f"   ⚠️  Last error: {metrics['last_error']}")
else:
    print(f"   ✅ No errors!")

client.close()

print("\n========== TEST COMPLETE ==========\n")
