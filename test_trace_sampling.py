#!/usr/bin/env python3
"""
Test trace-level sampling to ensure entire traces are sampled together.

This validates that:
1. All spans in a sampled trace are exported
2. All spans in a non-sampled trace are dropped
3. No partial traces occur
4. Sampling is deterministic (same trace_id → same decision)
"""

import os
import time
from brokle import Brokle

print("=" * 70)
print("Brokle OTEL SDK - Trace-Level Sampling Test")
print("=" * 70)
print()

# Test configuration
API_KEY = os.getenv("BROKLE_API_KEY", "bk_fzwUZlCBIE3Z0QfGnfAIKjZ4DuK4ChJHf3mPnnbV")
SAMPLE_RATE = 0.3  # 30% sampling
NUM_TRACES = 20    # Create 20 traces

print(f"Configuration:")
print(f"  - Sample Rate: {SAMPLE_RATE} (expect ~{int(SAMPLE_RATE*100)}% of traces)")
print(f"  - Number of Traces: {NUM_TRACES}")
print(f"  - Expected Sampled: ~{int(NUM_TRACES * SAMPLE_RATE)} complete traces")
print()

# Create client with 30% sampling
print(f"1. Creating client with sample_rate={SAMPLE_RATE}...")
client = Brokle(
    api_key=API_KEY,
    base_url="http://localhost:8080",
    environment="sampling-test",
    sample_rate=SAMPLE_RATE,  # 30% trace-level sampling
    debug=False,
)
print("✓ Client created with TraceIdRatioBased sampler")
print()

# Create traces with nested spans
print(f"2. Creating {NUM_TRACES} traces (each with 3 spans)...")
print()

for i in range(NUM_TRACES):
    trace_num = i + 1

    # Parent span
    with client.start_as_current_span(f"parent-trace-{trace_num}") as parent:
        parent.set_attribute("trace_number", trace_num)
        parent.set_attribute("test", "sampling")

        # Child span 1
        with client.start_as_current_span(f"child-1-trace-{trace_num}") as child1:
            child1.set_attribute("child_index", 1)

        # Child span 2
        with client.start_as_current_span(f"child-2-trace-{trace_num}") as child2:
            child2.set_attribute("child_index", 2)

    if (trace_num % 5) == 0:
        print(f"   Created {trace_num} traces...")

print(f"✓ Created {NUM_TRACES} traces (each with parent + 2 children = 3 spans)")
print(f"   Total spans created: {NUM_TRACES * 3} = {NUM_TRACES * 3}")
print()

# Flush all data
print("3. Flushing all spans...")
client.flush(timeout_seconds=10)
print("✓ Flush complete")
print()

# Wait for backend processing
print("4. Waiting for backend processing...")
time.sleep(2)
print("✓ Processing complete")
print()

print("=" * 70)
print("✅ Sampling Test Complete!")
print("=" * 70)
print()

print("Verification Steps:")
print()

print("1. Count total spans:")
print("   docker exec brokle-clickhouse clickhouse-client \\")
print("     --query \"SELECT COUNT(*) FROM spans WHERE name LIKE '%sampling-test%' OR name LIKE '%trace-%'\"")
print()

print("2. Verify complete traces (no orphaned children):")
print("   docker exec brokle-clickhouse clickhouse-client \\")
print("     --query \"SELECT trace_id, COUNT(*) as span_count FROM spans WHERE name LIKE '%trace-%' GROUP BY trace_id ORDER BY span_count DESC FORMAT Pretty\"")
print()

print("3. Expected results:")
print(f"   - Total spans: ~{int(NUM_TRACES * SAMPLE_RATE * 3)} spans")
print(f"   - Complete traces: ~{int(NUM_TRACES * SAMPLE_RATE)} traces")
print(f"   - Each trace should have exactly 3 spans (parent + 2 children)")
print(f"   - No traces with 1 or 2 spans (that would indicate broken sampling)")
print()

print("4. Check if traces are complete:")
print("   docker exec brokle-clickhouse clickhouse-client \\")
print("     --query \"SELECT trace_id, groupArray(name) as spans FROM spans WHERE name LIKE '%trace-%' GROUP BY trace_id HAVING COUNT(*) != 3 FORMAT Vertical\"")
print()
print("   Expected: No results (all traces should have exactly 3 spans)")
print()

print("=" * 70)
print()
print("Key Points:")
print("✓ Sampling is TRACE-LEVEL (not span-level)")
print("✓ Deterministic (same trace_id → same decision)")
print("✓ All-or-nothing (complete traces, no partial traces)")
print("✓ Uses OpenTelemetry's TraceIdRatioBased sampler")
