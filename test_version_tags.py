"""
Simple Test - Verify Version and Tags

Tests that version and tags are properly set and visible in span attributes.
"""

import time
from brokle import Brokle
from brokle.types.attributes import BrokleOtelSpanAttributes as Attrs

# Configuration
API_KEY = "bk_SZJvBQDr9brY80Ln1ceNtGZMoSNc175rs3gXbnLK"
BASE_URL = "http://localhost:8080"

# Initialize client with version
client = Brokle(
    api_key=API_KEY,
    base_url=BASE_URL,
    environment="test-version-tags",
    release="v2.5.0",  # Application version
    debug=True,
)

print("=" * 60)
print("Python SDK Test - Version and Tags")
print("=" * 60)

# Test 1: Span with version and tags
print("\n=== Test 1: Span with Tags ===")
with client.start_as_current_span("test-span-with-tags") as span:
    # Set some tags
    span.update_trace(tags=["production", "critical", "api-v2"])
    span.set_attribute("output", "test output with tags")
    time.sleep(0.1)
print("✓ Span with tags created")

# Test 2: Span with custom metadata
print("\n=== Test 2: Span with Metadata ===")
with client.start_as_current_span("test-span-with-metadata") as span:
    # Set tags and metadata together
    span.update_trace(
        tags=["experiment", "ml-model"],
        metadata={"experiment_id": "exp-123", "model": "gpt-4", "score": 0.95},
    )
    span.set_attribute("output", "test with metadata")
    time.sleep(0.1)
print("✓ Span with metadata created")

# Test 3: Multiple spans with same trace
print("\n=== Test 3: Multiple Spans Same Trace ===")
with client.start_as_current_span("parent-span") as parent:
    parent.update_trace(tags=["multi-step", "workflow"])
    parent.set_attribute("step", "parent")

    with client.start_as_current_span("child-span-1") as child:
        # Child inherits trace tags
        child.set_attribute("step", "child-1")
        time.sleep(0.05)

    with client.start_as_current_span("child-span-2") as child:
        # Add more tags to the trace
        child.update_trace(tags=["checkpoint"])
        child.set_attribute("step", "child-2")
        time.sleep(0.05)

print("✓ Multiple spans with tags created")

# Test 4: LLM generation with version tracking
print("\n=== Test 4: LLM Generation with Version ===")
with client.start_as_current_span("llm-generation-versioned") as span:
    span.set_attribute(Attrs.BROKLE_SPAN_TYPE, "generation")
    span.set_attribute(Attrs.GEN_AI_PROVIDER_NAME, "openai")
    span.set_attribute(Attrs.GEN_AI_REQUEST_MODEL, "gpt-4")
    span.update_trace(
        tags=["llm", "gpt4", "production"],
        metadata={"prompt_template": "v3", "deployment": "us-east-1"},
    )
    time.sleep(0.1)
print("✓ LLM generation with version created")

# Flush
print("\n=== Flushing Spans ===")
client.flush()
print("✓ All spans flushed to backend")

print("\n" + "=" * 60)
print("Verification Instructions")
print("=" * 60)
print("\nRun this command to verify version and tags:")
print("\n$ docker exec brokle-clickhouse clickhouse-client \\")
print("    --user brokle --password brokle_password \\")
print('    --query "')
print("    SELECT name, attributes")
print("    FROM spans")
print("    WHERE attributes LIKE '%test-%'")
print("    ORDER BY start_time DESC")
print("    LIMIT 5")
print('    FORMAT Vertical"')

print("\n✅ Expected in attributes column:")
print("   • brokle.version or version: 'v2.5.0'")
print("   • brokle.environment: 'test-version-tags'")
print("   • brokle.trace.tags or tags: ['production', 'critical', 'api-v2', ...]")
print("   • brokle.trace.metadata or metadata: {experiment_id, model, score, ...}")

print("\n💡 Notes:")
print("   • Version set globally via release parameter in Brokle()")
print("   • Tags are additive - can add more via update_trace()")
print("   • Metadata merges with existing metadata")
print("   • All values stored in OTEL-native attributes column")

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)
