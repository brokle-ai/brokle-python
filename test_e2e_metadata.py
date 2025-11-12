"""
End-to-End Test - Verify Metadata Structure

Tests that brokle.project_id and brokle.environment are NOT in resource attributes
and that environment is set as span attribute instead.
"""

import time
from brokle import Brokle
from brokle.types.attributes import BrokleOtelSpanAttributes as Attrs

# Configuration
API_KEY = "bk_J4opq4VfeKP3Cg5MB6w5KgxiRcG6YoXcG79tf7xY"
BASE_URL = "http://localhost:8080"

# Initialize client
client = Brokle(
    api_key=API_KEY,
    base_url=BASE_URL,
    environment="e2e-test-python",
    release="v1.0.0-python-e2e",
    debug=True,
)

print("=" * 60)
print("Python SDK E2E Test - Metadata Verification")
print("=" * 60)

# Test 1: Simple span
print("\n=== Test 1: Simple Span ===")
with client.start_as_current_span("python-simple-span") as span:
    span.set_attribute("output", "test output")
    span.set_attribute(Attrs.USER_ID, "python-user-123")  # user.id
    span.set_attribute(Attrs.SESSION_ID, "python-session-456")  # session.id
    time.sleep(0.1)
print("✓ Simple span created")

# Test 2: LLM Generation
print("\n=== Test 2: LLM Generation ===")
with client.start_as_current_span("python-chat-gpt4") as span:
    span.set_attribute(Attrs.BROKLE_SPAN_TYPE, "generation")
    span.set_attribute(Attrs.GEN_AI_PROVIDER_NAME, "openai")
    span.set_attribute(Attrs.GEN_AI_REQUEST_MODEL, "gpt-4")
    span.set_attribute(Attrs.GEN_AI_USAGE_INPUT_TOKENS, 12)
    span.set_attribute(Attrs.GEN_AI_USAGE_OUTPUT_TOKENS, 10)
    time.sleep(0.1)
print("✓ LLM generation created")

# Test 3: Nested spans
print("\n=== Test 3: Nested Spans ===")
with client.start_as_current_span("python-parent") as parent:
    parent.set_attribute("level", "parent")

    with client.start_as_current_span("python-child-1") as child:
        child.set_attribute("level", "child")
        time.sleep(0.05)

    with client.start_as_current_span("python-child-2") as child:
        child.set_attribute("level", "child")
        time.sleep(0.05)
print("✓ Nested spans created")

# Flush
print("\n=== Flushing Spans ===")
client.flush()
print("✓ All spans flushed to backend")

print("\n" + "=" * 60)
print("Verification Instructions")
print("=" * 60)
print("\nRun this command to verify metadata:")
print("\n$ docker exec brokle-clickhouse clickhouse-client \\")
print("    --user brokle --password brokle_password \\")
print("    --query \"")
print("    SELECT name, metadata, attributes")
print("    FROM spans")
print("    WHERE attributes LIKE '%python-%'")
print("    ORDER BY start_time DESC")
print("    LIMIT 3")
print("    FORMAT Vertical\"")

print("\n✅ Expected in metadata.resourceAttributes:")
print("   • service.name (OTEL default)")
print("   • telemetry.sdk.language: 'python'")
print("   • telemetry.sdk.name: 'opentelemetry'")
print("   • telemetry.sdk.version")
print("\n❌ Should NOT be in resourceAttributes:")
print("   • brokle.project_id (removed!)")
print("   • brokle.environment (now in span attributes!)")

print("\n✅ Expected in attributes column (OTEL-native):")
print("   • brokle.environment: 'e2e-test-python'")
print("   • brokle.release: 'v1.0.0-python-e2e'")
print("   • gen_ai.* attributes (for generations)")
print("   • user.id, session.id (OTEL standard)")
print("\n💡 Note: All Brokle data stored in attributes with brokle.* namespace")
print("   (routing, cache, governance data will appear here when set)")

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)
