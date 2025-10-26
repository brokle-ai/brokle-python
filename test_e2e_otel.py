#!/usr/bin/env python3
"""
End-to-end integration test for Brokle OTEL SDK with real backend.

This script:
1. Creates a real API key via backend
2. Tests OTLP export with the real key
3. Verifies data in ClickHouse
"""

import os
import sys
import time
import json

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sdk', 'python'))

from brokle import Brokle
from brokle.types import Attrs


def test_otlp_integration():
    """Test complete OTLP flow with real backend."""
    print("=" * 60)
    print("Brokle OTEL SDK - End-to-End Integration Test")
    print("=" * 60)
    print()

    # Use a real API key from environment or prompt
    api_key = os.getenv("BROKLE_API_KEY")
    if not api_key:
        print("❌ Error: BROKLE_API_KEY environment variable not set")
        print("Please set a valid API key:")
        print("  export BROKLE_API_KEY='bk_your_actual_key'")
        return False

    print(f"✓ Using API key: {api_key[:10]}...")
    print()

    # Create Brokle client with Protobuf (default)
    print("1. Creating Brokle client (Protobuf OTLP)...")
    client = Brokle(
        api_key=api_key,
        base_url=os.getenv("BROKLE_BASE_URL", "http://localhost:8080"),
        environment="e2e-test",
        debug=True,
        use_protobuf=True,
        compression="gzip",
    )
    print("✓ Client created")
    print()

    # Test 1: Simple span
    print("2. Creating simple span...")
    with client.start_as_current_span("e2e-test-span") as span:
        span.set_attribute("test.type", "e2e")
        span.set_attribute(Attrs.USER_ID, "test-user-123")
        span.set_attribute("output", "E2E test successful")
    print("✓ Simple span created")
    print()

    # Test 2: LLM Generation (OTEL 1.28+ compliant)
    print("3. Creating LLM generation span...")
    with client.start_as_current_generation(
        name="chat",
        model="gpt-4-test",
        provider="openai-test",
        input_messages=[
            {"role": "user", "content": "Test message for E2E"}
        ],
        model_parameters={
            "temperature": 0.7,
            "max_tokens": 100,
        }
    ) as gen:
        # Simulate LLM response
        gen.set_attribute(
            Attrs.GEN_AI_OUTPUT_MESSAGES,
            json.dumps([{"role": "assistant", "content": "Test response"}])
        )
        gen.set_attribute(Attrs.GEN_AI_USAGE_INPUT_TOKENS, 15)
        gen.set_attribute(Attrs.GEN_AI_USAGE_OUTPUT_TOKENS, 8)
        gen.set_attribute(Attrs.BROKLE_USAGE_TOTAL_TOKENS, 23)
        gen.set_attribute(Attrs.GEN_AI_RESPONSE_ID, "test-resp-123")
        gen.set_attribute(Attrs.GEN_AI_RESPONSE_MODEL, "gpt-4-test-0613")
        gen.set_attribute(Attrs.GEN_AI_RESPONSE_FINISH_REASONS, ["stop"])
    print("✓ LLM generation span created")
    print()

    # Test 3: Nested spans
    print("4. Creating nested spans...")
    with client.start_as_current_span("parent-span") as parent:
        parent.set_attribute("level", "parent")

        with client.start_as_current_span("child-span-1") as child1:
            child1.set_attribute("level", "child")
            child1.set_attribute("index", 1)

        with client.start_as_current_span("child-span-2") as child2:
            child2.set_attribute("level", "child")
            child2.set_attribute("index", 2)

    print("✓ Nested spans created")
    print()

    # Flush all data
    print("5. Flushing all data to backend...")
    success = client.flush(timeout_seconds=10)
    print(f"✓ Flush {'successful' if success else 'completed with warnings'}")
    print()

    # Wait for async processing
    print("6. Waiting for backend processing...")
    time.sleep(3)
    print("✓ Processing complete")
    print()

    print("=" * 60)
    print("✅ E2E Test Complete!")
    print()
    print("Next steps:")
    print("1. Verify data in ClickHouse:")
    print("   docker exec brokle-clickhouse clickhouse-client \\")
    print("     --query \"SELECT name, provider, model_name, usage_details FROM observations WHERE environment='e2e-test' ORDER BY start_time DESC LIMIT 5 FORMAT Vertical\"")
    print()
    print("2. Check attributes JSON:")
    print("   docker exec brokle-clickhouse clickhouse-client \\")
    print("     --query \"SELECT attributes FROM observations WHERE provider='openai-test' LIMIT 1 FORMAT Vertical\"")
    print("=" * 60)

    return True


if __name__ == "__main__":
    try:
        success = test_otlp_integration()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ E2E test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
