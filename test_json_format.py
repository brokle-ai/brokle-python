#!/usr/bin/env python3
"""Test OTLP with JSON format instead of Protobuf."""

import os
import json
from brokle import Brokle
from brokle.types import Attrs

print("Testing OTLP with JSON format...")
print()

# Create client with JSON format
client = Brokle(
    api_key="bk_fzwUZlCBIE3Z0QfGnfAIKjZ4DuK4ChJHf3mPnnbV",
    base_url="http://localhost:8080",
    environment="json-test",
    use_protobuf=False,  # Use JSON instead of Protobuf
    compression=None,     # No compression for debugging
    debug=True,
)

print("1. Creating test span with JSON OTLP...")
with client.start_as_current_generation(
    name="chat",
    model="gpt-4",
    provider="openai",
    input_messages=[{"role": "user", "content": "Test with JSON"}],
) as gen:
    gen.set_attribute(
        Attrs.GEN_AI_OUTPUT_MESSAGES,
        json.dumps([{"role": "assistant", "content": "JSON response"}])
    )
    gen.set_attribute(Attrs.GEN_AI_USAGE_INPUT_TOKENS, 10)
    gen.set_attribute(Attrs.GEN_AI_USAGE_OUTPUT_TOKENS, 5)

print("✓ Span created")
print()

print("2. Flushing data...")
client.flush()
print("✓ Data flushed")
print()

print("=" * 60)
print("✅ JSON OTLP test complete!")
print()
print("Check data with:")
print("  docker exec brokle-clickhouse clickhouse-client --query \\")
print("    \"SELECT name, provider, model_name FROM spans ORDER BY start_time DESC LIMIT 3 FORMAT Vertical\"")
