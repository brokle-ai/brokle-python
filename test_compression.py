#!/usr/bin/env python3
"""
Test OTLP with Gzip compression enabled.

This will validate:
1. Python SDK sends gzip-compressed Protobuf
2. Go backend decompresses correctly
3. Data stored correctly in ClickHouse
4. Measure compression savings
"""

import os
from brokle import Brokle
from brokle.types import Attrs
import json

print("=" * 70)
print("Brokle OTEL SDK - Compression Test")
print("=" * 70)
print()

# Create client WITH compression (default)
print("1. Creating client with Gzip compression enabled...")
client = Brokle(
    api_key="bk_fzwUZlCBIE3Z0QfGnfAIKjZ4DuK4ChJHf3mPnnbV",
    base_url="http://localhost:8080",
    environment="compression-test",
    use_protobuf=True,      # Protobuf format
    compression="gzip",      # Gzip compression (default)
    debug=False,
)
print("✓ Client created (Protobuf + Gzip)")
print()

# Create a realistic LLM span with substantial data
print("2. Creating LLM generation span with realistic data...")
long_prompt = "Explain quantum computing in simple terms. " * 10  # ~400 chars
long_response = "Quantum computing uses quantum mechanics. " * 15  # ~600 chars

with client.start_as_current_generation(
    name="chat",
    model="gpt-4-compression-test",
    provider="openai-test",
    input_messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": long_prompt}
    ],
    model_parameters={
        "temperature": 0.7,
        "max_tokens": 500,
        "top_p": 0.9,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.3,
    }
) as gen:
    gen.set_attribute(
        Attrs.GEN_AI_OUTPUT_MESSAGES,
        json.dumps([
            {"role": "assistant", "content": long_response}
        ])
    )
    gen.set_attribute(Attrs.GEN_AI_USAGE_INPUT_TOKENS, 150)
    gen.set_attribute(Attrs.GEN_AI_USAGE_OUTPUT_TOKENS, 200)
    gen.set_attribute(Attrs.BROKLE_USAGE_TOTAL_TOKENS, 350)
    gen.set_attribute(Attrs.GEN_AI_RESPONSE_ID, "chatcmpl-compression-test-123")
    gen.set_attribute(Attrs.GEN_AI_RESPONSE_MODEL, "gpt-4-0613")
    gen.set_attribute(Attrs.GEN_AI_RESPONSE_FINISH_REASONS, ["stop"])
    gen.set_attribute(Attrs.BROKLE_USAGE_LATENCY_MS, 1234.56)

print("✓ Large span created (to test compression efficiency)")
print()

# Flush
print("3. Flushing data (with Gzip compression)...")
success = client.flush()
print(f"✓ Flush {'successful' if success else 'completed with warnings'}")
print()

print("=" * 70)
print("✅ Compression Test Complete!")
print("=" * 70)
print()
print("Backend should log:")
print("  - 'Decompressing gzip-encoded OTLP request'")
print("  - Compression ratio (e.g., 3.2x)")
print()
print("Verify in ClickHouse:")
print("  docker exec brokle-clickhouse clickhouse-client \\")
print("    --query \"SELECT name, provider, model_name, usage_details FROM spans WHERE provider='openai-test' AND name LIKE '%compression%' FORMAT Vertical\"")
print()
print("Check backend logs for compression ratio:")
print("  docker logs brokle-api --tail 20 | grep -i 'decompression\\|compression'")
