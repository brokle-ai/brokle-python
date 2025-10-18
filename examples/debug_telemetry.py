"""
Debug script to test telemetry end-to-end.
"""
import os
import logging
import time
from openai import OpenAI
from brokle import wrap_openai
from brokle.observability import get_client

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set environment variables
os.environ["BROKLE_API_KEY"] = "bk_test"
os.environ["BROKLE_HOST"] = "http://localhost:8080"
os.environ["BROKLE_DEBUG"] = "true"
os.environ["BROKLE_TELEMETRY_ENABLED"] = "true"

os.environ["OPENAI_API_KEY"] = "sk-proj-testkeyforlocaldebuggingonly"

print("\n========== STARTING TELEMETRY DEBUG TEST ==========\n")

# Check if Brokle client is created
print("1. Getting Brokle client...")
brokle_client = get_client()
print(f"   Client created: {brokle_client}")
print(f"   Telemetry enabled: {brokle_client.config.telemetry_enabled}")
print(f"   API key: {brokle_client.config.api_key[:10]}...")
print(f"   Host: {brokle_client.config.host}")
print(f"   Environment: {brokle_client.config.environment}")

# Check background processor
print("\n2. Checking background processor...")
if hasattr(brokle_client, '_background_processor'):
    processor = brokle_client._background_processor
    print(f"   Processor exists: {processor}")
    print(f"   Worker alive: {processor._worker_thread and processor._worker_thread.is_alive()}")
    metrics = processor.get_metrics()
    print(f"   Metrics: {metrics}")
else:
    print("   ERROR: No background processor!")

# Wrap OpenAI client
print("\n3. Wrapping OpenAI client...")
client = OpenAI()
client = wrap_openai(client)
print(f"   Client wrapped: {hasattr(client, '_brokle_instrumented')}")
print(f"   Successful wraps: {getattr(client, '_brokle_successful_wraps', 0)}")
print(f"   Failed wraps: {getattr(client, '_brokle_failed_wraps', 0)}")

# Make API call
print("\n4. Making OpenAI API call...")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say 'Hello Brokle!' in 3 words."},
    ],
)
print(f"   Response: {response.choices[0].message.content}")

# Check processor metrics after call
print("\n5. Checking metrics after API call...")
metrics = brokle_client.get_processor_metrics()
print(f"   Queue depth: {metrics.get('queue_depth', 0)}")
print(f"   Items processed: {metrics.get('items_processed', 0)}")
print(f"   Items failed: {metrics.get('items_failed', 0)}")
print(f"   Batches processed: {metrics.get('batches_processed', 0)}")

# Wait a bit for background processing
print("\n6. Waiting for background processing...")
time.sleep(2)

# Flush and check again
print("\n7. Flushing processor...")
flushed = brokle_client.flush_processor(timeout=5.0)
print(f"   Flush successful: {flushed}")

# Final metrics
print("\n8. Final metrics...")
final_metrics = brokle_client.get_processor_metrics()
print(f"   Queue depth: {final_metrics.get('queue_depth', 0)}")
print(f"   Items processed: {final_metrics.get('items_processed', 0)}")
print(f"   Items failed: {final_metrics.get('items_failed', 0)}")
print(f"   Batches processed: {final_metrics.get('batches_processed', 0)}")
print(f"   Last error: {final_metrics.get('last_error', 'None')}")

print("\n========== TEST COMPLETE ==========\n")
