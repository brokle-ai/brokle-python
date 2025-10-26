#!/usr/bin/env python3
"""Debug script to capture what the OTEL SDK actually sends."""

import logging
from brokle import Brokle
from brokle.types import Attrs

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

print("Testing OTLP export with debug logging...")
print()

# Create client
client = Brokle(
    api_key="bk_fzwUZlCBIE3Z0QfGnfAIKjZ4DuK4ChJHf3mPnnbV",
    base_url="http://localhost:8080",
    environment="debug-test",
    debug=True,
)

# Create a simple span
print("Creating test span...")
with client.start_as_current_span("debug-span") as span:
    span.set_attribute("test", "value")

print("Flushing...")
client.flush()
print("Done")
