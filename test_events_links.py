#!/usr/bin/env python3
"""
Test OTEL Events and Links Arrays

Validates that events_* and links_* arrays in ClickHouse work correctly.

Usage:
    python test_events_links.py
"""

import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brokle import Brokle
from brokle.types.attributes import BrokleOtelSpanAttributes as Attrs
from opentelemetry.trace import SpanKind

# Configuration
API_KEY = os.getenv("BROKLE_API_KEY", "bk_SZJvBQDr9brY80Ln1ceNtGZMoSNc175rs3gXbnLK")
BASE_URL = os.getenv("BROKLE_BASE_URL", "http://localhost:8080")

# Color codes
GREEN = '\033[92m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
RESET = '\033[0m'


def print_header(text):
    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}{text:^80}{RESET}")
    print(f"{BLUE}{'='*80}{RESET}\n")


def print_info(text):
    print(f"{YELLOW}ℹ️  {text}{RESET}")


def print_success(text):
    print(f"{GREEN}✅ {text}{RESET}")


def test_events_array():
    """Test OTEL Events array population"""
    print_header("Test 1: OTEL Events Array")

    client = Brokle(api_key=API_KEY, base_url=BASE_URL, debug=True)

    try:
        attrs = {
            Attrs.BROKLE_SPAN_TYPE: "generation",
            Attrs.GEN_AI_PROVIDER_NAME: "openai",
            Attrs.GEN_AI_REQUEST_MODEL: "gpt-4",
        }

        with client.start_as_current_span(
            "streaming-generation",
            as_type="generation",
            attributes=attrs
        ) as span:
            # Simulate streaming with events
            chunks = ["Hello", " ", "World", "!", " How", " can", " I", " help?"]

            for i, chunk in enumerate(chunks):
                # Add event for each chunk
                span.add_event(
                    name=f"stream_chunk_{i}",
                    attributes={
                        "chunk.index": i,
                        "chunk.content": chunk,
                        "chunk.finish": i == len(chunks) - 1
                    }
                )
                print_info(f"Added event {i}: '{chunk}'")

            # Add final event
            span.add_event(
                name="stream_complete",
                attributes={
                    "total_chunks": len(chunks),
                    "full_response": "".join(chunks)
                }
            )

            # Get span ID
            span_context = span.get_span_context()
            span_id = format(span_context.span_id, '016x')

            print_success(f"Created span with {len(chunks) + 1} events: {span_id}")

        client.flush()
        time.sleep(3)

        print_info(f"\nValidate with:")
        print_info(f"docker exec brokle-clickhouse clickhouse-client --query \"SELECT span_id, events_name, events_attributes FROM spans WHERE span_id = '{span_id}' FORMAT Vertical\"")

        return span_id

    finally:
        client.close()


def test_links_array():
    """Test OTEL Links array population"""
    print_header("Test 2: OTEL Links Array")

    client = Brokle(api_key=API_KEY, base_url=BASE_URL, debug=True)

    try:
        # Create first trace (parent job)
        attrs1 = {Attrs.BROKLE_SPAN_TYPE: "span"}

        with client.start_as_current_span(
            "batch-parent",
            as_type="span",
            attributes=attrs1
        ) as parent_span:
            parent_context = parent_span.get_span_context()
            parent_trace_id = format(parent_context.trace_id, '032x')
            parent_span_id = format(parent_context.span_id, '016x')

            print_info(f"Created parent trace: {parent_trace_id}")
            print_info(f"Parent span: {parent_span_id}")

        client.flush()
        time.sleep(1)

        # Create second trace that links to first (child job)
        from opentelemetry.trace import Link
        from opentelemetry.trace.span import SpanContext, TraceFlags

        # Create link to parent
        parent_span_context = SpanContext(
            trace_id=int(parent_trace_id, 16),
            span_id=int(parent_span_id, 16),
            is_remote=True,
            trace_flags=TraceFlags(0x01)
        )

        link = Link(
            context=parent_span_context,
            attributes={
                "link.type": "follows_from",
                "link.description": "Batch child job"
            }
        )

        attrs2 = {Attrs.BROKLE_SPAN_TYPE: "span"}

        with client.start_as_current_span(
            "batch-child",
            as_type="span",
            attributes=attrs2,
            links=[link]  # Link to parent trace
        ) as child_span:
            child_context = child_span.get_span_context()
            child_span_id = format(child_context.span_id, '016x')
            child_trace_id = format(child_context.trace_id, '032x')

            print_success(f"Created child trace: {child_trace_id}")
            print_success(f"Child span: {child_span_id}")
            print_success(f"Linked to parent: {parent_trace_id}")

        client.flush()
        time.sleep(3)

        print_info(f"\nValidate with:")
        print_info(f"docker exec brokle-clickhouse clickhouse-client --query \"SELECT span_id, links_trace_id, links_span_id, links_attributes FROM spans WHERE span_id = '{child_span_id}' FORMAT Vertical\"")

        return child_span_id

    finally:
        client.close()


def main():
    print_header("OTEL Events & Links Array Validation")

    print(f"{YELLOW}Note: These arrays are typically empty for basic spans.{RESET}")
    print(f"{YELLOW}This test validates they work when needed for advanced features.{RESET}\n")

    # Test 1: Events
    events_span_id = test_events_array()

    # Test 2: Links
    links_span_id = test_links_array()

    print_header("Validation Commands")

    print(f"""
Run these commands to verify:

1. Check Events (streaming chunks):
   docker exec brokle-clickhouse clickhouse-client --query \\
     "SELECT span_id, arraySize(events_name) as event_count, events_name, \\
      events_timestamp, events_attributes FROM spans \\
      WHERE span_id = '{events_span_id}' FORMAT Vertical"

   Expected: event_count = 9, events_name = ['stream_chunk_0', 'stream_chunk_1', ...]

2. Check Links (cross-trace references):
   docker exec brokle-clickhouse clickhouse-client --query \\
     "SELECT span_id, links_trace_id, links_span_id, \\
      links_attributes FROM spans \\
      WHERE span_id = '{links_span_id}' FORMAT Vertical"

   Expected: links_trace_id has parent trace ID, links_attributes has link metadata

3. Verify Arrays Not Empty:
   docker exec brokle-clickhouse clickhouse-client --query \\
     "SELECT count() as spans_with_events FROM spans \\
      WHERE arraySize(events_name) > 0"

   Expected: spans_with_events >= 1

   docker exec brokle-clickhouse clickhouse-client --query \\
     "SELECT count() as spans_with_links FROM spans \\
      WHERE arraySize(links_trace_id) > 0"

   Expected: spans_with_links >= 1
""")


if __name__ == "__main__":
    main()
