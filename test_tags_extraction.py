#!/usr/bin/env python3
"""
Simple test to verify tags extraction in backend.

This test:
1. Sends a span with tags via Brokle SDK
2. Waits for backend processing
3. Queries ClickHouse to verify tags were extracted
4. Validates tags are stored correctly in traces table

Requirements:
- Backend running: make dev-server
- ClickHouse running: docker-compose up clickhouse
"""

import sys
import time
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import clickhouse_connect
except ImportError:
    print("❌ clickhouse-connect not installed")
    print("Install with: pip install clickhouse-connect")
    sys.exit(1)

from brokle import Brokle
from brokle.decorators import observe

# Configuration
API_KEY = "bk_SZJvBQDr9brY80Ln1ceNtGZMoSNc175rs3gXbnLK"
BASE_URL = "http://localhost:8080"

CLICKHOUSE_HOST = "localhost"
CLICKHOUSE_PORT = 8123
CLICKHOUSE_USER = "brokle"
CLICKHOUSE_PASSWORD = "brokle_password"
CLICKHOUSE_DATABASE = "default"


def get_clickhouse_client():
    """Get ClickHouse client connection."""
    try:
        client = clickhouse_connect.get_client(
            host=CLICKHOUSE_HOST,
            port=CLICKHOUSE_PORT,
            username=CLICKHOUSE_USER,
            password=CLICKHOUSE_PASSWORD,
            database=CLICKHOUSE_DATABASE,
        )
        return client
    except Exception as e:
        print(f"❌ Failed to connect to ClickHouse: {e}")
        print("Make sure ClickHouse is running: docker-compose up clickhouse")
        sys.exit(1)


def test_tags_extraction():
    """Test tags extraction from SDK to backend to ClickHouse."""
    print("\n" + "=" * 70)
    print("🧪 Testing Tags Extraction: SDK → Backend → ClickHouse")
    print("=" * 70 + "\n")

    # Initialize Brokle client
    print("📡 Initializing Brokle client...")
    brokle = Brokle(
        api_key=API_KEY,
        base_url=BASE_URL,
        environment="tags-test",
        flush_at=1,  # Flush immediately
        flush_interval=0.5,
    )
    print(f"✅ Connected to {BASE_URL}\n")

    # Create span with tags using decorator
    print("📝 Creating span with tags...")
    tags = ["backend-test", "tags-extraction", "python-sdk"]
    print(f"   Tags: {tags}")

    @observe(
        name="test-tags-extraction",
        as_type="span",
        tags=tags,
        session_id="test-session-tags",
        user_id="test-user-tags",
    )
    def test_operation():
        return "Tags extraction test complete"

    # Call the function to create the span
    result = test_operation()

    # Get trace ID from current context
    # Since we can't easily access trace_id from decorator, we'll query by tags
    trace_id = None
    span_id = None

    print(f"✅ Span created with tags")
    print(f"   Tags sent: {tags}\n")

    # Flush telemetry
    print("⏳ Flushing telemetry to backend...")
    brokle.flush()
    print("✅ Telemetry flushed\n")

    # Wait for backend processing
    print("⏳ Waiting 3 seconds for backend processing...")
    time.sleep(3)

    # Query ClickHouse
    print("🔍 Querying ClickHouse traces table...")
    ch_client = get_clickhouse_client()

    query = """
        SELECT
            trace_id,
            name,
            tags,
            user_id,
            session_id,
            environment
        FROM traces
        WHERE name = 'test-tags-extraction'
        AND environment = 'tags-test'
        ORDER BY created_at DESC
        LIMIT 1
    """

    result = ch_client.query(query)

    if result.row_count == 0:
        print("❌ Trace not found in ClickHouse")
        print("   Searched for: name='test-tags-extraction', environment='tags-test'")
        print("\n💡 Possible issues:")
        print("   1. Backend not running (make dev-server)")
        print("   2. Backend not processing telemetry")
        print("   3. ClickHouse migrations not up to date (make migrate-up)")
        return False

    row = result.first_row
    trace_id = row[0]
    print("✅ Trace found in ClickHouse")
    print(f"   Trace ID: {trace_id}\n")

    # Validate fields
    print("📊 Trace Data:")
    print(f"   trace_id: {row[0]}")
    print(f"   name: {row[1]}")
    print(f"   tags: {row[2]}")
    print(f"   user_id: {row[3]}")
    print(f"   session_id: {row[4]}")
    print(f"   environment: {row[5]}")
    print()

    # Validate tags
    extracted_tags = row[2]
    success = True

    if extracted_tags is None or len(extracted_tags) == 0:
        print("❌ TAGS NOT EXTRACTED!")
        print("   Expected: ['backend-test', 'tags-extraction', 'python-sdk']")
        print(f"   Got: {extracted_tags}")
        success = False
    else:
        print("✅ Tags extracted successfully!")
        print(f"   Expected: {tags}")
        print(f"   Got: {list(extracted_tags)}")

        # Check if all tags are present
        for tag in tags:
            if tag in extracted_tags:
                print(f"   ✅ Tag '{tag}' found")
            else:
                print(f"   ❌ Tag '{tag}' missing")
                success = False

    # Validate other fields
    print()
    if row[1] == "test-tags-extraction":
        print("✅ Trace name correct")
    else:
        print(f"❌ Trace name incorrect: expected 'test-tags-extraction', got '{row[1]}'")
        success = False

    if row[3] == "test-user-tags":
        print("✅ User ID correct")
    else:
        print(f"⚠️  User ID: expected 'test-user-tags', got '{row[3]}'")

    if row[4] == "test-session-tags":
        print("✅ Session ID correct")
    else:
        print(f"⚠️  Session ID: expected 'test-session-tags', got '{row[4]}'")

    if row[5] == "tags-test":
        print("✅ Environment correct")
    else:
        print(f"❌ Environment incorrect: expected 'tags-test', got '{row[5]}'")
        success = False

    # Final result
    print("\n" + "=" * 70)
    if success:
        print("🎉 ALL TESTS PASSED! Tags extraction working correctly!")
    else:
        print("❌ TESTS FAILED! Tags extraction needs investigation.")
    print("=" * 70 + "\n")

    brokle.close()
    return success


if __name__ == "__main__":
    try:
        success = test_tags_extraction()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
