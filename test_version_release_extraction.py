#!/usr/bin/env python3
"""
Test: Version & Release Extraction Verification
================================================

This test validates:
1. Release (automatic) - set via config.release or BROKLE_RELEASE env var
2. Version (manual) - set via decorator/span parameter for A/B testing
3. Backend extracts both from brokle.version and brokle.release attributes
4. Both stored correctly in ClickHouse traces table

Requirements:
- Backend running: make dev-server
- ClickHouse running: docker-compose up clickhouse
- API key: bk_SZJvBQDr9brY80Ln1ceNtGZMoSNc175rs3gXbnLK
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


def test_release_automatic():
    """Test 1: Release is automatically set from config."""
    print("\n" + "=" * 70)
    print("🧪 Test 1: Release (Automatic via Config)")
    print("=" * 70 + "\n")

    release_value = "v2.5.0-test"

    # Initialize client with release
    print(f"📡 Initializing Brokle client with release='{release_value}'...")
    brokle = Brokle(
        api_key=API_KEY,
        base_url=BASE_URL,
        environment="test-release",
        release=release_value,  # Set release in config
        flush_at=1,
        flush_interval=0.5,
    )
    print(f"✅ Client initialized with release={release_value}\n")

    # Create span (release should be added automatically)
    print("📝 Creating span (release added automatically by processor)...")

    @observe(name="test-release-automatic")
    def test_operation():
        return "Release test"

    result = test_operation()
    print("✅ Span created\n")

    # Flush and wait
    print("⏳ Flushing telemetry...")
    brokle.flush()
    time.sleep(3)

    # Query ClickHouse
    print("🔍 Querying ClickHouse for release...")
    ch_client = get_clickhouse_client()

    query = """
        SELECT trace_id, name, release, version, environment
        FROM traces
        WHERE name = 'test-release-automatic'
        AND environment = 'test-release'
        ORDER BY created_at DESC
        LIMIT 1
    """

    result = ch_client.query(query)

    if result.row_count == 0:
        print("❌ Trace not found")
        brokle.close()
        return False

    row = result.first_row
    print("✅ Trace found\n")

    print("📊 Trace Data:")
    print(f"   trace_id: {row[0]}")
    print(f"   name: {row[1]}")
    print(f"   release: {row[2]}")
    print(f"   version: {row[3]}")
    print(f"   environment: {row[4]}")
    print()

    # Validate release
    success = True
    if row[2] == release_value:
        print(f"✅ Release extracted correctly: '{row[2]}'")
    else:
        print(f"❌ Release incorrect: expected '{release_value}', got '{row[2]}'")
        success = False

    if row[3] is None:
        print("✅ Version is NULL (not set, as expected)")
    else:
        print(f"⚠️  Version unexpectedly set: '{row[3]}'")

    brokle.close()
    return success


def test_version_manual():
    """Test 2: Version must be explicitly passed to decorator."""
    print("\n" + "=" * 70)
    print("🧪 Test 2: Version (Manual via Decorator Parameter)")
    print("=" * 70 + "\n")

    version_value = "experiment-A-v1"

    # Initialize client (no version in config)
    print("📡 Initializing Brokle client (no version in config)...")
    brokle = Brokle(
        api_key=API_KEY,
        base_url=BASE_URL,
        environment="test-version",
        flush_at=1,
        flush_interval=0.5,
    )
    print("✅ Client initialized\n")

    # Create span with version parameter
    print(f"📝 Creating span with version='{version_value}' parameter...")

    @observe(
        name="test-version-manual",
        version=version_value,  # Explicitly pass version for A/B testing
    )
    def test_operation():
        return "Version test"

    result = test_operation()
    print("✅ Span created with version parameter\n")

    # Flush and wait
    print("⏳ Flushing telemetry...")
    brokle.flush()
    time.sleep(3)

    # Query ClickHouse
    print("🔍 Querying ClickHouse for version...")
    ch_client = get_clickhouse_client()

    query = """
        SELECT trace_id, name, release, version, environment
        FROM traces
        WHERE name = 'test-version-manual'
        AND environment = 'test-version'
        ORDER BY created_at DESC
        LIMIT 1
    """

    result = ch_client.query(query)

    if result.row_count == 0:
        print("❌ Trace not found")
        brokle.close()
        return False

    row = result.first_row
    print("✅ Trace found\n")

    print("📊 Trace Data:")
    print(f"   trace_id: {row[0]}")
    print(f"   name: {row[1]}")
    print(f"   release: {row[2]}")
    print(f"   version: {row[3]}")
    print(f"   environment: {row[4]}")
    print()

    # Validate version
    success = True
    if row[3] == version_value:
        print(f"✅ Version extracted correctly: '{row[3]}'")
    else:
        print(f"❌ Version incorrect: expected '{version_value}', got '{row[3]}'")
        success = False

    if row[2] is None:
        print("✅ Release is NULL (not set, as expected)")
    else:
        print(f"⚠️  Release unexpectedly set: '{row[2]}'")

    brokle.close()
    return success


def test_version_and_release_combined():
    """Test 3: Both version and release can be set together."""
    print("\n" + "=" * 70)
    print("🧪 Test 3: Version + Release Combined")
    print("=" * 70 + "\n")

    version_value = "experiment-B-v2"
    release_value = "v3.0.0-beta"

    # Initialize client with release
    print(f"📡 Initializing with release='{release_value}'...")
    brokle = Brokle(
        api_key=API_KEY,
        base_url=BASE_URL,
        environment="test-combined",
        release=release_value,
        flush_at=1,
        flush_interval=0.5,
    )
    print(f"✅ Client initialized\n")

    # Create span with version parameter
    print(f"📝 Creating span with version='{version_value}'...")

    @observe(
        name="test-combined",
        version=version_value,
    )
    def test_operation():
        return "Combined test"

    result = test_operation()
    print("✅ Span created\n")

    # Flush and wait
    print("⏳ Flushing telemetry...")
    brokle.flush()
    time.sleep(3)

    # Query ClickHouse
    print("🔍 Querying ClickHouse...")
    ch_client = get_clickhouse_client()

    query = """
        SELECT trace_id, name, release, version, environment
        FROM traces
        WHERE name = 'test-combined'
        AND environment = 'test-combined'
        ORDER BY created_at DESC
        LIMIT 1
    """

    result = ch_client.query(query)

    if result.row_count == 0:
        print("❌ Trace not found")
        brokle.close()
        return False

    row = result.first_row
    print("✅ Trace found\n")

    print("📊 Trace Data:")
    print(f"   trace_id: {row[0]}")
    print(f"   name: {row[1]}")
    print(f"   release: {row[2]}")
    print(f"   version: {row[3]}")
    print(f"   environment: {row[4]}")
    print()

    # Validate both
    success = True
    if row[2] == release_value:
        print(f"✅ Release extracted correctly: '{row[2]}'")
    else:
        print(f"❌ Release incorrect: expected '{release_value}', got '{row[2]}'")
        success = False

    if row[3] == version_value:
        print(f"✅ Version extracted correctly: '{row[3]}'")
    else:
        print(f"❌ Version incorrect: expected '{version_value}', got '{row[3]}'")
        success = False

    brokle.close()
    return success


def main():
    """Run all version/release extraction tests."""
    print("\n" + "🚀" * 35)
    print("🧪 Version & Release Extraction Test Suite")
    print("🚀" * 35)

    results = []

    # Test 1: Release (automatic)
    try:
        results.append(("Release (Automatic)", test_release_automatic()))
    except Exception as e:
        print(f"\n❌ Test 1 failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Release (Automatic)", False))

    # Test 2: Version (manual)
    try:
        results.append(("Version (Manual)", test_version_manual()))
    except Exception as e:
        print(f"\n❌ Test 2 failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Version (Manual)", False))

    # Test 3: Combined
    try:
        results.append(("Version + Release Combined", test_version_and_release_combined()))
    except Exception as e:
        print(f"\n❌ Test 3 failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Version + Release Combined", False))

    # Print summary
    print("\n" + "=" * 70)
    print("📊 Test Results Summary")
    print("=" * 70 + "\n")

    passed = 0
    failed = 0

    for test_name, result in results:
        if result:
            print(f"✅ {test_name}")
            passed += 1
        else:
            print(f"❌ {test_name}")
            failed += 1

    print(f"\n📊 Total: {passed} passed, {failed} failed\n")

    if failed == 0:
        print("=" * 70)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 70)
        print("\n✅ Version & Release extraction working correctly:")
        print("   • Release: Automatic (via config.release or BROKLE_RELEASE)")
        print("   • Version: Manual (via @observe(version=...) parameter)")
        print("   • Backend: Extracts both from brokle.version and brokle.release")
        print("   • ClickHouse: Both stored in traces.version and traces.release\n")
        return 0
    else:
        print("=" * 70)
        print("❌ SOME TESTS FAILED")
        print("=" * 70)
        print("\n💡 Possible issues:")
        print("   1. Backend not running (make dev-server)")
        print("   2. ClickHouse migrations not up to date (make migrate-up)")
        print("   3. Backend not processing telemetry correctly\n")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
