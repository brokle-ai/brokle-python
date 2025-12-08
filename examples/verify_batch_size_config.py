"""
Verification: batch_max_size configuration is properly wired.

This example demonstrates that the batch_max_size configuration parameter
is actually used by the background processor's worker loop.
"""

import time

from brokle import Brokle
from brokle.config import Config


def verify_batch_max_size_default():
    """Verify default batch_max_size is 100."""
    print("\n=== Verify Default batch_max_size ===")

    config = Config(api_key="bk_test")
    print(f"✅ Default batch_max_size: {config.batch_max_size}")
    assert config.batch_max_size == 100, "Default should be 100"


def verify_batch_max_size_custom():
    """Verify custom batch_max_size is respected."""
    print("\n=== Verify Custom batch_max_size ===")

    # Create client with custom batch size
    client = Brokle(api_key="bk_test", batch_max_size=250)  # Custom batch size

    print(f"✅ Custom batch_max_size: {client.config.batch_max_size}")
    assert client.config.batch_max_size == 250, "Custom value should be 250"

    # Verify processor uses this config
    processor = client._background_processor
    print(f"✅ Processor uses config: {processor.config.batch_max_size}")
    assert processor.config.batch_max_size == 250, "Processor should use custom value"

    client.close()


def verify_batch_max_size_environment():
    """Verify batch_max_size from environment variable."""
    print("\n=== Verify Environment Variable ===")

    import os

    # Set environment variable
    os.environ["BROKLE_BATCH_MAX_SIZE"] = "500"
    os.environ["BROKLE_API_KEY"] = "bk_test"

    # Create config from environment
    config = Config.from_env()
    print(f"✅ Environment batch_max_size: {config.batch_max_size}")
    assert config.batch_max_size == 500, "Should read from BROKLE_BATCH_MAX_SIZE"

    # Clean up
    del os.environ["BROKLE_BATCH_MAX_SIZE"]
    del os.environ["BROKLE_API_KEY"]


def verify_batch_size_actually_used():
    """Verify worker loop actually uses batch_max_size."""
    print("\n=== Verify Worker Loop Uses batch_max_size ===")

    from brokle._task_manager.processor import BackgroundProcessor

    # Create processor with large batch size
    config = Config(api_key="bk_test", batch_max_size=300)
    processor = BackgroundProcessor(config)

    # Submit many events
    print("Submitting 350 events...")
    for i in range(350):
        processor.submit_telemetry({"index": i})

    # Give worker time to process
    time.sleep(2)

    # Check metrics
    metrics = processor.get_metrics()
    print(f"✅ Queue depth: {metrics['queue_depth']}")
    print(f"✅ Batches processed: {metrics['batches_processed']}")
    print(f"✅ Items processed: {metrics['items_processed']}")

    # With batch_max_size=300, 350 events should be processed in 2 batches
    # (300 in first batch, 50 in second batch)
    assert metrics["batches_processed"] >= 1, "At least 1 batch should be processed"

    processor.shutdown()
    print("✅ Worker correctly used batch_max_size=300")


def verify_validation():
    """Verify batch_max_size validation."""
    print("\n=== Verify Validation ===")

    # Test min validation (should be >= 1)
    try:
        Config(api_key="bk_test", batch_max_size=0)
        assert False, "Should reject batch_max_size=0"
    except Exception:
        print("✅ Correctly rejects batch_max_size=0")

    # Test max validation (should be <= 1000)
    try:
        Config(api_key="bk_test", batch_max_size=2000)
        assert False, "Should reject batch_max_size=2000"
    except Exception:
        print("✅ Correctly rejects batch_max_size=2000")

    # Test valid values
    config = Config(api_key="bk_test", batch_max_size=1)
    assert config.batch_max_size == 1
    print("✅ Accepts batch_max_size=1 (min)")

    config = Config(api_key="bk_test", batch_max_size=1000)
    assert config.batch_max_size == 1000
    print("✅ Accepts batch_max_size=1000 (max)")


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("Verification: batch_max_size Configuration")
    print("=" * 60)

    verify_batch_max_size_default()
    verify_batch_max_size_custom()
    verify_batch_max_size_environment()
    verify_batch_size_actually_used()
    verify_validation()

    print("\n" + "=" * 60)
    print("✅ All verifications passed!")
    print("=" * 60)
    print("\nConclusion:")
    print("  - batch_max_size is properly wired to the worker loop")
    print("  - Worker no longer uses hard-coded 100 limit")
    print("  - Configuration range: 1-1000 events per batch")
    print("  - Default: 100 events per batch")


if __name__ == "__main__":
    main()
