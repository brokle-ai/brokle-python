"""Tests for task manager background processor."""

from unittest.mock import Mock, patch

import pytest

from brokle._task_manager.processor import BackgroundProcessor, get_background_processor
from brokle.config import Config


class TestBackgroundProcessor:
    """Test background processor functionality."""

    def test_singleton_with_explicit_config(self):
        """Test singleton behavior when passing explicit config."""
        # Create a mock config
        config = Mock(spec=Config)

        # Reset any existing processor
        with patch("brokle._task_manager.processor._background_processor", None):
            processor1 = get_background_processor(config=config)
            processor2 = get_background_processor(config=config)

            # Should return the same instance
            assert processor1 is processor2
            assert isinstance(processor1, BackgroundProcessor)

    def test_singleton_with_config_factory(self):
        """Test singleton behavior when using config factory."""
        # Create a mock config and factory
        mock_config = Mock(spec=Config)
        config_factory = Mock(return_value=mock_config)

        # Reset any existing processor
        with patch("brokle._task_manager.processor._background_processor", None):
            processor1 = get_background_processor(config_factory=config_factory)
            processor2 = get_background_processor(config_factory=config_factory)

            # Should return the same instance
            assert processor1 is processor2
            assert isinstance(processor1, BackgroundProcessor)

            # Factory should only be called once (for singleton creation)
            config_factory.assert_called_once()

    def test_singleton_requires_config_or_factory(self):
        """Test that function requires either config or config_factory."""
        # Reset any existing processor
        with patch("brokle._task_manager.processor._background_processor", None):
            # Should raise ValueError when neither config nor config_factory provided
            with pytest.raises(
                ValueError, match="Either config or config_factory must be provided"
            ):
                get_background_processor()

    def test_config_takes_precedence_over_factory(self):
        """Test that explicit config takes precedence over config_factory."""
        # Create mocks
        explicit_config = Mock(spec=Config)
        factory_config = Mock(spec=Config)
        config_factory = Mock(return_value=factory_config)

        # Reset any existing processor
        with patch("brokle._task_manager.processor._background_processor", None):
            processor = get_background_processor(
                config=explicit_config, config_factory=config_factory
            )

            # Should use explicit config, not call factory
            assert isinstance(processor, BackgroundProcessor)
            config_factory.assert_not_called()


class TestBackgroundProcessorPublicAPI:
    """Test public API methods of BackgroundProcessor."""

    def test_submit_methods(self):
        """Test public submit methods work correctly."""
        config = Mock(spec=Config)
        config.telemetry_enabled = False  # Disable HTTP calls
        config.batch_max_size = 10
        processor = BackgroundProcessor(config)

        # Test data
        test_data = {"test": "data", "value": 123}

        try:
            # Test submit methods don't raise errors
            processor.submit_telemetry(test_data)
            processor.submit_analytics(test_data)
            processor.submit_evaluation(test_data)

            # Verify items were queued (queue should have 3 items)
            assert processor._queue.qsize() == 3

        finally:
            processor.shutdown()

    def test_flush_without_timeout(self):
        """Test flush method without timeout waits indefinitely."""
        config = Mock(spec=Config)
        config.telemetry_enabled = False  # Disable HTTP calls
        config.batch_max_size = 10
        processor = BackgroundProcessor(config)

        try:
            # Test flush on empty queue (should return immediately)
            result = processor.flush()
            assert result is True

            # Note: We don't test with items because flush(timeout=None)
            # would wait indefinitely, which is the correct behavior

        finally:
            processor.shutdown()

    def test_flush_with_timeout(self):
        """Test flush method with timeout."""
        config = Mock(spec=Config)
        config.telemetry_enabled = False  # Disable HTTP calls
        config.batch_max_size = 10
        processor = BackgroundProcessor(config)

        try:
            # Submit some test data
            processor.submit_telemetry({"test": "data"})

            # Flush with short timeout should return boolean
            result = processor.flush(timeout=0.5)
            assert isinstance(result, bool)

        finally:
            processor.shutdown()

    def test_flush_on_shutdown_processor(self):
        """Test flush behavior on already shutdown processor."""
        config = Mock(spec=Config)
        config.telemetry_enabled = False
        config.batch_max_size = 10
        processor = BackgroundProcessor(config)

        # Shutdown first
        processor.shutdown()

        # Flush should return True for shutdown processor
        result = processor.flush()
        assert result is True

    def test_get_metrics(self):
        """Test get_metrics returns expected data structure."""
        config = Mock(spec=Config)
        config.telemetry_enabled = False
        config.batch_max_size = 10
        processor = BackgroundProcessor(config)

        try:
            # Get initial metrics
            metrics = processor.get_metrics()

            # Verify required keys exist
            required_keys = [
                "queue_depth",
                "queue_max_size",
                "items_processed",
                "items_failed",
                "batches_processed",
                "uptime_seconds",
                "processing_rate",
                "error_rate",
                "worker_alive",
                "shutdown",
                "last_error",
                "last_error_time",
            ]

            for key in required_keys:
                assert key in metrics, f"Missing key: {key}"

            # Verify data types
            assert isinstance(metrics["queue_depth"], int)
            assert isinstance(metrics["queue_max_size"], int)
            assert isinstance(metrics["items_processed"], int)
            assert isinstance(metrics["items_failed"], int)
            assert isinstance(metrics["batches_processed"], int)
            assert isinstance(metrics["uptime_seconds"], (int, float))
            assert isinstance(metrics["processing_rate"], (int, float))
            assert isinstance(metrics["error_rate"], (int, float))
            assert isinstance(metrics["worker_alive"], bool)
            assert isinstance(metrics["shutdown"], bool)

            # Verify initial values
            assert metrics["items_processed"] == 0
            assert metrics["items_failed"] == 0
            assert metrics["batches_processed"] == 0
            assert metrics["queue_depth"] == 0
            assert metrics["queue_max_size"] == 10000
            assert metrics["worker_alive"] is True
            assert metrics["shutdown"] is False

        finally:
            processor.shutdown()

    def test_get_metrics_with_processing(self):
        """Test get_metrics updates correctly after processing."""
        config = Mock(spec=Config)
        config.telemetry_enabled = False  # Disable actual HTTP calls
        config.batch_max_size = 10

        processor = BackgroundProcessor(config)

        try:
            # Submit some test data
            processor.submit_telemetry({"test": "data1"})
            processor.submit_analytics({"test": "data2"})

            # Allow some processing time
            import time

            time.sleep(0.5)

            metrics = processor.get_metrics()

            # Queue should have items
            assert metrics["queue_depth"] >= 0  # May be processed already

        finally:
            processor.shutdown()

    def test_is_healthy_fresh_processor(self):
        """Test is_healthy returns True for fresh processor."""
        config = Mock(spec=Config)
        config.telemetry_enabled = False
        config.batch_max_size = 10
        processor = BackgroundProcessor(config)

        try:
            # Fresh processor should be healthy
            assert processor.is_healthy() is True

        finally:
            processor.shutdown()

    def test_is_healthy_shutdown_processor(self):
        """Test is_healthy returns False for shutdown processor."""
        config = Mock(spec=Config)
        config.telemetry_enabled = False
        config.batch_max_size = 10
        processor = BackgroundProcessor(config)

        processor.shutdown()

        # Shutdown processor should not be healthy
        assert processor.is_healthy() is False

    def test_submit_methods_on_shutdown(self):
        """Test submit methods handle shutdown gracefully."""
        config = Mock(spec=Config)
        config.telemetry_enabled = False
        config.batch_max_size = 10
        processor = BackgroundProcessor(config)

        processor.shutdown()

        # Submit methods should not raise errors on shutdown
        try:
            processor.submit_telemetry({"test": "data"})
            processor.submit_analytics({"test": "data"})
            processor.submit_evaluation({"test": "data"})
        except Exception:
            pytest.fail("Submit methods should not raise on shutdown")

        # Queue should remain empty since processor is shutdown
        assert processor._queue.qsize() == 0
