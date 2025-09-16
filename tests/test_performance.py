"""
Performance tests for auto-instrumentation system.

This module tests the performance overhead of instrumentation to ensure
it doesn't significantly impact user applications.
"""

import asyncio
import time
import pytest
import statistics
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Callable

from brokle.auto_instrumentation import (
    auto_instrument,
    get_registry,
    reset_all_errors
)

from brokle.auto_instrumentation.openai_instrumentation import OpenAIInstrumentation
from brokle.auto_instrumentation.error_handlers import (
    safe_operation,
    safe_async_operation,
    instrumentation_context,
    CircuitBreaker,
    RetryPolicy,
    get_error_handler
)


class PerformanceBenchmark:
    """Helper class for performance benchmarking."""

    @staticmethod
    def measure_execution_time(func: Callable, iterations: int = 1000) -> Dict[str, float]:
        """Measure function execution time statistics."""
        times = []

        for _ in range(iterations):
            start_time = time.perf_counter()
            func()
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds

        return {
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'min': min(times),
            'max': max(times),
            'std': statistics.stdev(times) if len(times) > 1 else 0.0,
            'iterations': iterations
        }

    @staticmethod
    async def measure_async_execution_time(func: Callable, iterations: int = 1000) -> Dict[str, float]:
        """Measure async function execution time statistics."""
        times = []

        for _ in range(iterations):
            start_time = time.perf_counter()
            await func()
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds

        return {
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'min': min(times),
            'max': max(times),
            'std': statistics.stdev(times) if len(times) > 1 else 0.0,
            'iterations': iterations
        }


class TestInstrumentationOverhead:
    """Test instrumentation overhead and performance impact."""

    def setup_method(self):
        """Setup for each test method."""
        reset_all_errors()

    def test_registry_operations_performance(self):
        """Test performance of registry operations."""
        registry = get_registry()

        # Test get_status performance
        def get_status_operation():
            return registry.get_status()

        stats = PerformanceBenchmark.measure_execution_time(get_status_operation, 1000)

        # Should complete within reasonable time (< 1ms average)
        assert stats['mean'] < 1.0, f"get_status too slow: {stats['mean']:.2f}ms average"
        print(f"get_status performance: {stats['mean']:.2f}ms mean, {stats['median']:.2f}ms median")

        # Test get_health_report performance
        def get_health_operation():
            return registry.get_health_report()

        health_stats = PerformanceBenchmark.measure_execution_time(get_health_operation, 500)

        # Health report can be slightly more expensive but should still be fast (< 5ms average)
        assert health_stats['mean'] < 5.0, f"get_health_report too slow: {health_stats['mean']:.2f}ms average"
        print(f"get_health_report performance: {health_stats['mean']:.2f}ms mean, {health_stats['median']:.2f}ms median")

    def test_error_handler_overhead(self):
        """Test error handler operations overhead."""
        error_handler = get_error_handler()

        # Test is_operation_healthy performance
        def health_check_operation():
            return error_handler.is_operation_healthy("test_lib", "test_op")

        stats = PerformanceBenchmark.measure_execution_time(health_check_operation, 5000)

        # Health checks should be very fast (< 0.1ms average)
        assert stats['mean'] < 0.1, f"Health check too slow: {stats['mean']:.2f}ms average"
        print(f"Health check performance: {stats['mean']:.2f}ms mean, {stats['median']:.2f}ms median")

        # Test error handling performance
        test_error = Exception("Performance test error")

        def error_handling_operation():
            error_handler.handle_error(test_error, "perf_lib", "perf_op", error_handler.ErrorSeverity.LOW)

        error_stats = PerformanceBenchmark.measure_execution_time(error_handling_operation, 1000)

        # Error handling should be reasonably fast (< 1ms average)
        assert error_stats['mean'] < 1.0, f"Error handling too slow: {error_stats['mean']:.2f}ms average"
        print(f"Error handling performance: {error_stats['mean']:.2f}ms mean, {error_stats['median']:.2f}ms median")

    def test_circuit_breaker_performance(self):
        """Test circuit breaker performance impact."""
        circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)

        def successful_operation():
            return "success"

        # Test circuit breaker call overhead
        def circuit_breaker_operation():
            return circuit_breaker.call(successful_operation)

        stats = PerformanceBenchmark.measure_execution_time(circuit_breaker_operation, 2000)

        # Circuit breaker should add minimal overhead (< 0.5ms average)
        assert stats['mean'] < 0.5, f"Circuit breaker overhead too high: {stats['mean']:.2f}ms average"
        print(f"Circuit breaker performance: {stats['mean']:.2f}ms mean, {stats['median']:.2f}ms median")

    def test_safe_operation_decorator_overhead(self):
        """Test safe operation decorator overhead."""

        def baseline_function():
            return sum(range(100))  # Some work to measure

        @safe_operation("test_lib", "decorated_op", error_handler.ErrorSeverity.LOW)
        def decorated_function():
            return sum(range(100))  # Same work

        # Measure baseline performance
        baseline_stats = PerformanceBenchmark.measure_execution_time(baseline_function, 2000)

        # Measure decorated performance
        decorated_stats = PerformanceBenchmark.measure_execution_time(decorated_function, 2000)

        # Calculate overhead
        overhead_ms = decorated_stats['mean'] - baseline_stats['mean']
        overhead_percentage = (overhead_ms / baseline_stats['mean']) * 100

        print(f"Baseline: {baseline_stats['mean']:.4f}ms, Decorated: {decorated_stats['mean']:.4f}ms")
        print(f"Overhead: {overhead_ms:.4f}ms ({overhead_percentage:.1f}%)")

        # Overhead should be minimal (< 100% of baseline)
        assert overhead_percentage < 100.0, f"Decorator overhead too high: {overhead_percentage:.1f}%"

    @pytest.mark.asyncio
    async def test_async_safe_operation_overhead(self):
        """Test async safe operation decorator overhead."""

        async def baseline_async_function():
            await asyncio.sleep(0.001)  # Simulate async work
            return sum(range(100))

        @safe_async_operation("test_lib", "async_decorated_op", error_handler.ErrorSeverity.LOW)
        async def decorated_async_function():
            await asyncio.sleep(0.001)  # Same async work
            return sum(range(100))

        # Measure baseline performance
        baseline_stats = await PerformanceBenchmark.measure_async_execution_time(baseline_async_function, 500)

        # Measure decorated performance
        decorated_stats = await PerformanceBenchmark.measure_async_execution_time(decorated_async_function, 500)

        # Calculate overhead
        overhead_ms = decorated_stats['mean'] - baseline_stats['mean']
        overhead_percentage = (overhead_ms / baseline_stats['mean']) * 100

        print(f"Async Baseline: {baseline_stats['mean']:.4f}ms, Decorated: {decorated_stats['mean']:.4f}ms")
        print(f"Async Overhead: {overhead_ms:.4f}ms ({overhead_percentage:.1f}%)")

        # Overhead should be reasonable (< 50% of baseline for async operations)
        assert overhead_percentage < 50.0, f"Async decorator overhead too high: {overhead_percentage:.1f}%"

    @patch('brokle.auto_instrumentation.openai_instrumentation.OPENAI_AVAILABLE', True)
    @patch('brokle.auto_instrumentation.openai_instrumentation.openai')
    def test_openai_instrumentation_overhead(self, mock_openai):
        """Test OpenAI instrumentation wrapper overhead."""
        # Setup mock OpenAI
        mock_create = Mock(return_value=Mock(id="test", model="gpt-3.5-turbo"))
        mock_completions = Mock()
        mock_completions.create = mock_create

        mock_chat = Mock()
        mock_chat.completions = Mock()
        mock_chat.completions.Completions = mock_completions

        mock_resources = Mock()
        mock_resources.chat = mock_chat

        mock_openai.resources = mock_resources

        # Create instrumentation instance
        instrumentation = OpenAIInstrumentation()

        # Mock the client and config
        mock_client = Mock()
        mock_client.observability = Mock()
        mock_client.observability.create_observation_sync = Mock(return_value=Mock(id="obs_123"))
        mock_client.observability.complete_observation_sync = Mock()
        mock_client.observability.create_trace_sync = Mock(return_value=Mock(id="trace_123"))

        with patch.object(instrumentation, 'client', mock_client):
            with patch.object(instrumentation, 'config', Mock()):
                # Instrument the method
                instrumentation.instrument()

                # Test the wrapped method performance
                def call_openai():
                    return mock_completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "test"}])

                stats = PerformanceBenchmark.measure_execution_time(call_openai, 100)

                # Instrumentation overhead should be reasonable (< 5ms average)
                assert stats['mean'] < 5.0, f"OpenAI instrumentation overhead too high: {stats['mean']:.2f}ms"
                print(f"OpenAI instrumentation overhead: {stats['mean']:.2f}ms mean")

                # Clean up
                instrumentation.uninstrument()

    def test_retry_policy_performance(self):
        """Test retry policy performance for successful operations."""
        retry_policy = RetryPolicy(max_retries=3, base_delay=0.001)

        def successful_operation():
            return "success"

        def retry_operation():
            return retry_policy.execute(successful_operation)

        stats = PerformanceBenchmark.measure_execution_time(retry_operation, 1000)

        # Successful operations should have minimal retry overhead (< 1ms average)
        assert stats['mean'] < 1.0, f"Retry policy overhead too high: {stats['mean']:.2f}ms average"
        print(f"Retry policy (success) performance: {stats['mean']:.2f}ms mean")

    def test_context_manager_overhead(self):
        """Test instrumentation context manager overhead."""

        def baseline_operation():
            return sum(range(50))

        def context_manager_operation():
            with instrumentation_context("test_lib", "context_op", error_handler.ErrorSeverity.LOW):
                return sum(range(50))

        baseline_stats = PerformanceBenchmark.measure_execution_time(baseline_operation, 2000)
        context_stats = PerformanceBenchmark.measure_execution_time(context_manager_operation, 2000)

        overhead_ms = context_stats['mean'] - baseline_stats['mean']
        overhead_percentage = (overhead_ms / baseline_stats['mean']) * 100

        print(f"Context manager overhead: {overhead_ms:.4f}ms ({overhead_percentage:.1f}%)")

        # Context manager should add minimal overhead
        assert overhead_percentage < 200.0, f"Context manager overhead too high: {overhead_percentage:.1f}%"


class TestMemoryUsage:
    """Test memory usage patterns of instrumentation."""

    def setup_method(self):
        """Setup for each test method."""
        reset_all_errors()

    def test_error_tracking_memory_growth(self):
        """Test that error tracking doesn't cause memory leaks."""
        import gc

        error_handler = get_error_handler()

        # Force garbage collection and get initial object count
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Generate many errors
        test_error = Exception("Memory test error")
        for i in range(1000):
            error_handler.handle_error(
                test_error,
                f"lib_{i % 10}",
                f"op_{i % 5}",
                error_handler.ErrorSeverity.LOW
            )

        # Force garbage collection
        gc.collect()
        after_errors_objects = len(gc.get_objects())

        # Reset errors
        for lib_id in range(10):
            for op_id in range(5):
                error_handler.reset_errors(f"lib_{lib_id}", f"op_{op_id}")

        # Force garbage collection again
        gc.collect()
        final_objects = len(gc.get_objects())

        print(f"Objects: Initial={initial_objects}, After errors={after_errors_objects}, Final={final_objects}")

        # Memory should be released after reset (within reasonable bounds)
        memory_growth = final_objects - initial_objects
        assert memory_growth < 500, f"Potential memory leak: {memory_growth} objects not released"

    def test_instrumentation_state_memory(self):
        """Test instrumentation state doesn't accumulate memory."""
        import gc

        registry = get_registry()

        gc.collect()
        initial_objects = len(gc.get_objects())

        # Perform many registry operations
        for i in range(500):
            status = registry.get_status()
            health = registry.get_health_report()
            summary = registry.get_instrumentation_summary()

            # Clear references
            del status, health, summary

        gc.collect()
        final_objects = len(gc.get_objects())

        memory_growth = final_objects - initial_objects
        print(f"Registry operations memory growth: {memory_growth} objects")

        # Should not accumulate significant memory
        assert memory_growth < 300, f"Registry operations memory leak: {memory_growth} objects"


class TestConcurrencyPerformance:
    """Test performance under concurrent access."""

    def setup_method(self):
        """Setup for each test method."""
        reset_all_errors()

    def test_concurrent_error_handling(self):
        """Test error handling performance under concurrent access."""
        import threading
        import queue

        error_handler = get_error_handler()
        results_queue = queue.Queue()

        def worker(worker_id: int):
            # Each worker performs error handling operations
            start_time = time.perf_counter()

            for i in range(100):
                test_error = Exception(f"Worker {worker_id} error {i}")
                error_handler.handle_error(
                    test_error,
                    f"worker_{worker_id}",
                    f"operation_{i % 10}",
                    error_handler.ErrorSeverity.LOW
                )

                # Check health occasionally
                if i % 20 == 0:
                    error_handler.is_operation_healthy(f"worker_{worker_id}", f"operation_{i % 10}")

            end_time = time.perf_counter()
            worker_time = (end_time - start_time) * 1000  # Convert to ms
            results_queue.put(worker_time)

        # Run multiple workers concurrently
        num_workers = 10
        threads = []

        start_time = time.perf_counter()

        for worker_id in range(num_workers):
            thread = threading.Thread(target=worker, args=(worker_id,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        end_time = time.perf_counter()
        total_time = (end_time - start_time) * 1000

        # Collect worker times
        worker_times = []
        while not results_queue.empty():
            worker_times.append(results_queue.get())

        avg_worker_time = statistics.mean(worker_times)
        max_worker_time = max(worker_times)

        print(f"Concurrent error handling: Total={total_time:.2f}ms, Avg worker={avg_worker_time:.2f}ms, Max worker={max_worker_time:.2f}ms")

        # Performance should be reasonable even under concurrency
        assert avg_worker_time < 100.0, f"Concurrent error handling too slow: {avg_worker_time:.2f}ms average per worker"
        assert max_worker_time < 200.0, f"Worst-case concurrent performance too slow: {max_worker_time:.2f}ms"

    def test_concurrent_registry_access(self):
        """Test registry performance under concurrent access."""
        import threading
        import queue

        registry = get_registry()
        results_queue = queue.Queue()

        def registry_worker():
            start_time = time.perf_counter()

            for i in range(50):
                # Mix of different registry operations
                if i % 3 == 0:
                    registry.get_status()
                elif i % 3 == 1:
                    registry.get_health_report()
                else:
                    registry.get_instrumentation_summary()

            end_time = time.perf_counter()
            worker_time = (end_time - start_time) * 1000
            results_queue.put(worker_time)

        # Run concurrent registry access
        num_workers = 8
        threads = []

        for _ in range(num_workers):
            thread = threading.Thread(target=registry_worker)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Collect results
        worker_times = []
        while not results_queue.empty():
            worker_times.append(results_queue.get())

        avg_time = statistics.mean(worker_times)
        max_time = max(worker_times)

        print(f"Concurrent registry access: Avg={avg_time:.2f}ms, Max={max_time:.2f}ms per worker")

        # Should handle concurrent access efficiently
        assert avg_time < 50.0, f"Concurrent registry access too slow: {avg_time:.2f}ms average"


# Performance test markers
pytestmark = [
    pytest.mark.performance,
    pytest.mark.slow  # Mark as slow since these are performance tests
]

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])  # -s to see print statements