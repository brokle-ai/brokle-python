#!/usr/bin/env python3
"""
Performance Benchmarking Suite for Auto-Instrumentation.

This script provides comprehensive performance benchmarks for the Brokle
auto-instrumentation system to ensure minimal overhead and optimal performance.
"""

import asyncio
import gc
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable, Any
from unittest.mock import Mock, patch

# Add the SDK to the path for benchmarking
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import brokle.auto_instrumentation as brokle_ai
    from brokle.auto_instrumentation.error_handlers import (
        safe_operation, safe_async_operation, instrumentation_context,
        CircuitBreaker, RetryPolicy, get_error_handler
    )
    from brokle.auto_instrumentation.openai_instrumentation import OpenAIInstrumentation
    BROKLE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Brokle SDK not available: {e}")
    BROKLE_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    operation_name: str
    iterations: int
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    std_dev_ms: float
    p95_ms: float
    p99_ms: float
    ops_per_second: float
    total_time_ms: float
    overhead_percentage: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite."""

    def __init__(self, warmup_iterations: int = 100, test_iterations: int = 1000):
        self.warmup_iterations = warmup_iterations
        self.test_iterations = test_iterations
        self.results: List[BenchmarkResult] = []

    def benchmark_function(self, func: Callable, name: str, iterations: Optional[int] = None,
                         warmup: bool = True) -> BenchmarkResult:
        """Benchmark a function's performance."""
        iterations = iterations or self.test_iterations

        # Warmup
        if warmup:
            for _ in range(self.warmup_iterations):
                try:
                    func()
                except Exception:
                    pass  # Ignore warmup errors

        # Force garbage collection before measurement
        gc.collect()

        # Measure performance
        times = []
        start_time = time.perf_counter()

        for _ in range(iterations):
            iteration_start = time.perf_counter()
            try:
                func()
            except Exception:
                pass  # Continue benchmarking even if function fails
            iteration_end = time.perf_counter()
            times.append((iteration_end - iteration_start) * 1000)  # Convert to ms

        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000

        # Calculate statistics
        mean_ms = statistics.mean(times)
        median_ms = statistics.median(times)
        min_ms = min(times)
        max_ms = max(times)
        std_dev_ms = statistics.stdev(times) if len(times) > 1 else 0.0
        p95_ms = self._percentile(times, 0.95)
        p99_ms = self._percentile(times, 0.99)
        ops_per_second = 1000.0 / mean_ms if mean_ms > 0 else 0

        result = BenchmarkResult(
            operation_name=name,
            iterations=iterations,
            mean_ms=mean_ms,
            median_ms=median_ms,
            min_ms=min_ms,
            max_ms=max_ms,
            std_dev_ms=std_dev_ms,
            p95_ms=p95_ms,
            p99_ms=p99_ms,
            ops_per_second=ops_per_second,
            total_time_ms=total_time_ms
        )

        self.results.append(result)
        return result

    async def benchmark_async_function(self, func: Callable, name: str,
                                     iterations: Optional[int] = None) -> BenchmarkResult:
        """Benchmark an async function's performance."""
        iterations = iterations or self.test_iterations

        # Warmup
        for _ in range(min(self.warmup_iterations, 50)):  # Reduce async warmup
            try:
                await func()
            except Exception:
                pass

        # Force garbage collection before measurement
        gc.collect()

        # Measure performance
        times = []
        start_time = time.perf_counter()

        for _ in range(iterations):
            iteration_start = time.perf_counter()
            try:
                await func()
            except Exception:
                pass
            iteration_end = time.perf_counter()
            times.append((iteration_end - iteration_start) * 1000)

        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000

        # Calculate statistics (same as sync version)
        mean_ms = statistics.mean(times)
        median_ms = statistics.median(times)
        min_ms = min(times)
        max_ms = max(times)
        std_dev_ms = statistics.stdev(times) if len(times) > 1 else 0.0
        p95_ms = self._percentile(times, 0.95)
        p99_ms = self._percentile(times, 0.99)
        ops_per_second = 1000.0 / mean_ms if mean_ms > 0 else 0

        result = BenchmarkResult(
            operation_name=name,
            iterations=iterations,
            mean_ms=mean_ms,
            median_ms=median_ms,
            min_ms=min_ms,
            max_ms=max_ms,
            std_dev_ms=std_dev_ms,
            p95_ms=p95_ms,
            p99_ms=p99_ms,
            ops_per_second=ops_per_second,
            total_time_ms=total_time_ms
        )

        self.results.append(result)
        return result

    def benchmark_overhead(self, baseline_func: Callable, instrumented_func: Callable,
                          name: str, iterations: Optional[int] = None) -> BenchmarkResult:
        """Benchmark overhead by comparing baseline vs instrumented function."""
        iterations = iterations or self.test_iterations

        # Benchmark baseline
        baseline_result = self.benchmark_function(baseline_func, f"{name}_baseline", iterations, warmup=True)

        # Benchmark instrumented version
        instrumented_result = self.benchmark_function(instrumented_func, f"{name}_instrumented", iterations, warmup=True)

        # Calculate overhead
        overhead_ms = instrumented_result.mean_ms - baseline_result.mean_ms
        overhead_percentage = (overhead_ms / baseline_result.mean_ms) * 100 if baseline_result.mean_ms > 0 else 0

        # Create overhead result
        overhead_result = BenchmarkResult(
            operation_name=f"{name}_overhead",
            iterations=iterations,
            mean_ms=overhead_ms,
            median_ms=instrumented_result.median_ms - baseline_result.median_ms,
            min_ms=instrumented_result.min_ms - baseline_result.min_ms,
            max_ms=instrumented_result.max_ms - baseline_result.max_ms,
            std_dev_ms=instrumented_result.std_dev_ms,
            p95_ms=instrumented_result.p95_ms - baseline_result.p95_ms,
            p99_ms=instrumented_result.p99_ms - baseline_result.p99_ms,
            ops_per_second=instrumented_result.ops_per_second,
            total_time_ms=instrumented_result.total_time_ms,
            overhead_percentage=overhead_percentage
        )

        self.results.append(overhead_result)
        return overhead_result

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(percentile * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]

    def print_results(self):
        """Print benchmark results in a formatted table."""
        if not self.results:
            print("No benchmark results to display")
            return

        print("\n🚀 Auto-Instrumentation Performance Benchmark Results")
        print("=" * 80)

        # Print header
        print(f"{'Operation':<25} {'Mean (ms)':<10} {'P95 (ms)':<9} {'Ops/sec':<10} {'Overhead':<10}")
        print("-" * 80)

        # Print results
        for result in self.results:
            overhead_str = f"{result.overhead_percentage:.1f}%" if result.overhead_percentage is not None else "N/A"

            print(f"{result.operation_name:<25} "
                  f"{result.mean_ms:<10.3f} "
                  f"{result.p95_ms:<9.3f} "
                  f"{result.ops_per_second:<10.0f} "
                  f"{overhead_str:<10}")

        print("-" * 80)

    def save_results(self, filename: str):
        """Save benchmark results to JSON file."""
        data = {
            "benchmark_metadata": {
                "timestamp": time.time(),
                "warmup_iterations": self.warmup_iterations,
                "test_iterations": self.test_iterations,
                "python_version": sys.version,
                "platform": sys.platform
            },
            "results": [result.to_dict() for result in self.results]
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"📊 Benchmark results saved to: {filename}")


class InstrumentationBenchmarkSuite:
    """Comprehensive benchmark suite for auto-instrumentation."""

    def __init__(self):
        self.benchmark = PerformanceBenchmark(warmup_iterations=100, test_iterations=2000)

    def run_all_benchmarks(self):
        """Run all benchmark tests."""
        print("🧪 Starting Auto-Instrumentation Performance Benchmarks...")

        if not BROKLE_AVAILABLE:
            print("❌ Brokle SDK not available, cannot run benchmarks")
            return

        # Core functionality benchmarks
        self._benchmark_core_operations()

        # Error handling benchmarks
        self._benchmark_error_handling()

        # Registry operations
        self._benchmark_registry_operations()

        # Instrumentation overhead
        self._benchmark_instrumentation_overhead()

        # Concurrency benchmarks
        self._benchmark_concurrency()

        # Memory usage benchmarks
        self._benchmark_memory_usage()

        print("\n✅ All benchmarks completed!")
        self.benchmark.print_results()

        # Save results
        timestamp = int(time.time())
        filename = f"benchmark_results_{timestamp}.json"
        self.benchmark.save_results(filename)

    def _benchmark_core_operations(self):
        """Benchmark core auto-instrumentation operations."""
        print("\n🔧 Benchmarking Core Operations...")

        # Reset state for consistent benchmarks
        brokle_ai.reset_all_errors()

        # Benchmark status operations
        def get_status():
            return brokle_ai.get_status()

        self.benchmark.benchmark_function(get_status, "get_status", 5000)

        def get_health_report():
            return brokle_ai.get_health_report()

        self.benchmark.benchmark_function(get_health_report, "get_health_report", 1000)

        # Benchmark registry access
        def get_registry():
            return brokle_ai.get_registry()

        self.benchmark.benchmark_function(get_registry, "get_registry", 5000)

        # Benchmark error handler access
        def get_error_handler():
            return brokle_ai.get_error_handler()

        self.benchmark.benchmark_function(get_error_handler, "get_error_handler", 5000)

    def _benchmark_error_handling(self):
        """Benchmark error handling components."""
        print("\n⚠️ Benchmarking Error Handling...")

        error_handler = get_error_handler()

        # Benchmark health checks
        def health_check():
            return error_handler.is_operation_healthy("test_lib", "test_op")

        self.benchmark.benchmark_function(health_check, "health_check", 10000)

        # Benchmark error tracking
        def handle_error():
            test_error = Exception("Benchmark error")
            error_handler.handle_error(test_error, "bench_lib", "bench_op", brokle_ai.ErrorSeverity.LOW)

        self.benchmark.benchmark_function(handle_error, "handle_error", 2000)

        # Benchmark circuit breaker
        circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=1)

        def circuit_breaker_call():
            return circuit_breaker.call(lambda: "test_result")

        self.benchmark.benchmark_function(circuit_breaker_call, "circuit_breaker_call", 5000)

    def _benchmark_registry_operations(self):
        """Benchmark registry operations."""
        print("\n📚 Benchmarking Registry Operations...")

        registry = brokle_ai.get_registry()

        # Benchmark library listing
        def list_libraries():
            return registry.list_libraries()

        self.benchmark.benchmark_function(list_libraries, "list_libraries", 5000)

        # Benchmark available libraries
        def get_available():
            return registry.get_available_libraries()

        self.benchmark.benchmark_function(get_available, "get_available_libraries", 5000)

        # Benchmark instrumentation summary
        def get_summary():
            return registry.get_instrumentation_summary()

        self.benchmark.benchmark_function(get_summary, "get_instrumentation_summary", 1000)

    def _benchmark_instrumentation_overhead(self):
        """Benchmark instrumentation overhead vs baseline."""
        print("\n⏱️ Benchmarking Instrumentation Overhead...")

        # Baseline function (simple computation)
        def baseline_function():
            return sum(range(100))

        # Function with safe_operation decorator
        @safe_operation("benchmark", "test_op", brokle_ai.ErrorSeverity.LOW)
        def decorated_function():
            return sum(range(100))

        self.benchmark.benchmark_overhead(
            baseline_function,
            decorated_function,
            "safe_operation_decorator"
        )

        # Context manager overhead
        def context_manager_function():
            with instrumentation_context("benchmark", "context_op", brokle_ai.ErrorSeverity.LOW):
                return sum(range(100))

        self.benchmark.benchmark_overhead(
            baseline_function,
            context_manager_function,
            "context_manager"
        )

    def _benchmark_concurrency(self):
        """Benchmark concurrent operations."""
        print("\n🔄 Benchmarking Concurrency...")

        def concurrent_status_checks():
            """Simulate concurrent status checks."""
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(brokle_ai.get_status) for _ in range(10)]
                results = [future.result() for future in as_completed(futures)]
                return len(results)

        self.benchmark.benchmark_function(concurrent_status_checks, "concurrent_status_10_threads", 100)

        def concurrent_health_checks():
            """Simulate concurrent health checks."""
            error_handler = get_error_handler()
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(error_handler.is_operation_healthy, f"lib_{i}", "op")
                    for i in range(20)
                ]
                results = [future.result() for future in as_completed(futures)]
                return len(results)

        self.benchmark.benchmark_function(concurrent_health_checks, "concurrent_health_5_threads", 200)

    def _benchmark_memory_usage(self):
        """Benchmark memory usage patterns."""
        print("\n🧠 Benchmarking Memory Usage...")

        def memory_intensive_operations():
            """Perform memory-intensive operations."""
            # Generate many errors
            error_handler = get_error_handler()
            for i in range(100):
                test_error = Exception(f"Memory test {i}")
                error_handler.handle_error(test_error, f"mem_lib_{i % 10}", f"mem_op_{i % 5}",
                                         brokle_ai.ErrorSeverity.LOW)

            # Check status multiple times
            results = []
            for _ in range(50):
                results.append(brokle_ai.get_status())
                results.append(brokle_ai.get_health_report())

            # Reset errors
            brokle_ai.reset_all_errors()

            return len(results)

        # Measure memory usage
        gc.collect()
        gc.disable()  # Disable GC during measurement
        try:
            self.benchmark.benchmark_function(memory_intensive_operations, "memory_intensive_ops", 10)
        finally:
            gc.enable()

    async def _benchmark_async_operations(self):
        """Benchmark async operations."""
        print("\n⚡ Benchmarking Async Operations...")

        @safe_async_operation("benchmark", "async_op", brokle_ai.ErrorSeverity.LOW)
        async def async_decorated_function():
            await asyncio.sleep(0.001)  # Simulate async work
            return sum(range(100))

        await self.benchmark.benchmark_async_function(async_decorated_function, "async_safe_operation", 500)

    async def run_async_benchmarks(self):
        """Run async-specific benchmarks."""
        await self._benchmark_async_operations()


def main():
    """Main benchmark execution."""
    print("🚀 Brokle Auto-Instrumentation Performance Benchmark Suite")
    print("=" * 60)

    # Check system info
    print(f"Python Version: {sys.version}")
    print(f"Platform: {sys.platform}")

    if BROKLE_AVAILABLE:
        print("✅ Brokle SDK available")
    else:
        print("❌ Brokle SDK not available")
        return 1

    try:
        # Initialize auto-instrumentation
        brokle_ai.reset_all_errors()

        # Run benchmark suite
        suite = InstrumentationBenchmarkSuite()
        suite.run_all_benchmarks()

        # Run async benchmarks
        print("\n🔄 Running async benchmarks...")
        asyncio.run(suite.run_async_benchmarks())

        print("\n🎯 Performance Summary:")
        print("- Target overhead: < 1ms per operation")
        print("- Target health check: < 0.1ms")
        print("- Target concurrency: > 1000 ops/sec")

        # Analyze results for warnings
        critical_operations = ["get_status", "health_check", "circuit_breaker_call"]
        for result in suite.benchmark.results:
            if any(op in result.operation_name for op in critical_operations):
                if result.mean_ms > 1.0:
                    print(f"⚠️ WARNING: {result.operation_name} slow: {result.mean_ms:.3f}ms")
                elif result.mean_ms > 0.5:
                    print(f"🟡 CAUTION: {result.operation_name} moderate: {result.mean_ms:.3f}ms")
                else:
                    print(f"✅ GOOD: {result.operation_name} fast: {result.mean_ms:.3f}ms")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️ Benchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)