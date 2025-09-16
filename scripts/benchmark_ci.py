#!/usr/bin/env python3
"""
CI/CD Performance Benchmarking Script.

This script runs performance benchmarks in CI/CD environments and
validates that performance meets required thresholds.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

try:
    from benchmark_instrumentation import InstrumentationBenchmarkSuite, BenchmarkResult
    import brokle.auto_instrumentation as brokle_ai
    BENCHMARK_AVAILABLE = True
except ImportError as e:
    print(f"❌ Benchmark dependencies not available: {e}")
    BENCHMARK_AVAILABLE = False


class PerformanceThresholds:
    """Performance thresholds for CI/CD validation."""

    # Core operation thresholds (ms)
    GET_STATUS_MAX = 1.0
    HEALTH_CHECK_MAX = 0.1
    GET_REGISTRY_MAX = 0.5
    GET_ERROR_HANDLER_MAX = 0.1
    GET_HEALTH_REPORT_MAX = 5.0

    # Error handling thresholds (ms)
    HANDLE_ERROR_MAX = 2.0
    CIRCUIT_BREAKER_CALL_MAX = 0.5

    # Instrumentation overhead thresholds (%)
    SAFE_OPERATION_OVERHEAD_MAX = 100.0  # 100% overhead is acceptable for safety
    CONTEXT_MANAGER_OVERHEAD_MAX = 150.0

    # Concurrency thresholds (ops/sec)
    CONCURRENT_STATUS_MIN_OPS = 500
    CONCURRENT_HEALTH_MIN_OPS = 1000

    # Memory usage thresholds (ms)
    MEMORY_INTENSIVE_MAX = 50.0


class CIBenchmarkRunner:
    """CI/CD benchmark runner with validation."""

    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.thresholds = PerformanceThresholds()
        self.failures: List[str] = []

    def run_benchmarks(self) -> bool:
        """Run benchmarks and validate against thresholds."""
        if not BENCHMARK_AVAILABLE:
            print("❌ Benchmark dependencies not available")
            return False

        print("🚀 Running CI/CD Performance Benchmarks...")
        print("=" * 50)

        try:
            # Initialize and reset state
            brokle_ai.reset_all_errors()

            # Run benchmark suite
            suite = InstrumentationBenchmarkSuite()
            suite.run_all_benchmarks()

            # Validate results
            self._validate_results(suite.benchmark.results)

            # Save results
            self._save_results(suite.benchmark.results)

            # Print summary
            self._print_summary()

            return len(self.failures) == 0

        except Exception as e:
            print(f"❌ Benchmark execution failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _validate_results(self, results: List[BenchmarkResult]) -> None:
        """Validate benchmark results against thresholds."""
        print("\n🔍 Validating Performance Thresholds...")

        for result in results:
            self._validate_single_result(result)

    def _validate_single_result(self, result: BenchmarkResult) -> None:
        """Validate a single benchmark result."""
        name = result.operation_name
        mean_ms = result.mean_ms
        ops_per_second = result.ops_per_second
        overhead_pct = result.overhead_percentage

        # Core operations validation
        if "get_status" in name:
            self._check_threshold(name, mean_ms, self.thresholds.GET_STATUS_MAX, "ms")

        elif "health_check" in name:
            self._check_threshold(name, mean_ms, self.thresholds.HEALTH_CHECK_MAX, "ms")

        elif "get_registry" in name:
            self._check_threshold(name, mean_ms, self.thresholds.GET_REGISTRY_MAX, "ms")

        elif "get_error_handler" in name:
            self._check_threshold(name, mean_ms, self.thresholds.GET_ERROR_HANDLER_MAX, "ms")

        elif "get_health_report" in name:
            self._check_threshold(name, mean_ms, self.thresholds.GET_HEALTH_REPORT_MAX, "ms")

        # Error handling validation
        elif "handle_error" in name:
            self._check_threshold(name, mean_ms, self.thresholds.HANDLE_ERROR_MAX, "ms")

        elif "circuit_breaker_call" in name:
            self._check_threshold(name, mean_ms, self.thresholds.CIRCUIT_BREAKER_CALL_MAX, "ms")

        # Overhead validation
        elif "overhead" in name and overhead_pct is not None:
            if "safe_operation" in name:
                self._check_threshold(name, overhead_pct, self.thresholds.SAFE_OPERATION_OVERHEAD_MAX, "%")
            elif "context_manager" in name:
                self._check_threshold(name, overhead_pct, self.thresholds.CONTEXT_MANAGER_OVERHEAD_MAX, "%")

        # Concurrency validation (check ops/sec minimums)
        elif "concurrent_status" in name:
            self._check_minimum(name, ops_per_second, self.thresholds.CONCURRENT_STATUS_MIN_OPS, "ops/sec")

        elif "concurrent_health" in name:
            self._check_minimum(name, ops_per_second, self.thresholds.CONCURRENT_HEALTH_MIN_OPS, "ops/sec")

        # Memory usage validation
        elif "memory_intensive" in name:
            self._check_threshold(name, mean_ms, self.thresholds.MEMORY_INTENSIVE_MAX, "ms")

    def _check_threshold(self, name: str, value: float, threshold: float, unit: str) -> None:
        """Check if value is below threshold."""
        if value > threshold:
            failure = f"{name}: {value:.3f}{unit} > {threshold}{unit} threshold"
            self.failures.append(failure)
            print(f"❌ FAIL: {failure}")
        else:
            print(f"✅ PASS: {name}: {value:.3f}{unit} <= {threshold}{unit}")

    def _check_minimum(self, name: str, value: float, minimum: float, unit: str) -> None:
        """Check if value meets minimum requirement."""
        if value < minimum:
            failure = f"{name}: {value:.0f}{unit} < {minimum}{unit} minimum"
            self.failures.append(failure)
            print(f"❌ FAIL: {failure}")
        else:
            print(f"✅ PASS: {name}: {value:.0f}{unit} >= {minimum}{unit}")

    def _save_results(self, results: List[BenchmarkResult]) -> None:
        """Save benchmark results to files."""
        timestamp = int(time.time())

        # Save JSON results
        json_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        data = {
            "timestamp": timestamp,
            "ci_run": True,
            "python_version": sys.version,
            "platform": sys.platform,
            "results": [result.to_dict() for result in results],
            "validation": {
                "passed": len(self.failures) == 0,
                "failures": self.failures
            }
        }

        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"📊 Results saved to: {json_file}")

        # Save simple text report
        txt_file = self.output_dir / f"benchmark_report_{timestamp}.txt"
        with open(txt_file, 'w') as f:
            f.write("Auto-Instrumentation Performance Benchmark Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {time.ctime(timestamp)}\n")
            f.write(f"Python Version: {sys.version}\n")
            f.write(f"Platform: {sys.platform}\n\n")

            f.write("Results:\n")
            f.write("-" * 30 + "\n")
            for result in results:
                f.write(f"{result.operation_name:<30} {result.mean_ms:>8.3f}ms\n")

            f.write(f"\nValidation: {'PASSED' if len(self.failures) == 0 else 'FAILED'}\n")
            if self.failures:
                f.write("Failures:\n")
                for failure in self.failures:
                    f.write(f"  - {failure}\n")

        print(f"📄 Report saved to: {txt_file}")

    def _print_summary(self) -> None:
        """Print validation summary."""
        print("\n" + "=" * 50)
        print("🎯 CI/CD Performance Validation Summary")
        print("=" * 50)

        if not self.failures:
            print("✅ ALL PERFORMANCE THRESHOLDS PASSED!")
            print("🚀 Auto-instrumentation performance is within acceptable limits.")
        else:
            print(f"❌ {len(self.failures)} PERFORMANCE THRESHOLDS FAILED:")
            for i, failure in enumerate(self.failures, 1):
                print(f"  {i}. {failure}")

        print("=" * 50)


def main():
    """Main CI/CD benchmark execution."""
    # Handle command line arguments
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results"

    print("🔬 Brokle Auto-Instrumentation CI/CD Performance Validation")
    print("=" * 60)
    print(f"Output Directory: {output_dir}")
    print(f"Python Version: {sys.version}")
    print(f"Platform: {sys.platform}")

    runner = CIBenchmarkRunner(output_dir)
    success = runner.run_benchmarks()

    if success:
        print("\n🎉 Performance validation PASSED!")
        return 0
    else:
        print("\n💥 Performance validation FAILED!")
        print("Consider optimizing slow operations or adjusting thresholds.")
        return 1


if __name__ == "__main__":
    sys.exit(main())