# Performance Benchmarking Guide

This guide covers the comprehensive performance benchmarking system for Brokle's auto-instrumentation.

## 🎯 Overview

The benchmarking system ensures that auto-instrumentation adds minimal overhead while maintaining production-ready performance characteristics.

## 📊 Performance Targets

| Operation | Target | Rationale |
|-----------|--------|-----------|
| **Core Operations** |
| `get_status()` | < 1ms | Frequent status checks should be instant |
| Health checks | < 0.1ms | Circuit breaker decisions must be fast |
| Registry access | < 0.5ms | Library management should be responsive |
| **Error Handling** |
| Error tracking | < 2ms | Error handling shouldn't impact performance |
| Circuit breaker | < 0.5ms | Protection must be faster than failure |
| **Instrumentation Overhead** |
| Decorator overhead | < 100% | Safety is worth 2x baseline performance |
| Context manager | < 150% | Context management acceptable overhead |
| **Concurrency** |
| Status checks | > 500 ops/sec | Multi-threaded applications support |
| Health monitoring | > 1000 ops/sec | High-frequency health checks |

## 🛠️ Benchmarking Tools

### 1. Comprehensive Benchmark Suite

```bash
# Run full benchmark suite
python benchmark_instrumentation.py

# Example output:
🚀 Auto-Instrumentation Performance Benchmark Results
================================================================================
Operation                 Mean (ms)  P95 (ms) Ops/sec    Overhead
--------------------------------------------------------------------------------
get_status               0.045      0.089     22222      N/A
health_check             0.023      0.045     43478      N/A
get_registry             0.067      0.125     14925      N/A
safe_operation_overhead  0.156      0.289     6410       45.2%
context_manager_overhead 0.201      0.378     4975       78.3%
--------------------------------------------------------------------------------
```

### 2. CI/CD Integration

```bash
# Run CI/CD performance validation
python scripts/benchmark_ci.py

# Validates against thresholds and exits with status code
# - 0: All performance thresholds passed
# - 1: One or more thresholds failed
```

### 3. GitHub Actions Workflow

The system includes automated performance testing via GitHub Actions:

- **Performance Benchmarks**: Run on every push/PR
- **Regression Detection**: Compare PR performance with baseline
- **Daily Monitoring**: Track performance trends over time
- **Load Testing**: Simulate high-throughput scenarios

## 📈 Benchmark Categories

### Core Operations
Tests fundamental auto-instrumentation operations:
- Status retrieval and health reporting
- Registry management and library queries
- Error handler access and circuit breaker status

### Error Handling Performance
Validates error handling components don't degrade performance:
- Error tracking and circuit breaker operations
- Health checks under various conditions
- Recovery mechanism performance

### Instrumentation Overhead
Measures the actual cost of instrumentation:
- Decorator overhead vs baseline functions
- Context manager performance impact
- Real-world usage pattern simulation

### Concurrency & Scalability
Tests performance under concurrent access:
- Multi-threaded status checks
- Concurrent health monitoring
- High-load simulation with thread pools

### Memory Usage Patterns
Monitors memory efficiency:
- Memory growth during extended operation
- Garbage collection impact
- Memory leak detection

## 🔧 Running Benchmarks

### Local Development

```bash
# Quick performance check
python -c "
import brokle.auto_instrumentation as brokle_ai
import time

start = time.time()
for i in range(1000):
    brokle_ai.get_status()
end = time.time()

avg_ms = (end - start) * 1000 / 1000
print(f'get_status: {avg_ms:.3f}ms average')
"

# Full benchmark suite
python benchmark_instrumentation.py

# Save results to file
python benchmark_instrumentation.py > benchmark_results.txt
```

### CI/CD Environment

```bash
# Validate performance thresholds
python scripts/benchmark_ci.py benchmark_output/

# Check exit code
echo "Exit code: $?"
```

### Custom Benchmarking

```python
from benchmark_instrumentation import PerformanceBenchmark

# Create custom benchmark
benchmark = PerformanceBenchmark(warmup_iterations=50, test_iterations=1000)

# Benchmark custom function
def my_function():
    # Your code here
    pass

result = benchmark.benchmark_function(my_function, "my_operation")
print(f"Average time: {result.mean_ms:.3f}ms")
```

## 📊 Understanding Results

### Key Metrics

- **Mean (ms)**: Average execution time - primary performance indicator
- **P95 (ms)**: 95th percentile - represents worst-case performance
- **Ops/sec**: Operations per second - throughput capability
- **Overhead (%)**: Additional cost vs baseline - instrumentation impact

### Performance Analysis

```python
# Analyze benchmark results
def analyze_performance(results):
    for result in results:
        # Performance categories
        if result.mean_ms < 0.1:
            print(f"✅ {result.operation_name}: Excellent ({result.mean_ms:.3f}ms)")
        elif result.mean_ms < 1.0:
            print(f"🟢 {result.operation_name}: Good ({result.mean_ms:.3f}ms)")
        elif result.mean_ms < 5.0:
            print(f"🟡 {result.operation_name}: Acceptable ({result.mean_ms:.3f}ms)")
        else:
            print(f"🔴 {result.operation_name}: Slow ({result.mean_ms:.3f}ms)")

        # Overhead analysis
        if result.overhead_percentage:
            if result.overhead_percentage < 50:
                print(f"   Low overhead: {result.overhead_percentage:.1f}%")
            elif result.overhead_percentage < 100:
                print(f"   Moderate overhead: {result.overhead_percentage:.1f}%")
            else:
                print(f"   High overhead: {result.overhead_percentage:.1f}%")
```

## 🚨 Performance Alerts

### Threshold Violations

When benchmarks exceed thresholds:

1. **Immediate**: CI/CD fails, preventing deployment
2. **Investigation**: Review recent changes affecting performance
3. **Optimization**: Profile and optimize slow operations
4. **Validation**: Re-run benchmarks to confirm improvements

### Common Performance Issues

| Issue | Symptoms | Solutions |
|-------|----------|-----------|
| **Slow Status Checks** | > 1ms mean time | Cache status results, optimize queries |
| **High Overhead** | > 100% instrumentation cost | Reduce logging, optimize decorators |
| **Memory Leaks** | Growing memory usage | Check error tracking cleanup |
| **Concurrency Issues** | Low ops/sec under load | Review locking, use lock-free data structures |

## 📈 Performance Optimization

### Optimization Strategies

1. **Lazy Loading**: Defer expensive operations until needed
2. **Caching**: Cache frequently accessed data
3. **Circuit Breakers**: Prevent cascading failures
4. **Lock-Free**: Minimize synchronization overhead
5. **Memory Pools**: Reuse objects to reduce GC pressure

### Example Optimizations

```python
# Before: Direct database query each time
def get_status_slow():
    return database.query("SELECT status FROM instrumentation")

# After: Cached with TTL
@lru_cache(maxsize=1)
def get_status_fast():
    return database.query("SELECT status FROM instrumentation")
```

## 🔍 Profiling & Debugging

### Performance Profiling

```python
import cProfile
import brokle.auto_instrumentation as brokle_ai

# Profile specific operations
pr = cProfile.Profile()
pr.enable()

for i in range(1000):
    brokle_ai.get_status()

pr.disable()
pr.print_stats(sort='cumulative')
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
def memory_test():
    import brokle.auto_instrumentation as brokle_ai

    # Perform operations
    for i in range(100):
        brokle_ai.get_status()
        brokle_ai.get_health_report()

        if i % 10 == 0:
            brokle_ai.reset_all_errors()

# Run: python -m memory_profiler script.py
```

## 📋 Performance Checklist

### Pre-Release Validation

- [ ] All core operations under threshold
- [ ] Instrumentation overhead acceptable
- [ ] Concurrency targets met
- [ ] No memory leaks detected
- [ ] Load testing passed
- [ ] Regression tests clean

### Monitoring in Production

- [ ] Performance metrics collection enabled
- [ ] Alerting configured for threshold violations
- [ ] Regular benchmark execution scheduled
- [ ] Performance trend analysis automated
- [ ] Capacity planning data collected

## 🎛️ Configuration

### Environment Variables

```bash
# Benchmark configuration
export BENCHMARK_ITERATIONS=2000
export BENCHMARK_WARMUP=100
export BENCHMARK_OUTPUT_DIR="./benchmark_results"

# Performance thresholds
export PERF_THRESHOLD_STATUS_MS=1.0
export PERF_THRESHOLD_HEALTH_MS=0.1
export PERF_THRESHOLD_OVERHEAD_PCT=100.0
```

### Custom Thresholds

```python
from scripts.benchmark_ci import PerformanceThresholds

# Custom thresholds for specific environment
class ProductionThresholds(PerformanceThresholds):
    GET_STATUS_MAX = 0.5      # Stricter for production
    HEALTH_CHECK_MAX = 0.05   # Ultra-fast health checks
    SAFE_OPERATION_OVERHEAD_MAX = 75.0  # Lower overhead tolerance

# Apply custom thresholds
thresholds = ProductionThresholds()
```

## 🔗 Integration Examples

### Flask Application

```python
from flask import Flask, jsonify
import brokle.auto_instrumentation as brokle_ai
import time

app = Flask(__name__)

@app.route('/health/performance')
def performance_health():
    """Performance health endpoint."""
    start_time = time.time()

    # Test core operations
    status = brokle_ai.get_status()
    health = brokle_ai.get_health_report()

    end_time = time.time()
    response_time_ms = (end_time - start_time) * 1000

    return jsonify({
        "performance_health": "healthy" if response_time_ms < 10 else "degraded",
        "response_time_ms": response_time_ms,
        "overall_health_score": health["overall_health"]["score"],
        "instrumented_libraries": len([s for s in status.values() if s.value == "instrumented"])
    })
```

### Monitoring Dashboard

```python
import time
import brokle.auto_instrumentation as brokle_ai

def collect_performance_metrics():
    """Collect real-time performance metrics."""
    metrics = {}

    # Time status operation
    start = time.perf_counter()
    status = brokle_ai.get_status()
    metrics['status_time_ms'] = (time.perf_counter() - start) * 1000

    # Time health report
    start = time.perf_counter()
    health = brokle_ai.get_health_report()
    metrics['health_time_ms'] = (time.perf_counter() - start) * 1000

    # Add health score
    metrics['health_score'] = health['overall_health']['score']

    return metrics

# Use with your monitoring system
metrics = collect_performance_metrics()
# send_to_datadog(metrics)
# send_to_prometheus(metrics)
```

---

*Performance benchmarking ensures Brokle auto-instrumentation maintains excellent performance characteristics in production environments.*