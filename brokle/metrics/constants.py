"""
Metrics constants and bucket boundaries.

Defines metric names following OTEL GenAI semantic conventions and
custom bucket boundaries optimized for LLM workloads.

Bucket boundary design principles (based on vLLM, OpenTelemetry best practices):
- Token counts: 2x exponential scaling (aligns with LLM context windows)
- Latency: SLO-aligned with denser buckets near critical thresholds
- TTFT: Dense sub-100ms coverage (critical for streaming UX)
- 10-20 buckets per metric to balance precision and cardinality
"""


class MetricNames:
    """
    OpenTelemetry GenAI metric name constants.

    Follows OTEL semantic conventions for GenAI metrics:
    https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/
    """

    # ========== Token Usage Metrics ==========
    # Histogram of token counts per operation
    TOKEN_USAGE = "gen_ai.client.token.usage"

    # Separate counters for cumulative tracking
    INPUT_TOKENS = "gen_ai.client.input_tokens"
    OUTPUT_TOKENS = "gen_ai.client.output_tokens"
    TOTAL_TOKENS = "gen_ai.client.total_tokens"

    # ========== Duration Metrics ==========
    # Histogram of operation durations
    OPERATION_DURATION = "gen_ai.client.operation.duration"

    # Time to first token (streaming)
    TIME_TO_FIRST_TOKEN = "gen_ai.client.time_to_first_token"

    # Inter-token latency (streaming)
    INTER_TOKEN_LATENCY = "gen_ai.client.inter_token_latency"

    # ========== Request Metrics ==========
    # Counter of total requests
    REQUEST_COUNT = "gen_ai.client.request.count"

    # Counter of errors
    ERROR_COUNT = "gen_ai.client.error.count"

    # ========== Brokle Custom Metrics ==========
    # Cache hit ratio for prompt caching
    CACHE_HIT_RATIO = "brokle.cache_hit_ratio"

    # Request queue depth (for rate limiting insights)
    QUEUE_DEPTH = "brokle.queue_depth"


# ========== Token Count Bucket Boundaries ==========
# 2x exponential scaling (powers of 2) - aligns with LLM context windows
# Based on vLLM metrics implementation
# Covers: 1 token → 128k+ context windows
TOKEN_BOUNDARIES = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,  # Tiny (1-64 tokens)
    128,
    256,
    512,
    1024,  # Small (128-1k)
    2048,
    4096,
    8192,  # Medium (2k-8k)
    16384,
    32768,
    65536,  # Large context (16k-64k)
    131072,  # Very large (128k+)
]

# ========== Duration Bucket Boundaries (milliseconds) ==========
# SLO-aligned with denser buckets near typical LLM response times
# Based on OpenTelemetry defaults + LLM-specific thresholds
# Covers: 10ms cache hits → 30s slow operations
DURATION_BOUNDARIES = [
    10,
    25,
    50,
    75,
    100,  # Fast (10-100ms) - cache/simple
    150,
    200,
    300,
    500,  # Typical (150-500ms) - short responses
    750,
    1000,
    2000,  # Normal (750ms-2s) - medium responses
    5000,
    10000,
    30000,  # Slow (5-30s) - long generations
]

# ========== Time to First Token Boundaries (milliseconds) ==========
# Dense sub-100ms coverage (critical for streaming UX)
# Based on vLLM TTFT buckets
# Covers: 10ms → 10s (cold starts)
TTFT_BOUNDARIES = [
    10,
    25,
    50,
    75,
    100,  # Critical UX range (10-100ms)
    150,
    200,
    300,
    500,  # Acceptable (150-500ms)
    750,
    1000,
    2000,  # Slow (750ms-2s)
    5000,
    10000,  # Very slow / cold start (5-10s)
]

# ========== Inter-Token Latency Boundaries (milliseconds) ==========
# Dense low-latency coverage for streaming quality monitoring
# Covers: 1ms fast streaming → 500ms throttled
INTER_TOKEN_BOUNDARIES = [
    1,
    2,
    5,
    10,
    15,  # Fast streaming (1-15ms)
    20,
    30,
    50,
    75,
    100,  # Normal streaming (20-100ms)
    150,
    200,
    300,
    500,  # Slow / throttled (150-500ms)
]
