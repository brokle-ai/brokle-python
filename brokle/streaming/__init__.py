"""
Brokle Streaming Module.

Provides streaming accumulator, wrappers, and metrics for LLM streaming responses,
including time-to-first-token (TTFT) tracking and content accumulation.

Stream Wrappers (Recommended for wrap_openai/wrap_anthropic):
    BrokleStreamWrapper and BrokleAsyncStreamWrapper provide transparent
    instrumentation - users iterate normally and metrics are auto-recorded.

    >>> # With wrap_openai(), streaming is automatically instrumented
    >>> stream = client.chat.completions.create(..., stream=True)
    >>> for chunk in stream:  # Metrics tracked automatically
    ...     print(chunk.choices[0].delta.content)
    >>> # TTFT, ITL, content all recorded when stream ends

Manual Accumulator (For custom integrations):
    >>> import time
    >>> from brokle.streaming import StreamingAccumulator
    >>> accumulator = StreamingAccumulator(time.perf_counter())
    >>> for chunk in stream:
    ...     accumulator.on_chunk(chunk)
    >>> result = accumulator.finalize()
    >>> print(f"TTFT: {result.ttft_ms}ms")
"""

from .accumulator import StreamingAccumulator, StreamingResult
from .metrics import StreamingMetrics
from .wrappers import BrokleAsyncStreamWrapper, BrokleStreamWrapper

__all__ = [
    "StreamingAccumulator",
    "StreamingResult",
    "BrokleStreamWrapper",
    "BrokleAsyncStreamWrapper",
    "StreamingMetrics",
]
