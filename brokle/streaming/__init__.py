"""
Brokle Streaming Module.

Provides streaming accumulator and metrics for LLM streaming responses,
including time-to-first-token (TTFT) tracking and content accumulation.

Example:
    >>> from brokle.streaming import StreamingAccumulator
    >>> accumulator = StreamingAccumulator()
    >>> for chunk in stream:
    ...     accumulator.on_chunk(chunk)
    >>> result = accumulator.finalize()
    >>> print(f"TTFT: {result.ttft_ms}ms")
"""

from .accumulator import StreamingAccumulator, StreamingResult
from .metrics import StreamingMetrics

__all__ = [
    "StreamingAccumulator",
    "StreamingResult",
    "StreamingMetrics",
]
