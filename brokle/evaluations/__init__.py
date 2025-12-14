"""
Brokle Evaluations Manager

Provides evaluation and scoring functionality accessed via client.evaluations.

Example (Sync):
    >>> from brokle import Brokle
    >>> with Brokle(api_key="bk_...") as client:
    ...     # Future functionality:
    ...     # result = client.evaluations.run(trace_id, "accuracy")
    ...     # score = client.evaluations.score(span_id, "relevance", 0.95)
    ...     pass

Example (Async):
    >>> from brokle import AsyncBrokle
    >>> async with AsyncBrokle(api_key="bk_...") as client:
    ...     # Future functionality:
    ...     # result = await client.evaluations.run(trace_id, "accuracy")
    ...     # score = await client.evaluations.score(span_id, "relevance", 0.95)
    ...     pass

Note:
    This is a stub manager for future evaluation functionality.
    Methods will raise NotImplementedError until the API is ready.
"""

from ._managers import AsyncEvaluationsManager, EvaluationsManager

__all__ = [
    "AsyncEvaluationsManager",
    "EvaluationsManager",
]
