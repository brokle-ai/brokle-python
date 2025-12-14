"""
Evaluations Manager

Provides both synchronous and asynchronous evaluation operations for Brokle.

Sync Usage:
    >>> with Brokle(api_key="bk_...") as client:
    ...     # Future: result = client.evaluations.run(trace_id, "accuracy")
    ...     pass

Async Usage:
    >>> async with AsyncBrokle(api_key="bk_...") as client:
    ...     # Future: result = await client.evaluations.run(trace_id, "accuracy")
    ...     pass

Note:
    This is a stub manager for future evaluation functionality.
    Methods will raise NotImplementedError until the API is ready.
"""

from typing import Any, Dict

from ._base import BaseAsyncEvaluationsManager, BaseSyncEvaluationsManager


class EvaluationsManager(BaseSyncEvaluationsManager):
    """
    Sync evaluations manager for Brokle.

    All methods are synchronous. Uses SyncHTTPClient (httpx.Client) internally -
    no event loop involvement.

    Example:
        >>> with Brokle(api_key="bk_...") as client:
        ...     # Future: result = client.evaluations.run(trace_id, "accuracy")
        ...     pass
    """

    def run(
        self,
        trace_id: str,
        evaluator: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run an evaluation on a trace.

        Args:
            trace_id: Trace ID to evaluate
            evaluator: Evaluator name
            **kwargs: Additional evaluator-specific parameters

        Returns:
            Evaluation result

        Raises:
            NotImplementedError: This is a stub for future functionality

        Note:
            This is a stub implementation. Will be implemented when
            evaluation API is ready.
        """
        return self._run(trace_id, evaluator, **kwargs)

    def score(
        self,
        span_id: str,
        name: str,
        value: float,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Submit a quality score for a span.

        Args:
            span_id: Span ID to score
            name: Score name (e.g., 'accuracy', 'relevance')
            value: Score value
            **kwargs: Additional metadata

        Returns:
            Score submission result

        Raises:
            NotImplementedError: This is a stub for future functionality

        Note:
            This is a stub implementation. Will be implemented when
            scoring API is ready.
        """
        return self._score(span_id, name, value, **kwargs)


class AsyncEvaluationsManager(BaseAsyncEvaluationsManager):
    """
    Async evaluations manager for AsyncBrokle.

    All methods are async and return coroutines that must be awaited.
    Uses AsyncHTTPClient (httpx.AsyncClient) internally.

    Example:
        >>> async with AsyncBrokle(api_key="bk_...") as client:
        ...     # Future: result = await client.evaluations.run(trace_id, "accuracy")
        ...     pass
    """

    async def run(
        self,
        trace_id: str,
        evaluator: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run an evaluation on a trace.

        Args:
            trace_id: Trace ID to evaluate
            evaluator: Evaluator name
            **kwargs: Additional evaluator-specific parameters

        Returns:
            Evaluation result

        Raises:
            NotImplementedError: This is a stub for future functionality

        Note:
            This is a stub implementation. Will be implemented when
            evaluation API is ready.
        """
        return await self._run(trace_id, evaluator, **kwargs)

    async def score(
        self,
        span_id: str,
        name: str,
        value: float,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Submit a quality score for a span.

        Args:
            span_id: Span ID to score
            name: Score name (e.g., 'accuracy', 'relevance')
            value: Score value
            **kwargs: Additional metadata

        Returns:
            Score submission result

        Raises:
            NotImplementedError: This is a stub for future functionality

        Note:
            This is a stub implementation. Will be implemented when
            scoring API is ready.
        """
        return await self._score(span_id, name, value, **kwargs)
