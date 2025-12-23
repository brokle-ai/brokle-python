"""
Evaluations Manager

Provides both synchronous and asynchronous evaluation operations for Brokle.

Supports two scoring modes:
1. Direct score: Pass name + value directly
2. Scorer function: Pass a scorer callable with output/expected

Sync Usage:
    >>> from brokle import Brokle, ScoreType
    >>> from brokle.scorers import ExactMatch
    >>>
    >>> client = Brokle(api_key="bk_...")
    >>>
    >>> # Direct score
    >>> client.evaluations.score(
    ...     trace_id="abc123",
    ...     name="quality",
    ...     value=0.9,
    ... )
    >>>
    >>> # With scorer
    >>> exact = ExactMatch()
    >>> client.evaluations.score(
    ...     trace_id="abc123",
    ...     scorer=exact,
    ...     output="Paris",
    ...     expected="Paris",
    ... )

Async Usage:
    >>> async with AsyncBrokle(api_key="bk_...") as client:
    ...     await client.evaluations.score(trace_id="abc123", name="quality", value=0.9)
"""

from typing import Any, Dict, List, Optional, Union

from ._base import BaseAsyncEvaluationsManager, BaseSyncEvaluationsManager
from .types import ScoreResult, ScoreSource, ScoreType, ScoreValue, ScorerProtocol


class EvaluationsManager(BaseSyncEvaluationsManager):
    """
    Sync evaluations manager for Brokle.

    All methods are synchronous. Uses SyncHTTPClient (httpx.Client) internally -
    no event loop involvement.

    Example:
        >>> from brokle import Brokle
        >>> from brokle.scorers import ExactMatch
        >>>
        >>> client = Brokle(api_key="bk_...")
        >>>
        >>> # Direct score
        >>> client.evaluations.score(
        ...     trace_id="abc123",
        ...     name="accuracy",
        ...     value=0.95,
        ... )
        >>>
        >>> # With built-in scorer
        >>> exact = ExactMatch(name="answer_match")
        >>> client.evaluations.score(
        ...     trace_id="abc123",
        ...     scorer=exact,
        ...     output="4",
        ...     expected="4",
        ... )
    """

    def score(
        self,
        trace_id: str,
        scorer: Optional[ScorerProtocol] = None,
        output: Any = None,
        expected: Any = None,
        name: Optional[str] = None,
        value: Optional[float] = None,
        type: ScoreType = ScoreType.NUMERIC,
        source: ScoreSource = ScoreSource.CODE,
        span_id: Optional[str] = None,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Score a trace or span.

        Two modes:
        1. With scorer: Pass scorer callable + output/expected
        2. Direct: Pass name + value directly

        Args:
            trace_id: Trace ID to score
            scorer: Optional scorer callable (ExactMatch, Contains, custom @scorer)
            output: The actual output to evaluate (for scorer mode)
            expected: The expected/reference output (for scorer mode)
            name: Score name (required for direct mode)
            value: Score value (required for direct mode)
            type: Score type (NUMERIC, CATEGORICAL, BOOLEAN)
            source: Score source (code, llm, human)
            span_id: Optional span ID for span-level scoring
            reason: Human-readable explanation
            metadata: Additional metadata
            **kwargs: Additional arguments passed to scorer

        Returns:
            Single score dict or list of score dicts (if scorer returns List[ScoreResult])

        Raises:
            ValueError: If neither scorer nor (name + value) are provided
            ScoreError: If the API request fails

        Example:
            >>> # Direct score
            >>> client.evaluations.score(
            ...     trace_id="abc123",
            ...     name="quality",
            ...     value=0.9,
            ...     reason="High quality response",
            ... )
            >>>
            >>> # With scorer
            >>> from brokle.scorers import ExactMatch
            >>> exact = ExactMatch()
            >>> client.evaluations.score(
            ...     trace_id="abc123",
            ...     scorer=exact,
            ...     output="Paris",
            ...     expected="Paris",
            ... )
        """
        if scorer is not None:
            try:
                result = scorer(output=output, expected=expected, **kwargs)
            except Exception as e:
                scorer_name = (
                    getattr(scorer, "name", None)
                    or getattr(scorer, "__name__", "unknown")
                )
                return self._score(
                    trace_id=trace_id,
                    name=scorer_name,
                    value=0.0,
                    type="NUMERIC",
                    source=source.value,
                    span_id=span_id,
                    reason=f"Scorer failed: {str(e)}",
                    metadata={"scoring_failed": True, "error": str(e)},
                )

            results = self._normalize_score_result(result, scorer)

            responses: List[Dict[str, Any]] = []
            for score_result in results:
                resp = self._score(
                    trace_id=trace_id,
                    name=score_result.name,
                    value=score_result.value,
                    type=score_result.type.value,
                    source=source.value,
                    span_id=span_id,
                    string_value=score_result.string_value,
                    reason=score_result.reason or reason,
                    metadata=score_result.metadata or metadata,
                )
                responses.append(resp)

            return responses[0] if len(responses) == 1 else responses
        else:
            if name is None or value is None:
                raise ValueError("name and value required when not using scorer")
            return self._score(
                trace_id=trace_id,
                name=name,
                value=value,
                type=type.value,
                source=source.value,
                span_id=span_id,
                reason=reason,
                metadata=metadata,
            )

    def score_batch(
        self,
        scores: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Submit multiple scores to the API.

        Args:
            scores: List of score dictionaries with keys:
                - trace_id: Trace ID (required)
                - name: Score name (required)
                - value: Score value (required)
                - type: Score type (optional, default: "NUMERIC")
                - source: Score source (optional, default: "code")
                - span_id: Span ID (optional)
                - string_value: String value (optional)
                - reason: Reason (optional)
                - metadata: Metadata (optional)

        Returns:
            Batch submission result

        Example:
            >>> client.evaluations.score_batch([
            ...     {"trace_id": "abc123", "name": "accuracy", "value": 0.9},
            ...     {"trace_id": "abc123", "name": "relevance", "value": 0.85},
            ... ])
        """
        return self._score_batch(scores)

    def _normalize_score_result(
        self, result: ScoreValue, scorer: ScorerProtocol
    ) -> List[ScoreResult]:
        """Normalize any scorer return type to List[ScoreResult]."""
        scorer_name = (
            getattr(scorer, "name", None) or getattr(scorer, "__name__", "scorer")
        )

        if isinstance(result, list):
            return result
        elif isinstance(result, ScoreResult):
            return [result]
        elif isinstance(result, bool):
            return [
                ScoreResult(
                    name=scorer_name, value=1.0 if result else 0.0, type=ScoreType.BOOLEAN
                )
            ]
        elif isinstance(result, (int, float)):
            return [ScoreResult(name=scorer_name, value=float(result))]
        else:
            raise TypeError(
                f"Scorer must return ScoreResult, List[ScoreResult], float, or bool, "
                f"got {type(result).__name__}"
            )

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


class AsyncEvaluationsManager(BaseAsyncEvaluationsManager):
    """
    Async evaluations manager for AsyncBrokle.

    All methods are async and return coroutines that must be awaited.
    Uses AsyncHTTPClient (httpx.AsyncClient) internally.

    Example:
        >>> async with AsyncBrokle(api_key="bk_...") as client:
        ...     await client.evaluations.score(
        ...         trace_id="abc123",
        ...         name="quality",
        ...         value=0.9,
        ...     )
    """

    async def score(
        self,
        trace_id: str,
        scorer: Optional[ScorerProtocol] = None,
        output: Any = None,
        expected: Any = None,
        name: Optional[str] = None,
        value: Optional[float] = None,
        type: ScoreType = ScoreType.NUMERIC,
        source: ScoreSource = ScoreSource.CODE,
        span_id: Optional[str] = None,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Score a trace or span (async).

        Two modes:
        1. With scorer: Pass scorer callable + output/expected
        2. Direct: Pass name + value directly

        Args:
            trace_id: Trace ID to score
            scorer: Optional scorer callable (ExactMatch, Contains, custom @scorer)
            output: The actual output to evaluate (for scorer mode)
            expected: The expected/reference output (for scorer mode)
            name: Score name (required for direct mode)
            value: Score value (required for direct mode)
            type: Score type (NUMERIC, CATEGORICAL, BOOLEAN)
            source: Score source (code, llm, human)
            span_id: Optional span ID for span-level scoring
            reason: Human-readable explanation
            metadata: Additional metadata
            **kwargs: Additional arguments passed to scorer

        Returns:
            Single score dict or list of score dicts (if scorer returns List[ScoreResult])

        Raises:
            ValueError: If neither scorer nor (name + value) are provided
            ScoreError: If the API request fails
        """
        if scorer is not None:
            try:
                result = scorer(output=output, expected=expected, **kwargs)
            except Exception as e:
                scorer_name = (
                    getattr(scorer, "name", None)
                    or getattr(scorer, "__name__", "unknown")
                )
                return await self._score(
                    trace_id=trace_id,
                    name=scorer_name,
                    value=0.0,
                    type="NUMERIC",
                    source=source.value,
                    span_id=span_id,
                    reason=f"Scorer failed: {str(e)}",
                    metadata={"scoring_failed": True, "error": str(e)},
                )

            results = self._normalize_score_result(result, scorer)

            responses: List[Dict[str, Any]] = []
            for score_result in results:
                resp = await self._score(
                    trace_id=trace_id,
                    name=score_result.name,
                    value=score_result.value,
                    type=score_result.type.value,
                    source=source.value,
                    span_id=span_id,
                    string_value=score_result.string_value,
                    reason=score_result.reason or reason,
                    metadata=score_result.metadata or metadata,
                )
                responses.append(resp)

            return responses[0] if len(responses) == 1 else responses
        else:
            if name is None or value is None:
                raise ValueError("name and value required when not using scorer")
            return await self._score(
                trace_id=trace_id,
                name=name,
                value=value,
                type=type.value,
                source=source.value,
                span_id=span_id,
                reason=reason,
                metadata=metadata,
            )

    async def score_batch(
        self,
        scores: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Submit multiple scores to the API (async).

        Args:
            scores: List of score dictionaries with keys:
                - trace_id: Trace ID (required)
                - name: Score name (required)
                - value: Score value (required)
                - type: Score type (optional, default: "NUMERIC")
                - source: Score source (optional, default: "code")
                - span_id: Span ID (optional)
                - string_value: String value (optional)
                - reason: Reason (optional)
                - metadata: Metadata (optional)

        Returns:
            Batch submission result
        """
        return await self._score_batch(scores)

    def _normalize_score_result(
        self, result: ScoreValue, scorer: ScorerProtocol
    ) -> List[ScoreResult]:
        """Normalize any scorer return type to List[ScoreResult]."""
        scorer_name = (
            getattr(scorer, "name", None) or getattr(scorer, "__name__", "scorer")
        )

        if isinstance(result, list):
            return result
        elif isinstance(result, ScoreResult):
            return [result]
        elif isinstance(result, bool):
            return [
                ScoreResult(
                    name=scorer_name, value=1.0 if result else 0.0, type=ScoreType.BOOLEAN
                )
            ]
        elif isinstance(result, (int, float)):
            return [ScoreResult(name=scorer_name, value=float(result))]
        else:
            raise TypeError(
                f"Scorer must return ScoreResult, List[ScoreResult], float, or bool, "
                f"got {type(result).__name__}"
            )

    async def run(
        self,
        trace_id: str,
        evaluator: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run an evaluation on a trace (async).

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
