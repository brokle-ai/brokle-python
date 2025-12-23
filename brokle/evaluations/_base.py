"""
Base Evaluations Managers

Provides separate sync and async base implementations for evaluation operations.

Architecture:
- BaseSyncEvaluationsManager: Uses SyncHTTPClient (no event loop)
- BaseAsyncEvaluationsManager: Uses AsyncHTTPClient (async/await)

This design eliminates event loop lifecycle issues.
"""

from typing import Any, Dict, List, Optional

from .._http import AsyncHTTPClient, SyncHTTPClient, unwrap_response
from ..config import BrokleConfig
from .exceptions import ScoreError


class _BaseEvaluationsManagerMixin:
    """
    Shared functionality for both sync and async evaluations managers.

    Contains utility methods that don't depend on HTTP client type.
    """

    _config: BrokleConfig

    def _log(self, message: str, *args: Any) -> None:
        """Log debug messages."""
        if self._config.debug:
            print(f"[Brokle Evaluations] {message}", *args)


class BaseSyncEvaluationsManager(_BaseEvaluationsManagerMixin):
    """
    Sync base class for evaluations manager.

    Uses SyncHTTPClient (httpx.Client) - no event loop involvement.
    All methods are synchronous.
    """

    def __init__(
        self,
        http_client: SyncHTTPClient,
        config: BrokleConfig,
    ):
        """
        Initialize sync evaluations manager.

        Args:
            http_client: Sync HTTP client
            config: Brokle configuration
        """
        self._http = http_client
        self._config = config

    def _score(
        self,
        trace_id: str,
        name: str,
        value: float,
        type: str = "NUMERIC",
        source: str = "code",
        span_id: Optional[str] = None,
        string_value: Optional[str] = None,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Submit a score to the API (sync).

        Args:
            trace_id: Trace ID to score
            name: Score name (e.g., 'accuracy', 'relevance')
            value: Score value (typically 0.0-1.0)
            type: Score type (NUMERIC, CATEGORICAL, BOOLEAN)
            source: Score source (code, llm, human)
            span_id: Optional span ID for span-level scoring
            string_value: String value for CATEGORICAL scores
            reason: Human-readable explanation
            metadata: Additional metadata

        Returns:
            Score submission result

        Raises:
            ScoreError: If the API request fails
        """
        self._log(f"Scoring trace: {trace_id} - {name}={value}")

        payload: Dict[str, Any] = {
            "trace_id": trace_id,
            "name": name,
            "value": value,
            "type": type,
            "source": source,
        }
        if span_id:
            payload["span_id"] = span_id
        if string_value:
            payload["string_value"] = string_value
        if reason:
            payload["reason"] = reason
        if metadata:
            payload["metadata"] = metadata

        try:
            raw_response = self._http.post("/v1/scores", json=payload)
            return unwrap_response(raw_response, resource_type="Score")
        except ValueError as e:
            raise ScoreError(f"Failed to submit score: {e}")
        except Exception as e:
            raise ScoreError(f"Failed to submit score: {e}")

    def _score_batch(
        self,
        scores: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Submit multiple scores to the API (sync).

        Args:
            scores: List of score dictionaries with keys:
                - trace_id: Trace ID
                - name: Score name
                - value: Score value
                - type: Score type (optional)
                - source: Score source (optional)
                - span_id: Span ID (optional)
                - string_value: String value (optional)
                - reason: Reason (optional)
                - metadata: Metadata (optional)

        Returns:
            Batch submission result

        Raises:
            ScoreError: If the API request fails
        """
        self._log(f"Submitting batch of {len(scores)} scores")

        try:
            raw_response = self._http.post("/v1/scores/batch", json={"scores": scores})
            return unwrap_response(raw_response, resource_type="Scores")
        except ValueError as e:
            raise ScoreError(f"Failed to submit scores batch: {e}")
        except Exception as e:
            raise ScoreError(f"Failed to submit scores batch: {e}")

    def _run(
        self,
        trace_id: str,
        evaluator: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run an evaluation on a trace (sync).

        Args:
            trace_id: Trace ID to evaluate
            evaluator: Evaluator name
            **kwargs: Additional evaluator-specific parameters

        Returns:
            Evaluation result

        Note:
            This is a stub implementation. Will be implemented when
            evaluation API is ready.
        """
        self._log(f"Evaluating trace: {trace_id} with {evaluator}")
        raise NotImplementedError(
            "Evaluations API not yet implemented. "
            "This is a stub for future functionality."
        )

    def _shutdown(self) -> None:
        """
        Internal cleanup method (sync).

        Called by parent client during shutdown.
        """
        pass


class BaseAsyncEvaluationsManager(_BaseEvaluationsManagerMixin):
    """
    Async base class for evaluations manager.

    Uses AsyncHTTPClient (httpx.AsyncClient) - requires async context.
    All methods are async.
    """

    def __init__(
        self,
        http_client: AsyncHTTPClient,
        config: BrokleConfig,
    ):
        """
        Initialize async evaluations manager.

        Args:
            http_client: Async HTTP client
            config: Brokle configuration
        """
        self._http = http_client
        self._config = config

    async def _score(
        self,
        trace_id: str,
        name: str,
        value: float,
        type: str = "NUMERIC",
        source: str = "code",
        span_id: Optional[str] = None,
        string_value: Optional[str] = None,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Submit a score to the API (async).

        Args:
            trace_id: Trace ID to score
            name: Score name (e.g., 'accuracy', 'relevance')
            value: Score value (typically 0.0-1.0)
            type: Score type (NUMERIC, CATEGORICAL, BOOLEAN)
            source: Score source (code, llm, human)
            span_id: Optional span ID for span-level scoring
            string_value: String value for CATEGORICAL scores
            reason: Human-readable explanation
            metadata: Additional metadata

        Returns:
            Score submission result

        Raises:
            ScoreError: If the API request fails
        """
        self._log(f"Scoring trace: {trace_id} - {name}={value}")

        payload: Dict[str, Any] = {
            "trace_id": trace_id,
            "name": name,
            "value": value,
            "type": type,
            "source": source,
        }
        if span_id:
            payload["span_id"] = span_id
        if string_value:
            payload["string_value"] = string_value
        if reason:
            payload["reason"] = reason
        if metadata:
            payload["metadata"] = metadata

        try:
            raw_response = await self._http.post("/v1/scores", json=payload)
            return unwrap_response(raw_response, resource_type="Score")
        except ValueError as e:
            raise ScoreError(f"Failed to submit score: {e}")
        except Exception as e:
            raise ScoreError(f"Failed to submit score: {e}")

    async def _score_batch(
        self,
        scores: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Submit multiple scores to the API (async).

        Args:
            scores: List of score dictionaries with keys:
                - trace_id: Trace ID
                - name: Score name
                - value: Score value
                - type: Score type (optional)
                - source: Score source (optional)
                - span_id: Span ID (optional)
                - string_value: String value (optional)
                - reason: Reason (optional)
                - metadata: Metadata (optional)

        Returns:
            Batch submission result

        Raises:
            ScoreError: If the API request fails
        """
        self._log(f"Submitting batch of {len(scores)} scores")

        try:
            raw_response = await self._http.post(
                "/v1/scores/batch", json={"scores": scores}
            )
            return unwrap_response(raw_response, resource_type="Scores")
        except ValueError as e:
            raise ScoreError(f"Failed to submit scores batch: {e}")
        except Exception as e:
            raise ScoreError(f"Failed to submit scores batch: {e}")

    async def _run(
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

        Note:
            This is a stub implementation. Will be implemented when
            evaluation API is ready.
        """
        self._log(f"Evaluating trace: {trace_id} with {evaluator}")
        raise NotImplementedError(
            "Evaluations API not yet implemented. "
            "This is a stub for future functionality."
        )

    async def _shutdown(self) -> None:
        """
        Internal cleanup method (async).

        Called by parent client during shutdown.
        """
        pass
