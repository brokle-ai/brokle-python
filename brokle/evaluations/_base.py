"""
Base Evaluations Managers

Provides separate sync and async base implementations for evaluation operations.

Architecture:
- BaseSyncEvaluationsManager: Uses SyncHTTPClient (no event loop)
- BaseAsyncEvaluationsManager: Uses AsyncHTTPClient (async/await)

This design eliminates event loop lifecycle issues.
"""

from typing import Any, Dict

from .._http import AsyncHTTPClient, SyncHTTPClient
from ..config import BrokleConfig


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

    def _score(
        self,
        span_id: str,
        name: str,
        value: float,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Submit a quality score for a span (sync).

        Args:
            span_id: Span ID to score
            name: Score name (e.g., 'accuracy', 'relevance')
            value: Score value
            **kwargs: Additional metadata

        Returns:
            Score submission result

        Note:
            This is a stub implementation. Will be implemented when
            scoring API is ready.
        """
        self._log(f"Scoring span: {span_id} - {name}={value}")
        raise NotImplementedError(
            "Scoring API not yet implemented. "
            "This is a stub for future functionality."
        )

    def _shutdown(self) -> None:
        """
        Internal cleanup method (sync).

        Called by parent client during shutdown.
        """
        pass  # Nothing to clean up for now


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

    async def _score(
        self,
        span_id: str,
        name: str,
        value: float,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Submit a quality score for a span (async).

        Args:
            span_id: Span ID to score
            name: Score name (e.g., 'accuracy', 'relevance')
            value: Score value
            **kwargs: Additional metadata

        Returns:
            Score submission result

        Note:
            This is a stub implementation. Will be implemented when
            scoring API is ready.
        """
        self._log(f"Scoring span: {span_id} - {name}={value}")
        raise NotImplementedError(
            "Scoring API not yet implemented. "
            "This is a stub for future functionality."
        )

    async def _shutdown(self) -> None:
        """
        Internal cleanup method (async).

        Called by parent client during shutdown.
        """
        pass  # Nothing to clean up for now
