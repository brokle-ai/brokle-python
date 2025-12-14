"""
Base Evaluations Manager

Provides the async-first internal implementation for evaluation operations.
Both AsyncEvaluationsManager and EvaluationsManager inherit from this class.
"""

from typing import Any, Dict, Optional

from .._http import AsyncHTTPClient
from ..config import BrokleConfig


class BaseEvaluationsManager:
    """
    Base class for evaluations manager with async internal implementation.

    This class provides all the internal async methods that both
    AsyncEvaluationsManager and EvaluationsManager use.
    """

    def __init__(
        self,
        http_client: AsyncHTTPClient,
        config: BrokleConfig,
    ):
        """
        Initialize the evaluations manager.

        Args:
            http_client: Shared async HTTP client
            config: Brokle configuration
        """
        self._http = http_client
        self._config = config

    def _log(self, message: str, *args: Any) -> None:
        """Log debug messages."""
        if self._config.debug:
            print(f"[Brokle Evaluations] {message}", *args)

    async def _run(
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
        Submit a quality score for a span.

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
        Internal cleanup method.

        Called by parent client during shutdown.
        """
        pass  # Nothing to clean up for now
