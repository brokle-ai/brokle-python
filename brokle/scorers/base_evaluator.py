"""
Base Evaluator Class

Provides the foundation for all pre-built LLM evaluators.
"""

from typing import Any, Dict, Optional

from .llm_scorer import LLMScorer
from ..scores.types import ScoreResult


class BaseEvaluator:
    """
    Base class for pre-built LLM evaluators.

    Provides common initialization and configuration for all evaluators.
    Subclasses should define:
    - DEFAULT_NAME: Default name for the score
    - DEFAULT_PROMPT: The evaluation prompt template
    - SCORE_DESCRIPTION: Human-readable description for documentation
    """

    DEFAULT_NAME: str = "evaluator"
    DEFAULT_PROMPT: str = ""
    SCORE_DESCRIPTION: str = ""

    def __init__(
        self,
        client: Any,
        name: Optional[str] = None,
        model: str = "gpt-4o",
        credential_id: Optional[str] = None,
        temperature: float = 0.0,
        use_cot: bool = True,
    ):
        """
        Initialize a pre-built evaluator.

        Args:
            client: Brokle client instance (sync or async)
            name: Custom name for the score (default: class-specific)
            model: LLM model to use (default: gpt-4o)
            credential_id: Optional credential ID for the model provider
            temperature: Sampling temperature (default: 0.0 for deterministic)
            use_cot: Use chain-of-thought reasoning (default: True for explainability)
        """
        self._scorer = LLMScorer(
            client=client,
            name=name or self.DEFAULT_NAME,
            prompt=self.DEFAULT_PROMPT,
            model=model,
            credential_id=credential_id,
            temperature=temperature,
            use_cot=use_cot,
        )
        self.name = name or self.DEFAULT_NAME

    def __call__(
        self,
        output: Any,
        expected: Any = None,
        **kwargs: Any,
    ) -> ScoreResult:
        """
        Evaluate the output.

        Args:
            output: The generated output to evaluate
            expected: Expected output or reference (optional, depends on evaluator)
            **kwargs: Additional context passed to the LLM

        Returns:
            ScoreResult with value (0.0-1.0), reason, and metadata
        """
        return self._scorer(output=output, expected=expected, **kwargs)

    async def __call_async__(
        self,
        output: Any,
        expected: Any = None,
        **kwargs: Any,
    ) -> ScoreResult:
        """Async version of evaluation."""
        return await self._scorer.__call_async__(output=output, expected=expected, **kwargs)


def create_evaluator(
    client: Any,
    evaluator_type: str,
    **kwargs: Any,
) -> BaseEvaluator:
    """
    Factory function to create evaluators by name.

    Args:
        client: Brokle client instance
        evaluator_type: Name of evaluator (e.g., "factuality", "relevance")
        **kwargs: Additional arguments passed to evaluator

    Returns:
        Configured evaluator instance

    Example:
        >>> evaluator = create_evaluator(client, "factuality", model="gpt-4o")
    """
    # Import here to avoid circular imports
    from .factuality import Factuality, Hallucination
    from .relevance import Relevance, AnswerRelevance
    from .quality import Coherence, Fluency, Completeness
    from .safety import Safety, Toxicity
    from .rag import ContextPrecision, ContextRecall, Faithfulness

    evaluators: Dict[str, type] = {
        "factuality": Factuality,
        "hallucination": Hallucination,
        "relevance": Relevance,
        "answer_relevance": AnswerRelevance,
        "coherence": Coherence,
        "fluency": Fluency,
        "completeness": Completeness,
        "safety": Safety,
        "toxicity": Toxicity,
        "context_precision": ContextPrecision,
        "context_recall": ContextRecall,
        "faithfulness": Faithfulness,
    }

    evaluator_type_lower = evaluator_type.lower().replace("-", "_")
    if evaluator_type_lower not in evaluators:
        available = ", ".join(sorted(evaluators.keys()))
        raise ValueError(
            f"Unknown evaluator type: {evaluator_type}. "
            f"Available evaluators: {available}"
        )

    return evaluators[evaluator_type_lower](client, **kwargs)


def list_evaluators() -> Dict[str, str]:
    """
    List all available pre-built evaluators with descriptions.

    Returns:
        Dict mapping evaluator names to their descriptions

    Example:
        >>> for name, desc in list_evaluators().items():
        ...     print(f"{name}: {desc}")
    """
    # Import here to avoid circular imports
    from .factuality import Factuality, Hallucination
    from .relevance import Relevance, AnswerRelevance
    from .quality import Coherence, Fluency, Completeness
    from .safety import Safety, Toxicity
    from .rag import ContextPrecision, ContextRecall, Faithfulness

    return {
        "factuality": Factuality.SCORE_DESCRIPTION,
        "hallucination": Hallucination.SCORE_DESCRIPTION,
        "relevance": Relevance.SCORE_DESCRIPTION,
        "answer_relevance": AnswerRelevance.SCORE_DESCRIPTION,
        "coherence": Coherence.SCORE_DESCRIPTION,
        "fluency": Fluency.SCORE_DESCRIPTION,
        "completeness": Completeness.SCORE_DESCRIPTION,
        "safety": Safety.SCORE_DESCRIPTION,
        "toxicity": Toxicity.SCORE_DESCRIPTION,
        "context_precision": ContextPrecision.SCORE_DESCRIPTION,
        "context_recall": ContextRecall.SCORE_DESCRIPTION,
        "faithfulness": Faithfulness.SCORE_DESCRIPTION,
    }


__all__ = [
    "BaseEvaluator",
    "create_evaluator",
    "list_evaluators",
]
