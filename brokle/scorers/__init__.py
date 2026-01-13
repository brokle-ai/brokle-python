"""
Brokle Scorers Module

Provides built-in scorers, LLM-as-Judge scorers, pre-built evaluators, and decorators
for creating custom evaluation functions.

Built-in Scorers (Heuristic):
- ExactMatch: Exact string comparison
- Contains: Substring matching
- RegexMatch: Regex pattern matching
- JSONValid: JSON validity check
- LengthCheck: String length validation

LLM-as-Judge Scorers:
- LLMScorer: Use LLM models to evaluate outputs with project credentials

Pre-built Evaluators (LLM-as-Judge with standardized prompts):

Factuality:
- Factuality: Evaluates factual accuracy
- Hallucination: Detects hallucinations

Relevance:
- Relevance: Evaluates response relevance
- AnswerRelevance: Evaluates Q&A answer relevance

Quality:
- Coherence: Evaluates logical coherence
- Fluency: Evaluates linguistic fluency
- Completeness: Evaluates response completeness

Safety:
- Safety: Evaluates content safety
- Toxicity: Detects toxic language

RAG:
- ContextPrecision: RAG context precision
- ContextRecall: RAG context recall
- Faithfulness: RAG answer faithfulness

Decorators:
- @scorer: Create custom scorers from functions
- @multi_scorer: Create scorers that return multiple scores

Usage:
    >>> from brokle import Brokle
    >>> from brokle.scorers import ExactMatch, Contains, LLMScorer, scorer, ScoreResult
    >>> from brokle.scorers import Factuality, Relevance  # Pre-built evaluators
    >>>
    >>> client = Brokle(api_key="bk_...")
    >>>
    >>> # Built-in scorer
    >>> exact = ExactMatch(name="answer_match")
    >>> client.scores.submit(
    ...     trace_id="abc123",
    ...     scorer=exact,
    ...     output="Paris",
    ...     expected="Paris",
    ... )
    >>>
    >>> # LLM-as-Judge scorer
    >>> relevance = LLMScorer(
    ...     client=client,
    ...     name="relevance",
    ...     prompt="Rate relevance 0-10: {{output}}",
    ...     model="gpt-4o",
    ... )
    >>>
    >>> # Pre-built evaluator (recommended for common use cases)
    >>> factuality = Factuality(client, model="gpt-4o")
    >>> result = factuality(
    ...     output="Paris is the capital of France.",
    ...     expected="What is the capital of France?"
    ... )
    >>> print(f"Score: {result.value}, Reason: {result.reason}")
    >>>
    >>> # Custom scorer
    >>> @scorer
    ... def similarity(output, expected=None, **kwargs):
    ...     return 0.85  # Auto-wrapped as ScoreResult
    >>>
    >>> client.scores.submit(
    ...     trace_id="abc123",
    ...     scorer=similarity,
    ...     output="result",
    ...     expected="expected",
    ... )
"""

# Re-export ScoreResult for convenience in custom scorers
from ..scores.types import ScoreResult, ScoreType
from .base import Contains, ExactMatch, JSONValid, LengthCheck, RegexMatch
from .decorator import multi_scorer, scorer
from .llm_scorer import LLMScorer

# Base evaluator and factory functions
from .base_evaluator import BaseEvaluator, create_evaluator, list_evaluators

# Pre-built LLM evaluators - Category: Factuality
from .factuality import Factuality, Hallucination

# Pre-built LLM evaluators - Category: Relevance
from .relevance import AnswerRelevance, Relevance

# Pre-built LLM evaluators - Category: Quality
from .quality import Coherence, Completeness, Fluency

# Pre-built LLM evaluators - Category: Safety
from .safety import Safety, Toxicity

# Pre-built LLM evaluators - Category: RAG
from .rag import ContextPrecision, ContextRecall, Faithfulness

__all__ = [
    # Built-in scorers (heuristic)
    "ExactMatch",
    "Contains",
    "RegexMatch",
    "JSONValid",
    "LengthCheck",
    # LLM-as-Judge scorers
    "LLMScorer",
    # Base evaluator and factory
    "BaseEvaluator",
    "create_evaluator",
    "list_evaluators",
    # Pre-built evaluators - Factuality
    "Factuality",
    "Hallucination",
    # Pre-built evaluators - Relevance
    "Relevance",
    "AnswerRelevance",
    # Pre-built evaluators - Quality
    "Coherence",
    "Fluency",
    "Completeness",
    # Pre-built evaluators - Safety
    "Safety",
    "Toxicity",
    # Pre-built evaluators - RAG
    "ContextPrecision",
    "ContextRecall",
    "Faithfulness",
    # Decorators
    "scorer",
    "multi_scorer",
    # Types (for custom scorers)
    "ScoreResult",
    "ScoreType",
]
