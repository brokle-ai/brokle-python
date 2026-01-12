"""
RAG Evaluators

Evaluators specifically designed for Retrieval-Augmented Generation (RAG) applications.
Includes context precision, context recall, and faithfulness metrics.
"""

from .base_evaluator import BaseEvaluator


class ContextPrecision(BaseEvaluator):
    """
    Evaluates precision of retrieved context in RAG applications.

    Checks if the retrieved context contains relevant information.
    Returns 1.0 for highly precise context, 0.0 for irrelevant context.

    Example:
        >>> context_precision = ContextPrecision(client)
        >>> result = context_precision(
        ...     output="Retrieved chunk: 'Paris is the capital of France...'",
        ...     expected="What is the capital of France?"
        ... )
    """

    DEFAULT_NAME = "context_precision"
    SCORE_DESCRIPTION = "Measures precision of retrieved context for RAG"
    DEFAULT_PROMPT = """You are an expert at evaluating RAG (Retrieval-Augmented Generation) systems.

Question/Query:
{{expected}}

Retrieved Context:
{{output}}

Evaluate the precision of the retrieved context for answering the question.

Consider:
1. Does the context contain information relevant to the question?
2. How much of the context is useful vs. noise?
3. Is the key information needed to answer present?
4. Would this context help generate a correct answer?

Respond with a JSON object:
{
    "score": <float between 0.0 and 1.0>,
    "reason": "<explanation of context precision>"
}

Score guidelines:
- 1.0: Perfect precision - all context is highly relevant
- 0.7-0.9: High precision - mostly relevant with some noise
- 0.4-0.6: Moderate precision - mix of relevant and irrelevant
- 0.1-0.3: Low precision - mostly irrelevant
- 0.0: No precision - context is completely irrelevant"""


class ContextRecall(BaseEvaluator):
    """
    Evaluates recall of context retrieval in RAG applications.

    Checks if the retrieved context covers all necessary information.
    Returns 1.0 for complete coverage, 0.0 for missing critical info.

    Example:
        >>> context_recall = ContextRecall(client)
        >>> result = context_recall(
        ...     output="Retrieved: 'Paris is the capital.'",
        ...     expected="Full answer should mention: Paris, capital, France, population"
        ... )
    """

    DEFAULT_NAME = "context_recall"
    SCORE_DESCRIPTION = "Measures recall/coverage of retrieved context for RAG"
    DEFAULT_PROMPT = """You are an expert at evaluating RAG (Retrieval-Augmented Generation) systems.

Expected information needed:
{{expected}}

Retrieved Context:
{{output}}

Evaluate the recall of the retrieved context - does it contain all necessary information?

Consider:
1. Does the context cover all key points needed?
2. Is any critical information missing?
3. Would the context enable a complete answer?
4. Are there gaps that would require additional retrieval?

Respond with a JSON object:
{
    "score": <float between 0.0 and 1.0>,
    "reason": "<explanation of what information is covered or missing>"
}

Score guidelines:
- 1.0: Complete recall - all necessary information present
- 0.7-0.9: High recall - most key information present
- 0.4-0.6: Moderate recall - some important info missing
- 0.1-0.3: Low recall - significant gaps
- 0.0: No recall - missing all critical information"""


class Faithfulness(BaseEvaluator):
    """
    Evaluates faithfulness of generated answer to retrieved context.

    For RAG: checks if the answer only uses information from context.
    Returns 1.0 for fully faithful, 0.0 for unfaithful.

    Example:
        >>> faithfulness = Faithfulness(client)
        >>> result = faithfulness(
        ...     output="Answer: Paris, population 2.1 million",
        ...     expected="Context: Paris is the capital of France."
        ... )
    """

    DEFAULT_NAME = "faithfulness"
    SCORE_DESCRIPTION = "Measures faithfulness of answer to source context"
    DEFAULT_PROMPT = """You are an expert at evaluating RAG (Retrieval-Augmented Generation) systems.

Source Context:
{{expected}}

Generated Answer:
{{output}}

Evaluate how faithful the generated answer is to the source context.

Consider:
1. Does the answer only use information from the context?
2. Does it add information not present in the context?
3. Does it contradict anything in the context?
4. Are any claims in the answer unsupported by the context?

Respond with a JSON object:
{
    "score": <float between 0.0 and 1.0>,
    "reason": "<explanation of faithfulness, citing specific discrepancies if any>"
}

Score guidelines:
- 1.0: Fully faithful - all claims supported by context
- 0.7-0.9: Highly faithful with minor additions
- 0.4-0.6: Partially faithful - some unsupported claims
- 0.1-0.3: Mostly unfaithful - significant unsupported information
- 0.0: Not faithful - contradicts or ignores context"""


__all__ = [
    "ContextPrecision",
    "ContextRecall",
    "Faithfulness",
]
