"""
Relevance Evaluators

Evaluators for assessing response relevance to input/questions.
"""

from .base_evaluator import BaseEvaluator


class Relevance(BaseEvaluator):
    """
    Evaluates whether the output is relevant to the input/question.

    Checks if the response directly addresses what was asked.
    Returns 1.0 for highly relevant, 0.0 for completely irrelevant.

    Example:
        >>> relevance = Relevance(client)
        >>> result = relevance(
        ...     output="Python is a programming language.",
        ...     expected="What is the weather like today?"
        ... )
    """

    DEFAULT_NAME = "relevance"
    SCORE_DESCRIPTION = "Measures how relevant the output is to the input"
    DEFAULT_PROMPT = """You are an expert at evaluating response relevance.

Input/Question:
{{expected}}

Output to evaluate:
{{output}}

Evaluate how relevant the output is to the input/question.

Consider:
1. Does the output directly address the question or request?
2. Is the information provided useful for the query?
3. Does it stay on topic or drift to unrelated subjects?
4. Does it provide what was actually asked for?

Respond with a JSON object:
{
    "score": <float between 0.0 and 1.0>,
    "reason": "<explanation of relevance assessment>"
}

Score guidelines:
- 1.0: Perfectly relevant - directly and completely addresses the input
- 0.7-0.9: Highly relevant with minor tangents
- 0.4-0.6: Partially relevant - addresses some aspects but misses others
- 0.1-0.3: Mostly irrelevant - barely touches on the topic
- 0.0: Completely irrelevant - does not address the input at all"""


class AnswerRelevance(BaseEvaluator):
    """
    Evaluates answer relevance specifically for Q&A tasks.

    More focused than general Relevance - specifically checks if the answer
    properly responds to the question format and expectations.

    Example:
        >>> answer_relevance = AnswerRelevance(client)
        >>> result = answer_relevance(
        ...     output="42",
        ...     expected="What is 6 times 7?"
        ... )
    """

    DEFAULT_NAME = "answer_relevance"
    SCORE_DESCRIPTION = "Measures how well the answer addresses the question"
    DEFAULT_PROMPT = """You are an expert at evaluating Q&A quality.

Question:
{{expected}}

Answer:
{{output}}

Evaluate how well the answer addresses the question.

Consider:
1. Does the answer directly respond to what was asked?
2. Is the answer type appropriate (e.g., a number for "how many", a yes/no for binary questions)?
3. Does it provide the specific information requested?
4. Is the answer complete or does it leave parts of the question unanswered?

Respond with a JSON object:
{
    "score": <float between 0.0 and 1.0>,
    "reason": "<explanation of how well the answer addresses the question>"
}

Score guidelines:
- 1.0: Perfect answer that fully addresses the question
- 0.7-0.9: Good answer with minor omissions
- 0.4-0.6: Partial answer - addresses some aspects
- 0.1-0.3: Weak answer - barely addresses the question
- 0.0: Does not answer the question at all"""


__all__ = [
    "Relevance",
    "AnswerRelevance",
]
