"""
Quality Evaluators

Evaluators for assessing text quality: coherence, fluency, and completeness.
"""

from .base_evaluator import BaseEvaluator


class Coherence(BaseEvaluator):
    """
    Evaluates the logical coherence of the output.

    Checks if the response flows logically and makes sense.
    Returns 1.0 for highly coherent, 0.0 for incoherent.

    Example:
        >>> coherence = Coherence(client)
        >>> result = coherence(
        ...     output="The sky is blue because of Rayleigh scattering...",
        ...     expected="Why is the sky blue?"
        ... )
    """

    DEFAULT_NAME = "coherence"
    SCORE_DESCRIPTION = "Measures logical flow and clarity of the output"
    DEFAULT_PROMPT = """You are an expert at evaluating text coherence and clarity.

Input/Context:
{{expected}}

Output to evaluate:
{{output}}

Evaluate the logical coherence of the output.

Consider:
1. Does the text flow logically from one point to the next?
2. Are ideas connected with appropriate transitions?
3. Is there a clear structure and organization?
4. Are there any contradictions within the text?
5. Is the reasoning sound and easy to follow?

Respond with a JSON object:
{
    "score": <float between 0.0 and 1.0>,
    "reason": "<explanation of coherence assessment>"
}

Score guidelines:
- 1.0: Perfectly coherent - clear, logical, well-organized
- 0.7-0.9: Highly coherent with minor issues
- 0.4-0.6: Moderately coherent - some logical gaps or unclear sections
- 0.1-0.3: Poorly coherent - difficult to follow, disorganized
- 0.0: Incoherent - no logical flow, contradictory or nonsensical"""


class Fluency(BaseEvaluator):
    """
    Evaluates the linguistic fluency of the output.

    Checks grammar, style, and readability.
    Returns 1.0 for perfectly fluent, 0.0 for unreadable.

    Example:
        >>> fluency = Fluency(client)
        >>> result = fluency(output="This is a well-written sentence.")
    """

    DEFAULT_NAME = "fluency"
    SCORE_DESCRIPTION = "Measures grammatical correctness and readability"
    DEFAULT_PROMPT = """You are an expert linguist evaluating text fluency.

Output to evaluate:
{{output}}

Evaluate the linguistic fluency of the output.

Consider:
1. Is the grammar correct?
2. Is the writing style appropriate and consistent?
3. Is the text readable and easy to understand?
4. Are word choices appropriate and natural?
5. Is there proper punctuation and sentence structure?

Respond with a JSON object:
{
    "score": <float between 0.0 and 1.0>,
    "reason": "<explanation of fluency assessment with specific examples>"
}

Score guidelines:
- 1.0: Perfectly fluent - reads naturally, no errors
- 0.7-0.9: Highly fluent with minor issues
- 0.4-0.6: Moderately fluent - some grammatical or stylistic issues
- 0.1-0.3: Poor fluency - multiple errors, hard to read
- 0.0: Not fluent - unreadable or severely broken"""


class Completeness(BaseEvaluator):
    """
    Evaluates whether the output completely addresses the request.

    Checks if all aspects of the question/task were addressed.
    Returns 1.0 for complete, 0.0 for severely incomplete.

    Example:
        >>> completeness = Completeness(client)
        >>> result = completeness(
        ...     output="The capital is Paris.",
        ...     expected="What is the capital of France and its population?"
        ... )
    """

    DEFAULT_NAME = "completeness"
    SCORE_DESCRIPTION = "Measures whether all aspects of the request are addressed"
    DEFAULT_PROMPT = """You are an expert at evaluating response completeness.

Input/Request:
{{expected}}

Output to evaluate:
{{output}}

Evaluate whether the output completely addresses all aspects of the request.

Consider:
1. Were all parts of the question answered?
2. Are there any obvious omissions?
3. Is the level of detail appropriate?
4. Were all sub-questions or requirements addressed?

Respond with a JSON object:
{
    "score": <float between 0.0 and 1.0>,
    "reason": "<explanation identifying any missing elements>"
}

Score guidelines:
- 1.0: Fully complete - addresses all aspects thoroughly
- 0.7-0.9: Mostly complete with minor omissions
- 0.4-0.6: Partially complete - addresses main points but misses some
- 0.1-0.3: Incomplete - major aspects not addressed
- 0.0: Severely incomplete - barely addresses the request"""


__all__ = [
    "Coherence",
    "Fluency",
    "Completeness",
]
