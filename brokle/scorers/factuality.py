"""
Factuality Evaluators

Evaluators for assessing factual accuracy and detecting hallucinations.
"""

from .base_evaluator import BaseEvaluator


class Factuality(BaseEvaluator):
    """
    Evaluates whether the output is factually correct.

    Checks if the generated response contains accurate, verifiable information.
    Returns 1.0 for fully factual, 0.0 for contains false information.

    Example:
        >>> factuality = Factuality(client)
        >>> result = factuality(
        ...     output="The Eiffel Tower is 330 meters tall and was built in 1889.",
        ...     expected="Provide facts about the Eiffel Tower"
        ... )
    """

    DEFAULT_NAME = "factuality"
    SCORE_DESCRIPTION = "Measures factual accuracy of the output"
    DEFAULT_PROMPT = """You are an expert fact-checker. Evaluate whether the following output is factually correct.

Input/Question: {{expected}}

Output to evaluate:
{{output}}

Evaluate the factual accuracy of the output. Consider:
1. Are all stated facts verifiable and correct?
2. Are there any factual errors or inaccuracies?
3. Are claims properly supported or are they speculation presented as fact?

Respond with a JSON object:
{
    "score": <float between 0.0 and 1.0>,
    "reason": "<brief explanation of the factuality assessment>"
}

Score guidelines:
- 1.0: All facts are accurate and verifiable
- 0.7-0.9: Mostly factual with minor inaccuracies
- 0.4-0.6: Mix of factual and inaccurate information
- 0.1-0.3: Mostly inaccurate or misleading
- 0.0: Completely false or fabricated"""


class Hallucination(BaseEvaluator):
    """
    Detects hallucinations in the output.

    Checks if the output contains information not supported by the input/context.
    Returns 1.0 for no hallucinations, 0.0 for severe hallucinations.

    This is particularly important for RAG applications where the model
    should only use information from the provided context.

    Example:
        >>> hallucination = Hallucination(client)
        >>> result = hallucination(
        ...     output="According to the document, sales increased 50%.",
        ...     expected="The document states sales increased 25% year over year."
        ... )
    """

    DEFAULT_NAME = "hallucination"
    SCORE_DESCRIPTION = "Detects unsupported or fabricated information"
    DEFAULT_PROMPT = """You are an expert at detecting hallucinations in AI-generated text.

Reference/Context (ground truth):
{{expected}}

Output to evaluate:
{{output}}

Analyze whether the output contains hallucinations - information that is not supported by or contradicts the reference context.

Consider:
1. Does the output claim things not present in the reference?
2. Does it contradict information in the reference?
3. Does it make up specific details (names, numbers, dates) not in the reference?
4. Does it present speculation as fact?

Respond with a JSON object:
{
    "score": <float between 0.0 and 1.0>,
    "reason": "<specific examples of hallucinations found, or confirmation of accuracy>"
}

Score guidelines:
- 1.0: No hallucinations - all information is grounded in the reference
- 0.7-0.9: Minor hallucinations or slight exaggerations
- 0.4-0.6: Moderate hallucinations - some claims unsupported
- 0.1-0.3: Significant hallucinations - many fabricated details
- 0.0: Severe hallucinations - contradicts reference or completely fabricated"""


__all__ = [
    "Factuality",
    "Hallucination",
]
