"""
Top-Level Evaluate Function

Provides a simplified interface for running evaluations, following the pattern
used by Braintrust, LangSmith, and other competitors.

This is a convenience wrapper that:
1. Gets or creates a Brokle client singleton
2. Creates a dataset if needed (from list of dicts)
3. Runs the experiment with the provided evaluators
4. Returns comprehensive results

Usage:
    >>> from brokle import evaluate
    >>> from brokle.scorers import Factuality, Relevance
    >>>
    >>> # With a dataset object
    >>> results = evaluate(
    ...     task=lambda item: call_llm(item["question"]),
    ...     data=dataset,
    ...     evaluators=[Factuality(client), Relevance(client)],
    ...     experiment_name="gpt-4o-v2",
    ... )
    >>>
    >>> # With raw data
    >>> results = evaluate(
    ...     task=lambda item: call_llm(item["question"]),
    ...     data=[
    ...         {"input": {"question": "What is 2+2?"}, "expected": {"answer": "4"}},
    ...         {"input": {"question": "Capital of France?"}, "expected": {"answer": "Paris"}},
    ...     ],
    ...     evaluators=[Factuality(client)],
    ...     experiment_name="qa-test",
    ... )
    >>>
    >>> # With key mapping
    >>> results = evaluate(
    ...     task=lambda item: call_llm(item["prompt"]),
    ...     data=dataset,
    ...     evaluators=[Relevance(client)],
    ...     experiment_name="mapped-test",
    ...     scoring_key_mapping={"input": "prompt", "output": "response"},
    ... )
"""

import asyncio
import inspect
import uuid
from typing import Any, Callable, Dict, List, Optional, Union

from .datasets.dataset import AsyncDataset, Dataset, DatasetItemInput
from .experiments.types import EvaluationResults, ProgressCallback, TaskFunction
from .scores.types import ScoreResult, ScorerProtocol, ScoreType


class KeyMappingScorer:
    """
    Wrapper scorer that remaps keys before passing to the underlying scorer.

    This allows evaluators to work with different field names in the data.
    """

    def __init__(
        self,
        scorer: ScorerProtocol,
        mapping: Dict[str, str],
    ):
        """
        Initialize the key mapping wrapper.

        Args:
            scorer: The underlying scorer to wrap
            mapping: Dict mapping scorer keys to data keys
                     e.g., {"output": "response"} means the scorer's "output"
                     will receive the data's "response" field
        """
        self._scorer = scorer
        self._mapping = mapping
        # Preserve the name from the underlying scorer
        self.name = getattr(scorer, "name", scorer.__class__.__name__)

    def __call__(
        self,
        output: Any,
        expected: Any = None,
        input: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ScoreResult:
        """
        Call the underlying scorer with remapped keys.

        The mapping is applied to kwargs and input dict.
        """
        # Remap the main arguments if specified
        actual_output = output
        actual_expected = expected
        actual_input = input

        if "output" in self._mapping and input:
            # If output should come from a different key in input
            key = self._mapping["output"]
            if key in input:
                actual_output = input[key]

        if "expected" in self._mapping and input:
            key = self._mapping["expected"]
            if key in input:
                actual_expected = input[key]

        if "input" in self._mapping and input:
            key = self._mapping["input"]
            if key in input:
                actual_input = {key: input[key]}

        return self._scorer(
            output=actual_output,
            expected=actual_expected,
            input=actual_input,
            **kwargs,
        )


def _wrap_scorers_with_mapping(
    scorers: List[ScorerProtocol],
    mapping: Optional[Dict[str, str]],
) -> List[ScorerProtocol]:
    """Wrap scorers with key mapping if a mapping is provided."""
    if not mapping:
        return scorers
    return [KeyMappingScorer(scorer, mapping) for scorer in scorers]


def _create_temp_dataset_items(
    data: List[Dict[str, Any]],
) -> List[DatasetItemInput]:
    """
    Convert raw data dicts to DatasetItemInput format.

    Expected format for each item:
    - {"input": {...}, "expected": {...}, "metadata": {...}}
    - or just {"input": {...}}
    """
    items = []
    for item in data:
        if "input" in item:
            # Standard format
            items.append(
                DatasetItemInput(
                    input=item["input"],
                    expected=item.get("expected"),
                    metadata=item.get("metadata"),
                )
            )
        else:
            # Treat entire dict as input
            items.append(DatasetItemInput(input=item))
    return items


def evaluate(
    task: TaskFunction,
    data: Union[Dataset, str, List[Dict[str, Any]]],
    evaluators: List[ScorerProtocol],
    experiment_name: Optional[str] = None,
    *,
    max_concurrency: int = 10,
    trial_count: int = 1,
    metadata: Optional[Dict[str, Any]] = None,
    scoring_key_mapping: Optional[Dict[str, str]] = None,
    on_progress: Optional[ProgressCallback] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> EvaluationResults:
    """
    Run an evaluation experiment with a simple, high-level interface.

    This function provides a convenient way to run evaluations, similar to
    Braintrust's Eval() and LangSmith's evaluate() functions.

    Args:
        task: Function that takes an input dict and returns an output.
              The input dict comes from dataset items.
              Example: lambda item: call_llm(item["question"])

        data: The data to evaluate. Can be:
              - A Dataset object
              - A dataset ID string
              - A list of dicts with format:
                [{"input": {...}, "expected": {...}}, ...]

        evaluators: List of scorers/evaluators to run on each item.
                   Can be built-in scorers (ExactMatch, Contains) or
                   pre-built evaluators (Factuality, Relevance).

        experiment_name: Name for the experiment (auto-generated if not provided)

        max_concurrency: Maximum parallel task executions (default: 10)

        trial_count: Number of times to run each item (default: 1)

        metadata: Optional experiment metadata

        scoring_key_mapping: Optional dict to remap keys for scorers.
                            Example: {"output": "response", "input": "prompt"}
                            This maps the scorer's expected keys to the actual
                            field names in your data.

        on_progress: Optional callback: (completed, total) -> None

        api_key: Optional API key (uses BROKLE_API_KEY if not provided)

        base_url: Optional base URL (uses BROKLE_BASE_URL if not provided)

    Returns:
        EvaluationResults with:
        - experiment_id: ID of the created experiment
        - experiment_name: Name of the experiment
        - summary: Dict of scorer name -> SummaryStats (mean, std_dev, etc.)
        - items: List of EvaluationItem results
        - url: Dashboard URL to view the experiment

    Example:
        >>> from brokle import evaluate
        >>> from brokle.scorers import Factuality, Relevance
        >>>
        >>> # Define your task
        >>> def my_task(item):
        ...     return openai.chat.completions.create(
        ...         model="gpt-4o",
        ...         messages=[{"role": "user", "content": item["question"]}]
        ...     ).choices[0].message.content
        >>>
        >>> # Run evaluation
        >>> results = evaluate(
        ...     task=my_task,
        ...     data=dataset,
        ...     evaluators=[Factuality(client), Relevance(client)],
        ...     experiment_name="gpt-4o-qa-test",
        ...     max_concurrency=5,
        ... )
        >>>
        >>> # View results
        >>> print(f"Experiment: {results.experiment_name}")
        >>> for name, stats in results.summary.items():
        ...     print(f"  {name}: mean={stats['mean']:.3f}")
        >>> print(f"View at: {results.url}")

    Note:
        This function uses get_client() internally to get or create a Brokle
        client singleton. If you need more control, use client.experiments.run()
        directly.
    """
    # Import here to avoid circular imports
    from .client import get_client

    # Get or create client
    client = get_client(api_key=api_key, base_url=base_url)

    # Generate experiment name if not provided
    if experiment_name is None:
        experiment_name = f"eval-{uuid.uuid4().hex[:8]}"

    # Wrap scorers with key mapping if needed
    wrapped_scorers = _wrap_scorers_with_mapping(evaluators, scoring_key_mapping)

    # Handle different data types
    if isinstance(data, list):
        # Raw data - create a temporary dataset
        dataset_name = f"temp-{uuid.uuid4().hex[:8]}"
        dataset = client.datasets.create(
            name=dataset_name,
            description=f"Temporary dataset for {experiment_name}",
            metadata={"temporary": True, "experiment": experiment_name},
        )

        # Insert items
        items = _create_temp_dataset_items(data)
        dataset.insert(items)

        resolved_data: Union[Dataset, str] = dataset
    else:
        resolved_data = data

    # Run the experiment
    return client.experiments.run(
        name=experiment_name,
        dataset=resolved_data,
        task=task,
        scorers=wrapped_scorers,
        max_concurrency=max_concurrency,
        trial_count=trial_count,
        metadata=metadata,
        on_progress=on_progress,
    )


async def async_evaluate(
    task: Callable[[Dict[str, Any]], Any],
    data: Union[AsyncDataset, str, List[Dict[str, Any]]],
    evaluators: List[ScorerProtocol],
    experiment_name: Optional[str] = None,
    *,
    max_concurrency: int = 10,
    trial_count: int = 1,
    metadata: Optional[Dict[str, Any]] = None,
    scoring_key_mapping: Optional[Dict[str, str]] = None,
    on_progress: Optional[ProgressCallback] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> EvaluationResults:
    """
    Async version of evaluate().

    See evaluate() for full documentation. This version:
    - Accepts async task functions
    - Uses async dataset operations
    - Must be awaited

    Example:
        >>> async def my_async_task(item):
        ...     response = await async_llm_call(item["question"])
        ...     return response
        >>>
        >>> results = await async_evaluate(
        ...     task=my_async_task,
        ...     data=dataset,
        ...     evaluators=[Factuality(client)],
        ...     experiment_name="async-test",
        ... )
    """
    # Import here to avoid circular imports
    from .client import get_async_client

    # Get or create async client
    client = await get_async_client(api_key=api_key, base_url=base_url)

    # Generate experiment name if not provided
    if experiment_name is None:
        experiment_name = f"eval-{uuid.uuid4().hex[:8]}"

    # Wrap scorers with key mapping if needed
    wrapped_scorers = _wrap_scorers_with_mapping(evaluators, scoring_key_mapping)

    # Handle different data types
    if isinstance(data, list):
        # Raw data - create a temporary dataset
        dataset_name = f"temp-{uuid.uuid4().hex[:8]}"
        dataset = await client.datasets.create(
            name=dataset_name,
            description=f"Temporary dataset for {experiment_name}",
            metadata={"temporary": True, "experiment": experiment_name},
        )

        # Insert items
        items = _create_temp_dataset_items(data)
        await dataset.insert(items)

        resolved_data: Union[AsyncDataset, str] = dataset
    else:
        resolved_data = data

    # Run the experiment
    return await client.experiments.run(
        name=experiment_name,
        dataset=resolved_data,
        task=task,
        scorers=wrapped_scorers,
        max_concurrency=max_concurrency,
        trial_count=trial_count,
        metadata=metadata,
        on_progress=on_progress,
    )


__all__ = [
    "evaluate",
    "async_evaluate",
    "KeyMappingScorer",
]
