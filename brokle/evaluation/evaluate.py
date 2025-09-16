"""
Core evaluation functions for Brokle SDK.

LangSmith-inspired evaluate() and aevaluate() functions with Brokle enhancements.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable

from .base import BaseEvaluator, EvaluationResult, EvaluationConfig
from .._client import get_client

logger = logging.getLogger(__name__)


def evaluate(
    target_function: Callable,
    data: List[Dict[str, Any]],
    evaluators: List[BaseEvaluator],
    config: Optional[EvaluationConfig] = None,
    experiment_name: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Evaluate a function against a dataset using multiple evaluators.

    This follows LangSmith's evaluate() pattern with Brokle enhancements
    for AI platform metrics and observability.

    Args:
        target_function: Function to evaluate (can be sync or async)
        data: List of evaluation examples with inputs and expected outputs
        evaluators: List of evaluator instances
        config: Evaluation configuration
        experiment_name: Name for the evaluation experiment
        **kwargs: Additional parameters passed to target function

    Returns:
        Dictionary containing evaluation results and aggregated metrics

    Example:
        ```python
        from brokle import evaluate
        from brokle.evaluation import AccuracyEvaluator, RelevanceEvaluator

        def my_llm_function(prompt):
            return llm.generate(prompt)

        data = [
            {"input": "What is 2+2?", "expected": "4"},
            {"input": "What is the capital of France?", "expected": "Paris"}
        ]

        evaluators = [
            AccuracyEvaluator(),
            RelevanceEvaluator()
        ]

        results = evaluate(
            target_function=my_llm_function,
            data=data,
            evaluators=evaluators,
            experiment_name="math_qa_test"
        )
        ```
    """
    config = config or EvaluationConfig()

    # Initialize Brokle client for tracking
    client = get_client() if config.send_to_brokle else None

    # Start evaluation span
    evaluation_span = None
    if client:
        evaluation_span = client.span(
            name=f"evaluation_{experiment_name or 'unnamed'}",
            metadata={
                "experiment_name": experiment_name,
                "evaluator_count": len(evaluators),
                "data_count": len(data),
                "config": config.to_dict()
            },
            tags=["evaluation", "batch"] + config.tags
        )

    try:
        with evaluation_span or _nullcontext():
            # Check if target function is async
            if asyncio.iscoroutinefunction(target_function):
                return asyncio.run(
                    _async_evaluate_impl(
                        target_function, data, evaluators, config,
                        experiment_name, client, **kwargs
                    )
                )
            else:
                return _sync_evaluate_impl(
                    target_function, data, evaluators, config,
                    experiment_name, client, **kwargs
                )

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if evaluation_span:
            evaluation_span.set_attribute("error.type", type(e).__name__)
            evaluation_span.set_attribute("error.message", str(e))
        raise


async def aevaluate(
    target_function: Callable,
    data: List[Dict[str, Any]],
    evaluators: List[BaseEvaluator],
    config: Optional[EvaluationConfig] = None,
    experiment_name: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Async version of evaluate() with concurrency control.

    Args:
        target_function: Function to evaluate (can be sync or async)
        data: List of evaluation examples
        evaluators: List of evaluator instances
        config: Evaluation configuration
        experiment_name: Name for the evaluation experiment
        **kwargs: Additional parameters passed to target function

    Returns:
        Dictionary containing evaluation results and aggregated metrics
    """
    config = config or EvaluationConfig()

    # Initialize Brokle client for tracking
    client = get_client() if config.send_to_brokle else None

    # Start evaluation span
    evaluation_span = None
    if client:
        evaluation_span = client.span(
            name=f"async_evaluation_{experiment_name or 'unnamed'}",
            metadata={
                "experiment_name": experiment_name,
                "evaluator_count": len(evaluators),
                "data_count": len(data),
                "config": config.to_dict()
            },
            tags=["evaluation", "async", "batch"] + config.tags
        )

    try:
        with evaluation_span or _nullcontext():
            return await _async_evaluate_impl(
                target_function, data, evaluators, config,
                experiment_name, client, **kwargs
            )

    except Exception as e:
        logger.error(f"Async evaluation failed: {e}")
        if evaluation_span:
            evaluation_span.set_attribute("error.type", type(e).__name__)
            evaluation_span.set_attribute("error.message", str(e))
        raise


def _sync_evaluate_impl(
    target_function: Callable,
    data: List[Dict[str, Any]],
    evaluators: List[BaseEvaluator],
    config: EvaluationConfig,
    experiment_name: Optional[str],
    client: Any,
    **kwargs
) -> Dict[str, Any]:
    """Synchronous evaluation implementation."""

    all_results = []
    predictions = []
    execution_metadata = []

    # Generate predictions
    for i, example in enumerate(data):
        try:
            # Apply sampling if configured
            if config.sample_rate < 1.0:
                import random
                if random.random() > config.sample_rate:
                    continue

            # Apply filtering if configured
            if config.filter_fn and not config.filter_fn(example):
                continue

            # Check max samples limit
            if config.max_samples and len(predictions) >= config.max_samples:
                break

            # Extract input for target function
            input_data = example.get("input", example.get("inputs", example))

            # Execute target function with span tracking
            prediction_span = None
            if client:
                prediction_span = client.span(
                    name=f"prediction_{i}",
                    metadata={"example_index": i, "experiment": experiment_name}
                )

            with prediction_span or _nullcontext():
                prediction = target_function(input_data, **kwargs)
                predictions.append(prediction)

                # Capture execution metadata
                metadata = {
                    "example_index": i,
                    "input": input_data if config.include_inputs else None,
                    "output": prediction if config.include_outputs else None,
                    "expected": example.get("expected", example.get("reference"))
                }
                execution_metadata.append(metadata)

        except Exception as e:
            logger.error(f"Prediction failed for example {i}: {e}")
            if config.fail_fast:
                raise

            # Add failed prediction
            predictions.append(None)
            execution_metadata.append({
                "example_index": i,
                "error": str(e),
                "input": example.get("input") if config.include_inputs else None
            })

    # Run evaluators
    evaluator_results = {}
    for evaluator in evaluators:
        try:
            # Prepare data for evaluator
            eval_predictions = predictions
            eval_references = [
                meta.get("expected") for meta in execution_metadata
            ]
            eval_inputs = [
                meta.get("input") for meta in execution_metadata
            ]

            # Run batch evaluation
            results = evaluator.batch_evaluate(
                predictions=eval_predictions,
                references=eval_references,
                inputs=eval_inputs,
                config=config
            )

            evaluator_results[evaluator.name] = {
                "results": results,
                "stats": evaluator.get_stats()
            }

            all_results.extend(results)

        except Exception as e:
            logger.error(f"Evaluator {evaluator.name} failed: {e}")
            if config.fail_fast:
                raise

    # Calculate aggregate metrics
    aggregate_metrics = _calculate_aggregate_metrics(evaluator_results)

    # Prepare final results
    final_results = {
        "experiment_name": experiment_name,
        "total_examples": len(data),
        "processed_examples": len(predictions),
        "evaluator_results": evaluator_results,
        "aggregate_metrics": aggregate_metrics,
        "execution_metadata": execution_metadata if config.include_metadata else None,
        "config": config.to_dict()
    }

    # Send to Brokle if configured
    if client and config.send_to_brokle:
        _send_results_to_brokle(client, final_results, experiment_name)

    return final_results


async def _async_evaluate_impl(
    target_function: Callable,
    data: List[Dict[str, Any]],
    evaluators: List[BaseEvaluator],
    config: EvaluationConfig,
    experiment_name: Optional[str],
    client: Any,
    **kwargs
) -> Dict[str, Any]:
    """Asynchronous evaluation implementation with concurrency control."""

    semaphore = asyncio.Semaphore(config.max_concurrency)
    all_results = []
    predictions = []
    execution_metadata = []

    async def process_example(i: int, example: Dict[str, Any]):
        async with semaphore:
            try:
                # Apply sampling if configured
                if config.sample_rate < 1.0:
                    import random
                    if random.random() > config.sample_rate:
                        return None

                # Apply filtering if configured
                if config.filter_fn and not config.filter_fn(example):
                    return None

                # Extract input for target function
                input_data = example.get("input", example.get("inputs", example))

                # Execute target function with span tracking
                prediction_span = None
                if client:
                    prediction_span = client.span(
                        name=f"async_prediction_{i}",
                        metadata={"example_index": i, "experiment": experiment_name}
                    )

                with prediction_span or _nullcontext():
                    # Handle both sync and async target functions
                    if asyncio.iscoroutinefunction(target_function):
                        prediction = await asyncio.wait_for(
                            target_function(input_data, **kwargs),
                            timeout=config.timeout_seconds
                        )
                    else:
                        loop = asyncio.get_event_loop()
                        prediction = await loop.run_in_executor(
                            None, target_function, input_data, **kwargs
                        )

                    # Capture execution metadata
                    metadata = {
                        "example_index": i,
                        "input": input_data if config.include_inputs else None,
                        "output": prediction if config.include_outputs else None,
                        "expected": example.get("expected", example.get("reference"))
                    }

                    return prediction, metadata

            except asyncio.TimeoutError:
                logger.error(f"Prediction timeout for example {i}")
                return None, {
                    "example_index": i,
                    "error": f"timeout after {config.timeout_seconds}s",
                    "input": example.get("input") if config.include_inputs else None
                }

            except Exception as e:
                logger.error(f"Prediction failed for example {i}: {e}")
                if config.fail_fast:
                    raise

                return None, {
                    "example_index": i,
                    "error": str(e),
                    "input": example.get("input") if config.include_inputs else None
                }

    # Process all examples concurrently
    tasks = [process_example(i, example) for i, example in enumerate(data)]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    # Filter and organize results
    for result in results:
        if result is None:
            continue

        prediction, metadata = result
        predictions.append(prediction)
        execution_metadata.append(metadata)

        # Check max samples limit
        if config.max_samples and len(predictions) >= config.max_samples:
            break

    # Run evaluators asynchronously
    evaluator_tasks = []
    for evaluator in evaluators:
        eval_predictions = predictions
        eval_references = [meta.get("expected") for meta in execution_metadata]
        eval_inputs = [meta.get("input") for meta in execution_metadata]

        task = evaluator.abatch_evaluate(
            predictions=eval_predictions,
            references=eval_references,
            inputs=eval_inputs,
            config=config
        )
        evaluator_tasks.append((evaluator, task))

    # Wait for all evaluators to complete
    evaluator_results = {}
    for evaluator, task in evaluator_tasks:
        try:
            results = await task
            evaluator_results[evaluator.name] = {
                "results": results,
                "stats": evaluator.get_stats()
            }
            all_results.extend(results)

        except Exception as e:
            logger.error(f"Evaluator {evaluator.name} failed: {e}")
            if config.fail_fast:
                raise

    # Calculate aggregate metrics
    aggregate_metrics = _calculate_aggregate_metrics(evaluator_results)

    # Prepare final results
    final_results = {
        "experiment_name": experiment_name,
        "total_examples": len(data),
        "processed_examples": len(predictions),
        "evaluator_results": evaluator_results,
        "aggregate_metrics": aggregate_metrics,
        "execution_metadata": execution_metadata if config.include_metadata else None,
        "config": config.to_dict()
    }

    # Send to Brokle if configured
    if client and config.send_to_brokle:
        await _asend_results_to_brokle(client, final_results, experiment_name)

    return final_results


def _calculate_aggregate_metrics(evaluator_results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate aggregate metrics across all evaluators."""

    total_evaluations = 0
    total_score = 0.0
    evaluator_averages = {}

    for evaluator_name, data in evaluator_results.items():
        results = data["results"]
        stats = data["stats"]

        if results:
            scores = [r.score for r in results if r.score is not None]
            if scores:
                avg_score = sum(scores) / len(scores)
                evaluator_averages[evaluator_name] = avg_score
                total_score += sum(scores)
                total_evaluations += len(scores)

    overall_average = total_score / total_evaluations if total_evaluations > 0 else 0.0

    return {
        "overall_average_score": overall_average,
        "total_evaluations": total_evaluations,
        "evaluator_averages": evaluator_averages,
        "evaluator_count": len(evaluator_results)
    }


def _send_results_to_brokle(client: Any, results: Dict[str, Any], experiment_name: Optional[str]) -> None:
    """Send evaluation results to Brokle platform."""
    try:
        # Create evaluation result span
        result_span = client.span(
            name="evaluation_results",
            metadata={
                "experiment_name": experiment_name,
                "result_summary": results.get("aggregate_metrics", {}),
                "total_examples": results.get("total_examples", 0),
                "processed_examples": results.get("processed_examples", 0)
            },
            tags=["evaluation", "results", "summary"]
        )

        with result_span:
            # Log evaluation summary
            logger.info(f"Evaluation complete: {experiment_name}")
            logger.info(f"Overall average score: {results['aggregate_metrics']['overall_average_score']:.3f}")

    except Exception as e:
        logger.error(f"Failed to send results to Brokle: {e}")


async def _asend_results_to_brokle(client: Any, results: Dict[str, Any], experiment_name: Optional[str]) -> None:
    """Async version of sending results to Brokle."""
    try:
        # Create evaluation result span
        result_span = client.span(
            name="async_evaluation_results",
            metadata={
                "experiment_name": experiment_name,
                "result_summary": results.get("aggregate_metrics", {}),
                "total_examples": results.get("total_examples", 0),
                "processed_examples": results.get("processed_examples", 0)
            },
            tags=["evaluation", "async", "results", "summary"]
        )

        with result_span:
            # Log evaluation summary
            logger.info(f"Async evaluation complete: {experiment_name}")
            logger.info(f"Overall average score: {results['aggregate_metrics']['overall_average_score']:.3f}")

    except Exception as e:
        logger.error(f"Failed to send async results to Brokle: {e}")


class _nullcontext:
    """Null context manager for when client span is None."""
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False