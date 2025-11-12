"""
Example usage of Brokle SDK's enhanced observability features.

This example demonstrates how to use the new observability client and
enhanced @observe decorator for comprehensive LLM observability.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

# Import Brokle SDK
import brokle
from brokle.observability_decorators import observe_enhanced

# Create Brokle client with explicit configuration
client = brokle.Brokle(
    host="http://localhost:8080",  # Your Brokle instance
    api_key="bk_your_secret",  # Your API key
)


# Example 1: Basic function observability
@observe_enhanced(name="data_processing", as_type="span")
def process_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Process some data with automatic observability."""
    processed_count = 0
    total_value = 0

    for item in data:
        if "value" in item:
            total_value += item["value"]
            processed_count += 1

    return {
        "processed_count": processed_count,
        "total_value": total_value,
        "average": total_value / processed_count if processed_count > 0 else 0,
    }


# Example 2: Async LLM function with detailed observability
@observe_enhanced(
    name="ai_text_generation",
    as_type="llm",
    model="gpt-4",
    provider="openai",
    capture_timing=True,
    level="INFO",
)
async def generate_ai_text(prompt: str, max_tokens: int = 100) -> str:
    """Generate text using AI with comprehensive observability."""
    # Simulate AI API call
    await asyncio.sleep(0.5)  # Simulate API latency

    # Simulate token usage
    response_text = f"Generated response for: {prompt[:50]}..."

    # You could add quality scoring here
    await score_generation_quality(response_text, prompt)

    return response_text


# Example 3: Quality scoring
@observe_enhanced(name="quality_evaluation", as_type="event")
async def score_generation_quality(response: str, prompt: str) -> float:
    """Evaluate the quality of generated text."""
    # Simulate quality evaluation
    quality_score = min(0.95, len(response) / 100.0)  # Simple quality metric

    # Create a quality score in the observability system
    # This would typically be done in the trace/span context
    return quality_score


# Example 4: Complex workflow with nested spans
@observe_enhanced(name="complex_ai_workflow", as_type="span")
async def complex_ai_workflow(user_query: str) -> Dict[str, Any]:
    """A complex AI workflow with multiple steps and observability."""

    # Step 1: Preprocess the query
    preprocessed = await preprocess_query(user_query)

    # Step 2: Generate AI response
    ai_response = await generate_ai_text(preprocessed["processed_query"])

    # Step 3: Post-process the response
    final_response = await postprocess_response(ai_response, preprocessed)

    return final_response


@observe_enhanced(name="query_preprocessing", as_type="span")
async def preprocess_query(query: str) -> Dict[str, Any]:
    """Preprocess user query."""
    # Simulate preprocessing
    await asyncio.sleep(0.1)

    return {
        "original_query": query,
        "processed_query": query.strip().lower(),
        "word_count": len(query.split()),
        "preprocessing_timestamp": datetime.now(timezone.utc).isoformat(),
    }


@observe_enhanced(name="response_postprocessing", as_type="span")
async def postprocess_response(
    response: str, context: Dict[str, Any]
) -> Dict[str, Any]:
    """Post-process AI response."""
    # Simulate post-processing
    await asyncio.sleep(0.1)

    return {
        "original_response": response,
        "processed_response": response.strip(),
        "context": context,
        "response_length": len(response),
        "postprocessing_timestamp": datetime.now(timezone.utc).isoformat(),
    }


# Example 5: Direct observability client usage
async def direct_observability_example():
    """Example of using the observability client directly."""

    # Create a trace
    trace = await client.observability.create_trace(
        name="manual_trace_example",
        user_id="user123",
        session_id="session456",
        metadata={"example": "direct_client_usage"},
    )

    print(f"Created trace: {trace.id}")

    # Create an span
    span = await client.observability.create_span(
        trace_id=trace.id,
        name="manual_span",
        span_type="llm",
        model="gpt-4",
        provider="openai",
        input_data={"prompt": "What is the capital of France?"},
        prompt_tokens=10,
    )

    print(f"Created span: {span.id}")

    # Simulate some work
    await asyncio.sleep(1)

    # Complete the span
    completed = await client.observability.complete_span(
        span.id,
        output_data={"response": "The capital of France is Paris."},
        completion_tokens=8,
        total_tokens=18,
        total_cost=0.0001,
    )

    print(f"Completed span: {completed.id}")

    # Get trace statistics
    stats = await client.observability.get_trace_stats(trace.id)
    print(f"Trace stats: {stats}")

    # Create a quality score
    quality_score = await client.observability.create_quality_score(
        trace_id=trace.id,
        span_id=span.id,
        score_name="relevance",
        score_value=0.95,
        data_type="NUMERIC",
        source="AUTO",
        evaluator_name="example_evaluator",
        comment="High relevance score for geography question",
    )

    print(f"Created quality score: {quality_score.id}")


# Example 6: Batch operations
async def batch_operations_example():
    """Example of batch operations for high-throughput scenarios."""

    # Create multiple traces in batch
    trace_data = [
        {
            "name": f"batch_trace_{i}",
            "external_trace_id": f"ext_trace_{i}",
            "metadata": {"batch_index": i},
        }
        for i in range(5)
    ]

    traces = await client.observability.create_traces_batch(trace_data)
    print(f"Created {len(traces)} traces in batch")

    # Create multiple spans in batch
    span_data = [
        {
            "trace_id": traces[i % len(traces)].id,
            "name": f"batch_span_{i}",
            "external_span_id": f"ext_obs_{i}",
            "type": "llm",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "model": "gpt-3.5-turbo",
            "provider": "openai",
        }
        for i in range(10)
    ]

    spans = await client.observability.create_spans_batch(
        span_data
    )
    print(f"Created {len(spans)} spans in batch")


# Example 7: Analytics and querying
async def analytics_example():
    """Example of querying observability data for analytics."""

    # List recent traces
    traces = await client.observability.list_traces(
        limit=10, sort_by="created_at", sort_order="desc"
    )
    print(
        f"Found {traces.total} total traces, showing {len(traces.traces or [])} recent ones"
    )

    # List spans for a specific model
    spans = await client.observability.list_spans(model="gpt-4", limit=5)
    print(f"Found {spans.total} spans for GPT-4")

    # List quality scores
    quality_scores = await client.observability.list_quality_scores(
        score_name="relevance", limit=5
    )
    print(f"Found {quality_scores.total} relevance scores")


async def main():
    """Main example function."""
    print("=== Brokle Observability Examples ===\n")

    # Example 1: Basic function observability
    print("1. Basic function observability:")
    sample_data = [
        {"value": 10, "category": "A"},
        {"value": 20, "category": "B"},
        {"value": 15, "category": "A"},
    ]
    result = process_data(sample_data)
    print(f"Processing result: {result}\n")

    # Example 2: AI workflow with observability
    print("2. Complex AI workflow:")
    workflow_result = await complex_ai_workflow("What is machine learning?")
    print(f"Workflow result: {workflow_result}\n")

    # Example 3: Direct client usage
    print("3. Direct observability client usage:")
    await direct_observability_example()
    print()

    # Example 4: Batch operations
    print("4. Batch operations:")
    await batch_operations_example()
    print()

    # Example 5: Analytics
    print("5. Analytics and querying:")
    await analytics_example()
    print()

    print("=== Examples completed! ===")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
