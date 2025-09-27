"""
Native SDK Features Example

This example demonstrates the full native SDK capabilities including
advanced routing, analytics, evaluation, and all Brokle features.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List

from brokle import Brokle

# Shared client settings (use environment variables or update here for quick tests)
BROKLE_SETTINGS = {
    "api_key": "bk_your_api_key_here",
    "host": "http://localhost:8080",
    "otel_enabled": False,
}


async def basic_chat_completion():
    """Basic chat completion with native SDK."""
    async with Brokle(**BROKLE_SETTINGS) as client:
        response = await client.chat.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Explain quantum computing in simple terms.",
                },
            ],
            max_tokens=200,
            temperature=0.7,
            # Brokle native features
            routing_strategy="balanced",
            cache_strategy="semantic",
            cache_similarity_threshold=0.85,
            evaluation_metrics=["relevance", "clarity", "accuracy"],
            custom_tags={"topic": "quantum_computing", "difficulty": "beginner"},
        )

        print("Chat Response:")
        print(f"Content: {response.choices[0].message.content}")
        print(f"Model: {response.model}")

        # Industry standard pattern: Platform metadata via response.brokle.*
        if response.brokle:
            print(f"Provider: {response.brokle.provider}")
            print(f"Cost: ${response.brokle.cost_usd:.4f}")
            print(f"Latency: {response.brokle.latency_ms}ms")
            print(f"Cache Hit: {response.brokle.cache_hit}")
            print(f"Quality: {response.brokle.quality_score}")
        else:
            print("No platform metadata available")
        print()


async def advanced_routing_example():
    """Demonstrate advanced routing strategies."""
    async with Brokle(**BROKLE_SETTINGS) as client:

        # Cost-optimized routing
        print("1. Cost-Optimized Routing:")
        response = await client.chat.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "What is machine learning?"}],
            routing_strategy="cost_optimized",
            max_cost_usd=0.01,
            custom_tags={"priority": "low", "budget": "strict"},
        )
        # Industry standard pattern: Platform metadata via response.brokle.*
        if response.brokle:
            print(
                f"Provider: {response.brokle.provider}, Cost: ${response.brokle.cost_usd:.4f}"
            )
            print(f"Routing reason: {response.brokle.routing_reason}")
        print()

        # Quality-optimized routing
        print("2. Quality-Optimized Routing:")
        response = await client.chat.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": "Explain neural networks in detail."}
            ],
            routing_strategy="quality_optimized",
            evaluation_metrics=["accuracy", "depth", "clarity"],
            custom_tags={"priority": "high", "quality": "premium"},
        )
        # Industry standard pattern: Platform metadata via response.brokle.*
        if response.brokle:
            print(
                f"Provider: {response.brokle.provider}, Quality: {response.brokle.quality_score}"
            )
            print(f"Routing decision: {response.brokle.routing_decision}")
        print()

        # Latency-optimized routing
        print("3. Latency-Optimized Routing:")
        response = await client.chat.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Quick answer: What is AI?"}],
            routing_strategy="latency_optimized",
            max_tokens=50,
            custom_tags={"priority": "urgent", "response_time": "fast"},
        )
        # Industry standard pattern: Platform metadata via response.brokle.*
        if response.brokle:
            print(
                f"Provider: {response.brokle.provider}, Latency: {response.brokle.latency_ms}ms"
            )
        print()


async def semantic_caching_example():
    """Demonstrate semantic caching capabilities."""
    async with Brokle(**BROKLE_SETTINGS) as client:

        # First request
        print("1. First request (cache miss):")
        response1 = await client.chat.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            cache_strategy="semantic",
            cache_similarity_threshold=0.8,
            custom_tags={"query_type": "factual", "cache_test": "first"},
        )
        print(f"Response: {response1.choices[0].message.content}")

        # Industry standard pattern: Platform metadata via response.brokle.*
        if response1.brokle:
            print(f"Cache hit: {response1.brokle.cache_hit}")
            print(f"Cost: ${response1.brokle.cost_usd:.4f}")
        print()

        # Similar request (should hit cache)
        print("2. Similar request (cache hit):")
        response2 = await client.chat.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": "What's the capital city of France?"}
            ],
            cache_strategy="semantic",
            cache_similarity_threshold=0.8,
            custom_tags={"query_type": "factual", "cache_test": "similar"},
        )
        print(f"Response: {response2.choices[0].message.content}")

        # Industry standard pattern: Platform metadata via response.brokle.*
        if response2.brokle:
            print(f"Cache hit: {response2.brokle.cache_hit}")
            print(f"Similarity score: {response2.brokle.cache_similarity_score}")
            print(f"Cost: ${response2.brokle.cost_usd:.4f}")
        print()

        # Different request (cache miss)
        print("3. Different request (cache miss):")
        response3 = await client.chat.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "What is the largest ocean?"}],
            cache_strategy="semantic",
            cache_similarity_threshold=0.8,
            custom_tags={"query_type": "factual", "cache_test": "different"},
        )
        print(f"Response: {response3.choices[0].message.content}")

        # Industry standard pattern: Platform metadata via response.brokle.*
        if response3.brokle:
            print(f"Cache hit: {response3.brokle.cache_hit}")
            print(f"Cost: ${response3.brokle.cost_usd:.4f}")
        print()


async def evaluation_example():
    """Demonstrate response evaluation capabilities."""
    async with Brokle(**BROKLE_SETTINGS) as client:

        # Create response with evaluation
        response = await client.chat.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert chef."},
                {
                    "role": "user",
                    "content": "Give me a recipe for chocolate chip cookies.",
                },
            ],
            evaluation_metrics=["relevance", "accuracy", "helpfulness", "clarity"],
            custom_tags={"domain": "cooking", "recipe_type": "dessert"},
        )

        print("Response with Evaluation:")
        print(f"Content: {response.choices[0].message.content[:200]}...")

        # Industry standard pattern: Platform metadata via response.brokle.*
        if response.brokle:
            print(f"Evaluation scores: {response.brokle.evaluation_scores}")
            print(f"Overall quality: {response.brokle.quality_score}")
        print()

        # Submit additional feedback
        if hasattr(response, "id"):
            feedback_result = await client.evaluation.submit_feedback(
                response_id=response.id,
                feedback_type="rating",
                feedback_value=4.5,
                comment="Great recipe, very detailed instructions!",
            )
            print(f"Feedback submitted: {feedback_result}")
            print()


async def analytics_example():
    """Demonstrate analytics capabilities."""
    async with Brokle(**BROKLE_SETTINGS) as client:

        # Get real-time metrics
        print("1. Real-time Metrics:")
        try:
            metrics = await client.analytics.get_real_time_metrics()
            print(f"Real-time metrics: {metrics}")
            print()
        except Exception as e:
            print(f"Error getting real-time metrics: {e}")
            print()

        # Get historical metrics
        print("2. Historical Metrics:")
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)

            metrics = await client.analytics.get_metrics(
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                group_by=["provider", "model"],
                metrics=["request_count", "total_cost", "average_latency"],
                granularity="daily",
            )
            print(f"Historical metrics: {metrics}")
            print()
        except Exception as e:
            print(f"Error getting historical metrics: {e}")
            print()


async def embeddings_example():
    """Demonstrate embeddings with native features."""
    async with Brokle(**BROKLE_SETTINGS) as client:

        response = await client.embeddings.create(
            model="text-embedding-ada-002",
            input=[
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is a subset of artificial intelligence.",
                "Python is a popular programming language.",
            ],
            routing_strategy="cost_optimized",
            cache_strategy="exact",
            custom_tags={"operation": "embedding", "batch_size": 3},
        )

        print("Embeddings Response:")
        print(f"Model: {response.model}")
        print(f"Number of embeddings: {len(response.data)}")
        print(f"Embedding dimensions: {len(response.data[0].embedding)}")

        # Industry standard pattern: Platform metadata via response.brokle.*
        if response.brokle:
            print(f"Provider: {response.brokle.provider}")
            print(f"Cost: ${response.brokle.cost_usd:.4f}")
            print(f"Cache Hit: {response.brokle.cache_hit}")
        print()


async def completions_example():
    """Demonstrate text completions with native features."""
    async with Brokle(**BROKLE_SETTINGS) as client:

        response = await client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt="The future of artificial intelligence includes",
            max_tokens=100,
            temperature=0.8,
            routing_strategy="balanced",
            evaluation_metrics=["creativity", "coherence"],
            custom_tags={"type": "completion", "topic": "ai_future"},
        )

        print("Completion Response:")
        print(f"Text: {response.choices[0].text}")
        print(f"Model: {response.model}")

        # Industry standard pattern: Platform metadata via response.brokle.*
        if response.brokle:
            print(f"Provider: {response.brokle.provider}")
            print(f"Cost: ${response.brokle.cost_usd:.4f}")
            print(f"Quality score: {response.brokle.quality_score}")
        print()


async def error_handling_example():
    """Demonstrate comprehensive error handling."""
    async with Brokle(**BROKLE_SETTINGS) as client:

        # Test quota exceeded
        print("1. Testing quota limits:")
        try:
            response = await client.chat.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Test message"}],
                max_cost_usd=0.0001,  # Very low limit
                custom_tags={"test": "quota_limit"},
            )
            print(f"Response: {response.choices[0].message.content}")
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")
            if hasattr(e, "error_code"):
                print(f"Error code: {e.error_code}")
            if hasattr(e, "details"):
                print(f"Details: {e.details}")
        print()

        # Test invalid model
        print("2. Testing invalid model:")
        try:
            response = await client.chat.create(
                model="invalid-model-name",
                messages=[{"role": "user", "content": "Test message"}],
                custom_tags={"test": "invalid_model"},
            )
            print(f"Response: {response.choices[0].message.content}")
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")
        print()


async def batch_processing_example():
    """Demonstrate batch processing with native features."""
    async with Brokle(**BROKLE_SETTINGS) as client:

        questions = [
            "What is photosynthesis?",
            "How does gravity work?",
            "What causes the seasons?",
            "Why is the sky blue?",
            "How do computers work?",
        ]

        print("Batch Processing with Different Strategies:")

        # Process with different routing strategies
        for i, question in enumerate(questions):
            strategies = [
                "cost_optimized",
                "quality_optimized",
                "latency_optimized",
                "balanced",
            ]
            strategy = strategies[i % len(strategies)]

            response = await client.chat.create(
                model="gpt-4",
                messages=[{"role": "user", "content": question}],
                max_tokens=100,
                routing_strategy=strategy,
                cache_strategy="semantic",
                evaluation_metrics=["relevance", "accuracy"],
                custom_tags={
                    "batch_id": "science_questions",
                    "question_number": i + 1,
                    "strategy": strategy,
                },
            )

            print(f"Q{i+1}: {question}")
            print(f"A{i+1}: {response.choices[0].message.content[:100]}...")

            # Industry standard pattern: Platform metadata via response.brokle.*
            if response.brokle:
                print(f"Strategy: {strategy}, Provider: {response.brokle.provider}")
                print(
                    f"Cost: ${response.brokle.cost_usd:.4f}, Quality: {response.brokle.quality_score}"
                )
            print("---")


async def health_check_example():
    """Demonstrate health check and configuration."""
    async with Brokle(**BROKLE_SETTINGS) as client:

        # Health check
        print("1. Health Check:")
        try:
            health = await client.health_check()
            print(f"Health status: {health}")
            print()
        except Exception as e:
            print(f"Health check failed: {e}")
            print()

        # Get configuration
        print("2. Configuration Info:")
        try:
            config_info = await client.get_config_info()
            print(f"Configuration: {config_info}")
            print()
        except Exception as e:
            print(f"Config info failed: {e}")
            print()


async def main():
    """Run all native SDK examples."""
    print("=== Brokle Native SDK Examples ===\n")

    examples = [
        ("Basic Chat Completion", basic_chat_completion),
        ("Advanced Routing", advanced_routing_example),
        ("Semantic Caching", semantic_caching_example),
        ("Response Evaluation", evaluation_example),
        ("Analytics", analytics_example),
        ("Embeddings", embeddings_example),
        ("Text Completions", completions_example),
        ("Error Handling", error_handling_example),
        ("Batch Processing", batch_processing_example),
        ("Health Check", health_check_example),
    ]

    for name, example_func in examples:
        print(f"=== {name} ===")
        try:
            await example_func()
        except Exception as e:
            print(f"Example failed: {e}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
    print("=== All native SDK examples completed! ===")
