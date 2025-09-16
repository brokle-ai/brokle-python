"""
OpenAI Drop-in Replacement Example

This example shows how to use Brokle as a drop-in replacement for OpenAI SDK.
Just change the import and get intelligent routing, cost optimization, and observability.
"""

import os
import asyncio

# Drop-in replacement - just change the import!
# Before: from openai import OpenAI
# After: from brokle.openai import OpenAI
from brokle.openai import OpenAI, AsyncOpenAI

# Set up environment variables
os.environ["BROKLE_PUBLIC_KEY"] = "pk_your_public_key_here"
os.environ["BROKLE_HOST"] = "http://localhost:8000"
os.environ["BROKLE_SECRET_KEY"] = "sk_your_secret_key_here"


def sync_chat_example():
    """Example of synchronous chat completion."""
    client = OpenAI()
    
    # Standard OpenAI API call with Brokle enhancements
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        
        # Brokle specific parameters (optional)
        routing_strategy="cost_optimized",  # Route to cheapest provider
        cache_strategy="semantic",          # Use semantic caching
        max_cost_usd=0.01,                 # Cost limit
        custom_tags={"user_id": "123", "session": "chat_session_1"}
    )
    
    print("Response:", response.choices[0].message.content)
    print("Model used:", response.model)
    
    # Brokle metadata is included in response
    if hasattr(response, 'cost_usd'):
        print(f"Cost: ${response.cost_usd:.4f}")
    if hasattr(response, 'provider'):
        print(f"Provider: {response.provider}")
    if hasattr(response, 'cached'):
        print(f"Cached: {response.cached}")


def sync_completion_example():
    """Example of synchronous text completion."""
    client = OpenAI()
    
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt="Once upon a time in a galaxy far, far away",
        max_tokens=50,
        temperature=0.7,
        
        # Brokle parameters
        routing_strategy="quality_optimized",
        evaluation_metrics=["relevance", "creativity"]
    )
    
    print("Completion:", response.choices[0].text)
    print("Finish reason:", response.choices[0].finish_reason)


def sync_embedding_example():
    """Example of synchronous embeddings."""
    client = OpenAI()
    
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input="The quick brown fox jumps over the lazy dog",
        
        # Brokle parameters
        cache_strategy="exact",  # Use exact matching for embeddings
        custom_tags={"type": "embedding", "purpose": "similarity"}
    )
    
    print("Embedding dimensions:", len(response.data[0].embedding))
    print("Model used:", response.model)


async def async_chat_example():
    """Example of asynchronous chat completion."""
    async with AsyncOpenAI() as client:
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a creative writer."},
                {"role": "user", "content": "Write a short story about a robot learning to love."}
            ],
            max_tokens=200,
            temperature=0.8,
            
            # Brokle parameters
            routing_strategy="quality_optimized",
            evaluation_metrics=["creativity", "coherence", "relevance"],
            custom_tags={"genre": "sci-fi", "type": "short_story"}
        )
        
        print("Story:", response.choices[0].message.content)
        print("Usage:", response.usage)


async def async_streaming_example():
    """Example of asynchronous streaming completion."""
    async with AsyncOpenAI() as client:
        stream = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": "Count from 1 to 10 slowly."}
            ],
            stream=True,
            
            # Brokle parameters
            routing_strategy="latency_optimized",
            custom_tags={"type": "streaming", "purpose": "demo"}
        )
        
        print("Streaming response:")
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")
        print()


def batch_processing_example():
    """Example of batch processing with Brokle."""
    client = OpenAI()
    
    prompts = [
        "What is machine learning?",
        "Explain quantum computing.",
        "How does blockchain work?",
        "What is artificial intelligence?"
    ]
    
    responses = []
    for prompt in prompts:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            
            # Brokle parameters
            routing_strategy="cost_optimized",
            cache_strategy="semantic",
            cache_similarity_threshold=0.8,
            custom_tags={"batch": "tech_explanations", "type": "educational"}
        )
        responses.append(response)
    
    for i, response in enumerate(responses):
        print(f"Q{i+1}: {prompts[i]}")
        print(f"A{i+1}: {response.choices[0].message.content}")
        print("---")


def error_handling_example():
    """Example of error handling with Brokle."""
    client = OpenAI()
    
    try:
        # This will fail due to cost limit
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Write a very long story."}],
            max_tokens=10000,
            max_cost_usd=0.001,  # Very low cost limit
        )
        print("Response:", response.choices[0].message.content)
        
    except Exception as e:
        print(f"Error: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Brokle specific error handling
        if hasattr(e, 'error_code'):
            print(f"Error code: {e.error_code}")
        if hasattr(e, 'details'):
            print(f"Error details: {e.details}")


if __name__ == "__main__":
    print("=== OpenAI Drop-in Replacement Examples ===\n")
    
    print("1. Synchronous Chat Completion:")
    sync_chat_example()
    print()
    
    print("2. Synchronous Text Completion:")
    sync_completion_example()
    print()
    
    print("3. Synchronous Embeddings:")
    sync_embedding_example()
    print()
    
    print("4. Asynchronous Chat Completion:")
    asyncio.run(async_chat_example())
    print()
    
    print("5. Asynchronous Streaming:")
    asyncio.run(async_streaming_example())
    print()
    
    print("6. Batch Processing:")
    batch_processing_example()
    print()
    
    print("7. Error Handling:")
    error_handling_example()
    print()
    
    print("=== All examples completed! ===")