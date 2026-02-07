"""
OpenAI Wrapper Functions Example

This example shows how to use Brokle's wrapper functions for OpenAI.
Explicit wrapping approach with comprehensive observability for all OpenAI usage.

Note: wrap_openai() takes a single argument (the client instance).
Use @observe() or context managers to set trace attributes like tags, session_id, user_id.
"""

import asyncio
import os

# ✨ Wrapper Functions - explicit wrapping approach!
from openai import AsyncOpenAI, OpenAI

from brokle import Brokle, get_client, observe, wrap_openai

# Set up environment variables
os.environ["BROKLE_API_KEY"] = "bk_test"
os.environ["BROKLE_HOST"] = "http://localhost:8080"

os.environ["OPENAI_API_KEY"] = "sk-proj-testkeyforlocaldebuggingonly"

# Initialize Brokle client (reads from env vars)
brokle = get_client()


def sync_chat_example():
    """Example of synchronous chat completion with wrapper function."""
    print("💬 Sync Chat Completion with Wrapper")
    print("-" * 40)

    # Create OpenAI client and wrap it with Brokle (single argument)
    openai_client = OpenAI()
    wrapped_client = wrap_openai(openai_client)

    # Use @observe to attach trace attributes
    @observe(tags=["example", "sync", "chat"], session_id="demo_session_001")
    def do_chat():
        return wrapped_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            temperature=0.7,
            max_tokens=150,
        )

    response = do_chat()

    print("Response:", response.choices[0].message.content)
    print("Model used:", response.model)
    print("Usage:", response.usage)
    print("✅ Enhanced with comprehensive AI observability!")
    print()


def sync_completion_example():
    """Example of synchronous text completion with wrapper."""
    print("📝 Sync Text Completion with Wrapper")
    print("-" * 40)

    wrapped_client = wrap_openai(OpenAI())

    # Use a context manager to set trace attributes
    with brokle.start_as_current_span("text_completion") as span:
        span.update_trace(user_id="demo_user_123", tags=["completion", "creative"])

        response = wrapped_client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt="Once upon a time in a galaxy far, far away",
            max_tokens=50,
            temperature=0.7,
        )

    print("Completion:", response.choices[0].text)
    print("Finish reason:", response.choices[0].finish_reason)
    print("✅ Text completion enhanced with observability!")
    print()


def sync_embedding_example():
    """Example of synchronous embeddings with wrapper."""
    print("🔢 Sync Embeddings with Wrapper")
    print("-" * 40)

    wrapped_client = wrap_openai(OpenAI())

    response = wrapped_client.embeddings.create(
        model="text-embedding-ada-002",
        input="The quick brown fox jumps over the lazy dog",
    )

    print("Embedding dimensions:", len(response.data[0].embedding))
    print("Model used:", response.model)
    print("✅ Embeddings enhanced with metadata tracking!")
    print()


async def async_chat_example():
    """Example of asynchronous chat completion with wrapper."""
    print("⚡ Async Chat Completion with Wrapper")
    print("-" * 40)

    async_openai_client = AsyncOpenAI()
    wrapped_client = wrap_openai(async_openai_client)

    async with wrapped_client:
        response = await wrapped_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a creative writer."},
                {
                    "role": "user",
                    "content": "Write a short story about a robot learning to love.",
                },
            ],
            max_tokens=200,
            temperature=0.8,
        )

        print("Story:", response.choices[0].message.content)
        print("Usage:", response.usage)
        print("✅ Async request enhanced with comprehensive tracking!")
        print()


async def async_streaming_example():
    """Example of asynchronous streaming completion with wrapper."""
    print("🌊 Async Streaming with Wrapper")
    print("-" * 40)

    wrapped_client = wrap_openai(AsyncOpenAI())

    async with wrapped_client:
        stream = await wrapped_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Count from 1 to 10 slowly."}],
            stream=True,
        )

        print("Streaming response:")
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")
        print()
        print("✅ Streaming enhanced with real-time observability!")
        print()


def batch_processing_example():
    """Example of batch processing with wrapper."""
    print("📊 Batch Processing with Wrapper")
    print("-" * 40)

    wrapped_client = wrap_openai(OpenAI())

    prompts = [
        "What is machine learning?",
        "Explain quantum computing.",
        "How does blockchain work?",
        "What is artificial intelligence?",
    ]

    # Group batch calls under a single trace with session context
    with brokle.start_as_current_span("batch_processing") as span:
        span.update_trace(
            session_id="batch_session_003",
            tags=["batch", "education", "tech-explanations"],
        )

        responses = []
        for i, prompt in enumerate(prompts):
            response = wrapped_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
            )
            responses.append(response)

    for i, response in enumerate(responses):
        print(f"Q{i+1}: {prompts[i]}")
        print(f"A{i+1}: {response.choices[0].message.content}")
        print("---")

    print("✅ Batch processing with unified observability!")
    print()


def function_calling_example():
    """Example of function calling with wrapper."""
    print("🔧 Function Calling with Wrapper")
    print("-" * 40)

    wrapped_client = wrap_openai(OpenAI())

    # Define a function for the model to call
    functions = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    ]

    response = wrapped_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "What's the weather like in Boston?"}],
        functions=functions,
        function_call="auto",
    )

    message = response.choices[0].message
    if message.function_call:
        print("Function called:", message.function_call.name)
        print("Arguments:", message.function_call.arguments)
    else:
        print("Response:", message.content)

    print("✅ Function calling enhanced with tool observability!")
    print()


def error_handling_example():
    """Example of error handling with wrapper."""
    print("🚨 Error Handling with Wrapper")
    print("-" * 40)

    wrapped_client = wrap_openai(OpenAI())

    try:
        # This will fail due to invalid model
        response = wrapped_client.chat.completions.create(
            model="invalid-model-name", messages=[{"role": "user", "content": "Test"}]
        )
        print("Response:", response.choices[0].message.content)

    except Exception as e:
        print(f"Error: {e}")
        print(f"Error type: {type(e).__name__}")
        print("✅ Error automatically tracked with enhanced context!")
        print()


def cost_tracking_example():
    """Example showing cost tracking capabilities."""
    print("💰 Cost Tracking with Wrapper")
    print("-" * 40)

    wrapped_client = wrap_openai(OpenAI())

    # Multiple calls to demonstrate cost tracking
    total_requests = 3
    with brokle.start_as_current_span("cost_tracking") as span:
        span.update_trace(tags=["cost-tracking", "budget-monitoring"])

        for i in range(total_requests):
            response = wrapped_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": f"Tell me a fact about number {i+1}"}
                ],
                max_tokens=50,
            )
            print(f"Call {i+1}: {response.choices[0].message.content.strip()}")

    print(f"✅ All {total_requests} requests tracked with cost analysis!")
    print("📊 View detailed cost breakdowns in Brokle dashboard")
    print()


async def main():
    """Run all wrapper function examples."""
    print("🎯 OpenAI Wrapper Functions Examples")
    print("=" * 50)

    print("1. Synchronous Chat Completion:")
    sync_chat_example()

    print("2. Synchronous Text Completion:")
    sync_completion_example()

    print("3. Synchronous Embeddings:")
    sync_embedding_example()

    print("4. Asynchronous Chat Completion:")
    await async_chat_example()

    print("5. Asynchronous Streaming:")
    await async_streaming_example()

    print("6. Batch Processing:")
    batch_processing_example()

    print("7. Function Calling:")
    function_calling_example()

    print("8. Error Handling:")
    error_handling_example()

    print("9. Cost Tracking:")
    cost_tracking_example()

    print("🎉 All OpenAI Wrapper Examples Complete!")
    print("=" * 50)
    print("Key Wrapper Benefits:")
    print("• 🎯 Explicit control over wrapping behavior")
    print("• 📊 Comprehensive AI-specific observability")
    print("• ⚡ Works with sync, async, and streaming")
    print("• 🔧 Enhanced function calling and tool tracking")
    print("• 💰 Automatic cost tracking and optimization")
    print("• 🚨 Advanced error handling and debugging")
    print("• 📈 Unified analytics across all AI operations")


if __name__ == "__main__":
    asyncio.run(main())
