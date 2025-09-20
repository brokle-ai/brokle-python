"""
OpenAI Auto-Instrumentation Example

This example demonstrates Brokle's automatic instrumentation for OpenAI.
Shows how to enable comprehensive observability with zero code changes.
"""

import os
import asyncio

# ✨ STEP 1: Enable auto-instrumentation with a single import
import brokle.openai  # This automatically instruments ALL OpenAI usage

# STEP 2: Use OpenAI normally - everything is automatically tracked
from openai import OpenAI, AsyncOpenAI

# Set up environment variables for Brokle
os.environ["BROKLE_API_KEY"] = "ak_your_api_key_here"
os.environ["BROKLE_HOST"] = "http://localhost:8080"
os.environ["BROKLE_PROJECT_ID"] = "proj_your_project_id"


def check_instrumentation_status():
    """Check if auto-instrumentation is working properly."""
    print("🔍 Checking Brokle Auto-Instrumentation Status")
    print("=" * 50)

    status = brokle.openai.get_instrumentation_status()

    print(f"✅ Instrumented: {status['instrumented']}")
    print(f"📦 OpenAI Available: {status['openai_available']}")
    print(f"🔧 Wrapt Available: {status['wrapt_available']}")
    print(f"🌐 Brokle Client Available: {status['client_available']}")

    if status['errors']:
        print(f"⚠️ Errors: {status['errors']}")
    else:
        print("✨ No errors - ready to track OpenAI usage!")

    print()


def basic_chat_example():
    """Basic chat completion with automatic tracking."""
    print("💬 Basic Chat Completion Example")
    print("-" * 40)

    client = OpenAI()

    # Standard OpenAI call - automatically tracked by Brokle
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is machine learning in one sentence?"}
        ],
        max_tokens=100,
        temperature=0.7
    )

    print(f"Response: {response.choices[0].message.content}")
    print(f"Model: {response.model}")
    print(f"Tokens: {response.usage.total_tokens}")
    print("📊 This request was automatically tracked by Brokle!")
    print()


def embeddings_example():
    """Embeddings with automatic tracking."""
    print("🔢 Embeddings Example")
    print("-" * 40)

    client = OpenAI()

    # Standard OpenAI embeddings - automatically tracked
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input="Machine learning is transforming how we solve problems."
    )

    embedding = response.data[0].embedding
    print(f"Generated embedding with {len(embedding)} dimensions")
    print(f"Model: {response.model}")
    print(f"Tokens: {response.usage.total_tokens}")
    print("📊 Embedding request automatically tracked by Brokle!")
    print()


async def async_chat_example():
    """Async chat completion with automatic tracking."""
    print("⚡ Async Chat Completion Example")
    print("-" * 40)

    async with AsyncOpenAI() as client:
        # Async OpenAI call - automatically tracked by Brokle
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Explain async programming in Python briefly."}
            ],
            max_tokens=80
        )

        print(f"Response: {response.choices[0].message.content}")
        print(f"Model: {response.model}")
        print(f"Tokens: {response.usage.total_tokens}")
        print("📊 Async request automatically tracked by Brokle!")
        print()


async def streaming_example():
    """Streaming chat with automatic tracking."""
    print("🌊 Streaming Chat Example")
    print("-" * 40)

    async with AsyncOpenAI() as client:
        # Streaming call - automatically tracked by Brokle
        stream = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Count from 1 to 5, one number per message."}
            ],
            stream=True,
            max_tokens=50
        )

        print("Streaming response: ", end="")
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="", flush=True)

        print("\n📊 Streaming request automatically tracked by Brokle!")
        print()


def framework_compatibility_example():
    """Demonstrate compatibility with other frameworks."""
    print("🔗 Framework Compatibility Example")
    print("-" * 40)

    # This would work with any framework that uses OpenAI internally
    # Example: LangChain, LlamaIndex, or custom frameworks

    client = OpenAI()

    # Multiple calls - all automatically tracked
    for i in range(3):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"What is the number {i + 1}?"}
            ],
            max_tokens=20
        )
        print(f"Call {i + 1}: {response.choices[0].message.content.strip()}")

    print("📊 All 3 requests automatically tracked by Brokle!")
    print("✨ Works with ANY library that uses OpenAI under the hood!")
    print()


def error_handling_example():
    """Demonstrate that errors don't break instrumentation."""
    print("🚨 Error Handling Example")
    print("-" * 40)

    client = OpenAI()

    try:
        # This will fail (invalid model name)
        response = client.chat.completions.create(
            model="invalid-model-name",
            messages=[{"role": "user", "content": "Test"}]
        )
    except Exception as e:
        print(f"Expected error: {type(e).__name__}")
        print("📊 Error was automatically tracked by Brokle!")
        print("✅ Instrumentation continues working after errors")
    print()


async def main():
    """Run all examples to showcase auto-instrumentation."""
    print("🎯 Brokle OpenAI Auto-Instrumentation Demo")
    print("=" * 50)

    # Check status first
    check_instrumentation_status()

    # Run examples
    basic_chat_example()
    embeddings_example()
    await async_chat_example()
    await streaming_example()
    framework_compatibility_example()
    error_handling_example()

    print("🎉 Auto-Instrumentation Demo Complete!")
    print("=" * 50)
    print("Key Benefits:")
    print("• 🔍 Zero code changes - just add one import")
    print("• 📊 Automatic cost, token, and performance tracking")
    print("• ⚡ Works with sync, async, and streaming")
    print("• 🔗 Compatible with LangChain, LlamaIndex, etc.")
    print("• 🛡️ Never breaks your code - graceful error handling")
    print("• 📈 Comprehensive observability out of the box")


if __name__ == "__main__":
    asyncio.run(main())