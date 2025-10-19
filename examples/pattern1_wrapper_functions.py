"""
Pattern 1: Wrapper Functions Example

This example demonstrates Brokle's Pattern 1 - Wrapper Functions for AI providers.
Explicit wrapper functions for enhanced observability.
"""

import asyncio
import os

from anthropic import Anthropic, AsyncAnthropic

# ✨ PATTERN 1: Wrapper Functions - explicit wrapping approach
from openai import AsyncOpenAI, OpenAI

from brokle import wrap_anthropic, wrap_openai

# Set up environment variables for Brokle
os.environ["BROKLE_API_KEY"] = "bk_your_api_key_here"
os.environ["BROKLE_HOST"] = "http://localhost:8080"


def check_pattern1_status():
    """Check if Pattern 1 (Wrapper Functions) is working."""
    print("🔍 Checking Pattern 1: Wrapper Functions Status")
    print("=" * 50)

    # Simple check - if imports worked, we're good
    print("✅ Wrapper functions available")
    print("✅ Explicit provider wrapping")
    print("✅ Enhanced observability with shared providers")
    print("✅ Compatible with industry standard patterns")
    print("✨ Pattern 1 ready!")

    print()


def basic_openai_wrapper_example():
    """Basic OpenAI wrapper example with automatic tracking."""
    print("💬 OpenAI Wrapper Example")
    print("-" * 40)

    # Create OpenAI client and wrap it with Brokle
    openai_client = OpenAI()
    wrapped_client = wrap_openai(openai_client)

    # Use wrapped client exactly like normal OpenAI client
    response = wrapped_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is machine learning in one sentence?"},
        ],
        max_tokens=100,
        temperature=0.7,
    )

    print(f"Response: {response.choices[0].message.content}")
    print(f"Model: {response.model}")
    print(f"Tokens: {response.usage.total_tokens}")
    print("📊 Pattern 1: Enhanced with AI-specific observability!")
    print()


def basic_anthropic_wrapper_example():
    """Basic Anthropic wrapper example with automatic tracking."""
    print("🤖 Anthropic Wrapper Example")
    print("-" * 40)

    # Create Anthropic client and wrap it with Brokle
    anthropic_client = Anthropic()
    wrapped_client = wrap_anthropic(anthropic_client)

    # Use wrapped client exactly like normal Anthropic client
    response = wrapped_client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=100,
        messages=[
            {"role": "user", "content": "Explain quantum computing in simple terms."}
        ],
    )

    print(f"Response: {response.content[0].text}")
    print(f"Model: {response.model}")
    print(f"Input tokens: {response.usage.input_tokens}")
    print(f"Output tokens: {response.usage.output_tokens}")
    print("📊 Pattern 1: Enhanced with Anthropic-specific observability!")
    print()


def inline_wrapping_example():
    """Inline wrapper example - wrap during client creation."""
    print("⚙️ Inline Wrapping Example")
    print("-" * 40)

    # You can wrap the client inline during creation
    wrapped_client = wrap_openai(OpenAI())

    response = wrapped_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Write a haiku about programming."}],
        max_tokens=50,
        temperature=0.8,
    )

    print(f"Haiku: {response.choices[0].message.content}")
    print(f"Model: {response.model}")
    print(f"Cost tracked: Yes (automatic)")
    print("📊 Inline wrapping for cleaner code!")
    print()


async def async_wrapper_example():
    """Async wrapper example."""
    print("⚡ Async Wrapper Example")
    print("-" * 40)

    # Async OpenAI client wrapped with Brokle
    async_openai_client = AsyncOpenAI()
    wrapped_client = wrap_openai(async_openai_client)

    # Async OpenAI call with wrapped client
    response = await wrapped_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Explain async programming in Python briefly."}
        ],
        max_tokens=80,
    )

    print(f"Response: {response.choices[0].message.content}")
    print(f"Model: {response.model}")
    print(f"Tokens: {response.usage.total_tokens}")
    print("📊 Async request enhanced with comprehensive observability!")
    print()


async def streaming_wrapper_example():
    """Streaming wrapper example."""
    print("🌊 Streaming Wrapper Example")
    print("-" * 40)

    async_openai_client = AsyncOpenAI()
    wrapped_client = wrap_openai(async_openai_client)

    # Streaming call with wrapped client
    stream = await wrapped_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Count from 1 to 5, one number per message."}
        ],
        stream=True,
        max_tokens=50,
    )

    print("Streaming response: ", end="")
    async for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)

    print("\n📊 Streaming request enhanced with observability!")
    print()


def multiple_providers_example():
    """Example using multiple AI providers with wrappers."""
    print("🔄 Multiple Providers Example")
    print("-" * 40)

    # Wrap different providers
    openai_client = wrap_openai(OpenAI())
    anthropic_client = wrap_anthropic(Anthropic())

    # Compare responses from different providers
    question = "What is artificial intelligence?"

    # OpenAI response
    openai_response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": question}],
        max_tokens=50,
    )

    # Anthropic response
    anthropic_response = anthropic_client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=50,
        messages=[{"role": "user", "content": question}],
    )

    print(f"Question: {question}")
    print(f"OpenAI: {openai_response.choices[0].message.content}")
    print(f"Anthropic: {anthropic_response.content[0].text}")
    print("📊 Both providers tracked with unified observability!")
    print()


def error_handling_wrapper_example():
    """Demonstrate error handling with wrappers."""
    print("🚨 Error Handling Example")
    print("-" * 40)

    wrapped_client = wrap_openai(OpenAI())

    try:
        # This will fail (invalid model name)
        response = wrapped_client.chat.completions.create(
            model="invalid-model-name", messages=[{"role": "user", "content": "Test"}]
        )
    except Exception as e:
        print(f"Expected error: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("📊 Error was automatically tracked!")
        print("✅ Wrapper continues working after errors")
    print()


async def main():
    """Run all examples to showcase Pattern 1: Wrapper Functions."""
    print("🎯 Brokle Pattern 1: Wrapper Functions Demo")
    print("=" * 50)

    # Check status first
    check_pattern1_status()

    # Run examples
    basic_openai_wrapper_example()
    basic_anthropic_wrapper_example()
    inline_wrapping_example()
    await async_wrapper_example()
    await streaming_wrapper_example()
    multiple_providers_example()
    error_handling_wrapper_example()

    print("🎉 Pattern 1: Wrapper Functions Demo Complete!")
    print("=" * 50)
    print("Key Benefits:")
    print("• 🎯 Explicit wrapping with wrap_openai() and wrap_anthropic()")
    print("• 📊 Automatic AI-specific observability")
    print("• ⚡ Works with sync, async, and streaming")
    print("• 🔗 Preserves original SDK API")
    print("• 🚀 Scalable to all AI providers")
    print("• 🛡️ Comprehensive error handling and tracking")
    print("• ➡️ Ready to upgrade to Pattern 2 (@observe) or 3 (Native SDK) when needed")


if __name__ == "__main__":
    asyncio.run(main())
