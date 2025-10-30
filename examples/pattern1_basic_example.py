"""
Example usage of Brokle SDK's wrapper functions (Pattern 1).

This example demonstrates how to use wrapper functions to add Brokle observability
to existing OpenAI and Anthropic clients with explicit wrapping.
"""

import asyncio
import os

# Import Brokle SDK
import brokle
from brokle import wrap_openai

# Create Brokle client with explicit configuration
client = brokle.Brokle(
    host="http://localhost:8080",  # Your Brokle instance
    api_key="bk_your_secret",  # Your API key
)


def openai_wrapper_example():
    """Example using OpenAI with Brokle wrapper."""
    print("\n=== OpenAI Wrapper Example ===")

    try:
        from openai import OpenAI

        # Create OpenAI client
        openai_client = OpenAI(api_key="your-openai-api-key")

        # Wrap the client with Brokle observability
        wrapped_client = wrap_openai(openai_client)

        print("🤖 Making OpenAI API call (wrapped with Brokle)...")

        # This call will be automatically traced by Brokle
        response = wrapped_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "What is artificial intelligence?"}],
            max_tokens=100,
        )

        print(f"📝 Response: {response.choices[0].message.content[:100]}...")
        print("✅ OpenAI call completed with automatic observability")

        return response

    except ImportError:
        print("❌ OpenAI library not installed. Install with: pip install openai")
        return None
    except Exception as e:
        print(f"❌ OpenAI example failed: {e}")
        return None


async def openai_async_wrapper_example():
    """Example using OpenAI async client with Brokle wrapper."""
    print("\n=== OpenAI Async Wrapper Example ===")

    try:
        from openai import AsyncOpenAI

        # Create async OpenAI client
        openai_client = AsyncOpenAI(api_key="your-openai-api-key")

        # Wrap the async client with Brokle observability
        wrapped_client = wrap_openai(openai_client)

        print("🤖 Making async OpenAI API call (wrapped with Brokle)...")

        # This async call will be automatically traced by Brokle
        response = await wrapped_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Explain machine learning in simple terms."}
            ],
            max_tokens=100,
        )

        print(f"📝 Response: {response.choices[0].message.content[:100]}...")
        print("✅ Async OpenAI call completed with automatic observability")

        return response

    except ImportError:
        print("❌ OpenAI library not installed. Install with: pip install openai")
        return None
    except Exception as e:
        print(f"❌ Async OpenAI example failed: {e}")
        return None


def anthropic_wrapper_example():
    """Example using Anthropic with Brokle wrapper."""
    print("\n=== Anthropic Wrapper Example ===")

    try:
        from anthropic import Anthropic

        from brokle import wrap_anthropic

        # Create Anthropic client
        anthropic_client = Anthropic(api_key="your-anthropic-api-key")

        # Wrap the client with Brokle observability
        wrapped_client = wrap_anthropic(anthropic_client)

        print("🤖 Making Anthropic API call (wrapped with Brokle)...")

        # This call will be automatically traced by Brokle
        response = wrapped_client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": "What is the future of AI?"}],
            max_tokens=100,
        )

        print(f"📝 Response: {response.content[0].text[:100]}...")
        print("✅ Anthropic call completed with automatic observability")

        return response

    except ImportError:
        print("❌ Anthropic library not installed. Install with: pip install anthropic")
        return None
    except Exception as e:
        print(f"❌ Anthropic example failed: {e}")
        return None


def multiple_providers_example():
    """Example using multiple providers in sequence."""
    print("\n=== Multiple Providers Wrapper Example ===")

    results = []

    # Try OpenAI
    openai_result = openai_wrapper_example()
    if openai_result:
        results.append(("OpenAI", openai_result))

    # Try Anthropic
    anthropic_result = anthropic_wrapper_example()
    if anthropic_result:
        results.append(("Anthropic", anthropic_result))

    print(
        f"\n📊 Completed {len(results)} provider examples with automatic observability"
    )

    return results


async def check_observability_data():
    """Check the observability data captured automatically."""
    print("\n=== Checking Captured Observability Data ===")

    try:
        # Get recent traces
        traces = await client.observability.list_traces(
            limit=10, sort_by="created_at", sort_order="desc"
        )

        print(f"📈 Found {traces.total} total traces")

        if traces.traces:
            print("\n🔍 Recent traces:")
            for trace in traces.traces[:5]:
                print(f"   • {trace.name} (ID: {trace.id[:8]}...)")

        # Get recent observations
        observations = await client.observability.list_observations(
            limit=10, sort_by="created_at", sort_order="desc"
        )

        print(f"\n📊 Found {observations.total} total observations")

        if observations.observations:
            print("\n🔍 Recent observations:")
            for obs in observations.observations[:5]:
                provider = obs.provider or "unknown"
                model = obs.model or "unknown"
                cost = f"${obs.total_cost:.6f}" if obs.total_cost else "N/A"
                print(f"   • {obs.name} ({provider}/{model}) - Cost: {cost}")

    except Exception as e:
        print(f"❌ Failed to check observability data: {e}")


def inline_wrapping_example():
    """Example of inline wrapping (single expression)."""
    print("\n=== Inline Wrapping Example ===")

    try:
        from openai import OpenAI

        print("🔧 Creating and wrapping OpenAI client in one line...")

        # You can also wrap inline during client creation
        client = wrap_openai(OpenAI(api_key="your-openai-api-key"))

        print("🤖 Making API call with inline-wrapped client...")

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, world!"}],
            max_tokens=50,
        )

        print(f"📝 Response: {response.choices[0].message.content}")
        print("✅ Inline wrapping works perfectly!")

        return response

    except ImportError:
        print("❌ OpenAI library not installed")
        return None
    except Exception as e:
        print(f"❌ Inline example failed: {e}")
        return None


async def main():
    """Main example function."""
    print("=== Brokle Wrapper Functions Examples (Pattern 1) ===\n")

    # 1. Run examples with different providers
    multiple_providers_example()

    # 2. Inline wrapping example
    inline_wrapping_example()

    # 3. Async example
    await openai_async_wrapper_example()

    # 4. Check captured observability data
    await check_observability_data()

    print("\n✅ All wrapper function examples completed!")
    print("\n💡 Benefits of Wrapper Functions:")
    print("   • Explicit and clear wrapping with wrap_openai() / wrap_anthropic()")
    print("   • Automatic trace and observation creation")
    print("   • Cost and performance tracking")
    print("   • Provider-agnostic observability")
    print("   • Quality scoring and analytics")
    print("   • Preserves original SDK API")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
