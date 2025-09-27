"""
Example usage of Brokle SDK's auto-instrumentation features.

This example demonstrates how to automatically instrument popular LLM libraries
for comprehensive observability without manual decoration.
"""

import asyncio
import os
from typing import List

# Import Brokle SDK
import brokle
from brokle.auto_instrumentation import auto_instrument, get_status, print_status

# Create Brokle client with explicit configuration
client = brokle.Brokle(
    host="http://localhost:8080",  # Your Brokle instance
    api_key="bk_your_secret",  # Your API key
)


def setup_auto_instrumentation():
    """Setup auto-instrumentation for all available libraries."""
    print("=== Setting up Brokle Auto-Instrumentation ===\n")

    # Show current status
    print("📋 Current instrumentation status:")
    print_status()

    # Auto-instrument all available libraries
    print("\n🔧 Starting auto-instrumentation...")
    results = auto_instrument()

    print(f"\n✅ Auto-instrumentation completed:")
    for library, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"   {library}: {status}")

    print("\n📋 Final instrumentation status:")
    print_status()

    return results


def openai_example():
    """Example using OpenAI with auto-instrumentation."""
    print("\n=== OpenAI Auto-Instrumentation Example ===")

    try:
        import openai

        # Create OpenAI client
        openai_client = openai.OpenAI(api_key="your-openai-api-key")

        print("🤖 Making OpenAI API call (automatically instrumented)...")

        # This call will be automatically traced by Brokle
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "What is artificial intelligence?"}],
            max_tokens=100,
        )

        print(f"📝 Response: {response.choices[0].message.content[:100]}...")
        print("✅ OpenAI call completed with automatic observability")

        return response

    except ImportError:
        print("❌ OpenAI library not installed")
        return None
    except Exception as e:
        print(f"❌ OpenAI example failed: {e}")
        return None


async def openai_async_example():
    """Example using OpenAI async client with auto-instrumentation."""
    print("\n=== OpenAI Async Auto-Instrumentation Example ===")

    try:
        import openai

        # Create async OpenAI client
        openai_client = openai.AsyncOpenAI(api_key="your-openai-api-key")

        print("🤖 Making async OpenAI API call (automatically instrumented)...")

        # This async call will be automatically traced by Brokle
        response = await openai_client.chat.completions.create(
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
        print("❌ OpenAI library not installed")
        return None
    except Exception as e:
        print(f"❌ Async OpenAI example failed: {e}")
        return None


def anthropic_example():
    """Example using Anthropic with auto-instrumentation."""
    print("\n=== Anthropic Auto-Instrumentation Example ===")

    try:
        import anthropic

        # Create Anthropic client
        anthropic_client = anthropic.Anthropic(api_key="your-anthropic-api-key")

        print("🤖 Making Anthropic API call (automatically instrumented)...")

        # This call will be automatically traced by Brokle
        response = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": "What is the future of AI?"}],
            max_tokens=100,
        )

        print(f"📝 Response: {response.content[0].text[:100]}...")
        print("✅ Anthropic call completed with automatic observability")

        return response

    except ImportError:
        print("❌ Anthropic library not installed")
        return None
    except Exception as e:
        print(f"❌ Anthropic example failed: {e}")
        return None


def langchain_example():
    """Example using LangChain with auto-instrumentation."""
    print("\n=== LangChain Auto-Instrumentation Example ===")

    try:
        from langchain.chains import LLMChain
        from langchain.llms import OpenAI
        from langchain.prompts import PromptTemplate

        print("🔗 Creating LangChain components (automatically instrumented)...")

        # Create LLM
        llm = OpenAI(openai_api_key="your-openai-api-key", temperature=0.7)

        # Create prompt template
        prompt = PromptTemplate(
            input_variables=["topic"],
            template="Write a brief explanation about {topic}:",
        )

        # Create chain
        chain = LLMChain(llm=llm, prompt=prompt)

        print("🤖 Running LangChain (automatically instrumented)...")

        # This chain execution will be automatically traced by Brokle
        result = chain.run(topic="quantum computing")

        print(f"📝 Result: {result[:100]}...")
        print("✅ LangChain execution completed with automatic observability")

        return result

    except ImportError:
        print("❌ LangChain library not installed")
        return None
    except Exception as e:
        print(f"❌ LangChain example failed: {e}")
        return None


def multiple_providers_example():
    """Example using multiple providers in sequence."""
    print("\n=== Multiple Providers Auto-Instrumentation Example ===")

    results = []

    # Try OpenAI
    openai_result = openai_example()
    if openai_result:
        results.append(("OpenAI", openai_result))

    # Try Anthropic
    anthropic_result = anthropic_example()
    if anthropic_result:
        results.append(("Anthropic", anthropic_result))

    # Try LangChain
    langchain_result = langchain_example()
    if langchain_result:
        results.append(("LangChain", langchain_result))

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


def selective_instrumentation_example():
    """Example of selective instrumentation."""
    print("\n=== Selective Auto-Instrumentation Example ===")

    from brokle.auto_instrumentation import get_registry, instrument, uninstrument

    registry = get_registry()

    # Show current status
    print("📋 Current status:")
    registry.print_status()

    # Uninstrument all
    print("\n🔧 Removing all instrumentation...")
    uninstrument_results = registry.uninstrument_all()

    # Instrument only OpenAI
    print("\n🎯 Instrumenting only OpenAI...")
    openai_result = instrument("openai")
    print(f"OpenAI instrumentation: {'✅ Success' if openai_result else '❌ Failed'}")

    # Show updated status
    print("\n📋 Updated status:")
    registry.print_status()

    # Re-instrument all for other examples
    print("\n🔧 Re-instrumenting all libraries...")
    auto_instrument()


async def main():
    """Main example function."""
    print("=== Brokle Auto-Instrumentation Examples ===\n")

    # 1. Setup auto-instrumentation
    setup_results = setup_auto_instrumentation()

    # 2. Selective instrumentation example
    selective_instrumentation_example()

    # 3. Run examples with different providers
    multiple_providers_example()

    # 4. Async example
    await openai_async_example()

    # 5. Check captured observability data
    await check_observability_data()

    # 6. Final status
    print("\n📋 Final instrumentation status:")
    print_status()

    print("\n✅ All auto-instrumentation examples completed!")
    print("\n💡 Benefits of Auto-Instrumentation:")
    print("   • Zero code changes required")
    print("   • Automatic trace and observation creation")
    print("   • Cost and performance tracking")
    print("   • Provider-agnostic observability")
    print("   • Quality scoring and analytics")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
