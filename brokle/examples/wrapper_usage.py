"""
SDK wrapper usage examples for Brokle OpenTelemetry SDK.

This example demonstrates automatic LLM observability using SDK wrappers:
1. OpenAI wrapper
2. Anthropic wrapper
3. Mixed provider usage
"""

import os

from brokle import get_client
from brokle.wrappers import wrap_anthropic, wrap_openai


def example_openai_wrapper():
    """Example using OpenAI wrapper for automatic observability."""
    print("=== Example 1: OpenAI Wrapper ===\n")

    try:
        import openai
    except ImportError:
        print("OpenAI SDK not installed. Install with: pip install openai")
        return

    # Initialize Brokle
    brokle = get_client()

    # Create and wrap OpenAI client
    client = wrap_openai(
        openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", "sk-test-key"))
    )

    # Make OpenAI call (automatically traced)
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello, Brokle!'"},
            ],
            temperature=0.7,
            max_tokens=50,
            user="user-123",  # User tracking
        )

        print(f"Response: {response.choices[0].message.content}")
        print("✓ OpenAI call automatically traced")

    except Exception as e:
        print(f"OpenAI call failed (expected if API key is invalid): {e}")
        print("✓ Error automatically recorded in span")

    # Flush telemetry
    brokle.flush()
    print("Data flushed successfully\n")


def example_anthropic_wrapper():
    """Example using Anthropic wrapper for automatic observability."""
    print("=== Example 2: Anthropic Wrapper ===\n")

    try:
        import anthropic
    except ImportError:
        print("Anthropic SDK not installed. Install with: pip install anthropic")
        return

    # Initialize Brokle
    brokle = get_client()

    # Create and wrap Anthropic client
    client = wrap_anthropic(
        anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "sk-ant-test-key"))
    )

    # Make Anthropic call (automatically traced)
    try:
        response = client.messages.create(
            model="claude-3-opus",
            max_tokens=1024,
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": "Say 'Hello, Brokle!'"}],
            temperature=0.7,
        )

        print(f"Response: {response.content[0].text}")
        print("✓ Anthropic call automatically traced")

    except Exception as e:
        print(f"Anthropic call failed (expected if API key is invalid): {e}")
        print("✓ Error automatically recorded in span")

    # Flush telemetry
    brokle.flush()
    print("Data flushed successfully\n")


def example_mixed_providers():
    """Example using multiple LLM providers in a single workflow."""
    print("=== Example 3: Mixed Provider Usage ===\n")

    try:
        import anthropic
        import openai
    except ImportError as e:
        print(f"SDK not installed: {e}")
        print("Install with: pip install openai anthropic")
        return

    # Initialize Brokle
    brokle = get_client()

    # Create a parent span for the workflow
    with brokle.start_as_current_span("multi-llm-workflow") as workflow_span:
        workflow_span.set_attribute("workflow.type", "comparison")
        print("Starting multi-LLM workflow")

        # Wrap both providers
        openai_client = wrap_openai(
            openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", "sk-test-key"))
        )

        anthropic_client = wrap_anthropic(
            anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
            )
        )

        prompt = "Explain quantum computing in one sentence."

        # OpenAI call (automatically nested under workflow span)
        try:
            print("\n  Calling OpenAI...")
            openai_response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
            )
            print(f"  OpenAI: {openai_response.choices[0].message.content[:50]}...")

        except Exception as e:
            print(f"  OpenAI failed: {e}")

        # Anthropic call (automatically nested under workflow span)
        try:
            print("\n  Calling Anthropic...")
            anthropic_response = anthropic_client.messages.create(
                model="claude-3-opus",
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
            )
            print(f"  Anthropic: {anthropic_response.content[0].text[:50]}...")

        except Exception as e:
            print(f"  Anthropic failed: {e}")

        print("\n✓ Both LLM calls automatically nested under workflow span")

    # Flush telemetry
    brokle.flush()
    print("Data flushed successfully\n")


def example_streaming_detection():
    """Example showing automatic streaming detection."""
    print("=== Example 4: Streaming Detection ===\n")

    try:
        import openai
    except ImportError:
        print("OpenAI SDK not installed. Install with: pip install openai")
        return

    # Initialize Brokle
    brokle = get_client()

    # Wrap OpenAI client
    client = wrap_openai(
        openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", "sk-test-key"))
    )

    # Make streaming call (automatically detected)
    try:
        print("Making streaming call...")
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Count to 5"}],
            stream=True,  # Streaming enabled
            max_tokens=50,
        )

        # Collect streaming response
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content

        print(f"Streamed response: {full_response}")
        print("✓ Streaming flag automatically detected and recorded")

    except Exception as e:
        print(f"Streaming call failed (expected if API key is invalid): {e}")

    # Flush telemetry
    brokle.flush()
    print("Data flushed successfully\n")


# Main execution
if __name__ == "__main__":
    print("Brokle OpenTelemetry SDK - Wrapper Usage Examples\n")
    print("=" * 60)
    print()

    # Set environment variable for Brokle
    os.environ["BROKLE_API_KEY"] = os.getenv("BROKLE_API_KEY", "bk_test_key")
    os.environ["BROKLE_ENVIRONMENT"] = "development"
    os.environ["BROKLE_DEBUG"] = "true"

    # Run examples
    try:
        example_openai_wrapper()
        example_anthropic_wrapper()
        example_mixed_providers()
        example_streaming_detection()

        print("=" * 60)
        print("\nAll wrapper examples completed!")
        print("\nNote: These examples may fail if you don't have valid API keys.")
        print("Set environment variables:")
        print("  - BROKLE_API_KEY (required)")
        print("  - OPENAI_API_KEY (for OpenAI examples)")
        print("  - ANTHROPIC_API_KEY (for Anthropic examples)")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback

        traceback.print_exc()
