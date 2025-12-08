"""
Basic usage example for Brokle OpenTelemetry SDK.

This example demonstrates the core patterns for using the SDK:
1. Client initialization
2. Manual span creation
3. LLM generation tracking
4. Decorator usage
"""

import os

from brokle import Brokle, get_client
from brokle.decorators import observe
from brokle.types import Attrs


# Example 1: Explicit client initialization
def example_explicit_client():
    """Example using explicit client initialization."""
    print("=== Example 1: Explicit Client ===\n")

    # Create client with explicit configuration
    client = Brokle(
        api_key="bk_your_secret_key_here",
        base_url="http://localhost:8080",
        environment="development",
        debug=True,
    )

    # Create a simple span
    with client.start_as_current_span("my-operation") as span:
        span.set_attribute("custom.key", "custom value")
        span.set_attribute(Attrs.USER_ID, "user-123")
        print("Created span: my-operation")

    # Flush to ensure data is sent
    client.flush()
    print("Data flushed successfully\n")


# Example 2: Singleton pattern with environment variables
def example_singleton_client():
    """Example using singleton pattern with environment variables."""
    print("=== Example 2: Singleton Client (Environment Variables) ===\n")

    # Set environment variables (in production, these would be set in your environment)
    os.environ["BROKLE_API_KEY"] = "bk_your_secret_key_here"
    os.environ["BROKLE_ENVIRONMENT"] = "production"
    os.environ["BROKLE_DEBUG"] = "true"

    # Get singleton client
    client = get_client()

    # Create a span
    with client.start_as_current_span("singleton-operation") as span:
        span.set_attribute("result", "success")
        print("Created span using singleton client")

    # Flush
    client.flush()
    print("Data flushed successfully\n")


# Example 3: LLM generation tracking
def example_llm_generation():
    """Example tracking an LLM generation."""
    print("=== Example 3: LLM Generation Tracking ===\n")

    client = get_client()

    # Track an LLM generation (OTEL 1.28+ compliant)
    with client.start_as_current_generation(
        name="chat",
        model="gpt-4",
        provider="openai",
        input_messages=[{"role": "user", "content": "What is the meaning of life?"}],
        model_parameters={
            "temperature": 0.7,
            "max_tokens": 100,
        },
    ) as generation:
        # Simulate LLM call
        print("Simulating LLM API call...")

        # Update with response
        generation.set_attribute(
            Attrs.GEN_AI_OUTPUT_MESSAGES, '[{"role": "assistant", "content": "42"}]'
        )
        generation.set_attribute(Attrs.GEN_AI_USAGE_INPUT_TOKENS, 10)
        generation.set_attribute(Attrs.GEN_AI_USAGE_OUTPUT_TOKENS, 5)
        generation.set_attribute(Attrs.BROKLE_USAGE_TOTAL_TOKENS, 15)
        generation.set_attribute(Attrs.GEN_AI_RESPONSE_ID, "chatcmpl-123")
        generation.set_attribute(Attrs.GEN_AI_RESPONSE_MODEL, "gpt-4-0613")
        generation.set_attribute(Attrs.GEN_AI_RESPONSE_FINISH_REASONS, ["stop"])

        print("LLM generation tracked successfully")

    client.flush()
    print("Data flushed successfully\n")


# Example 4: Using the @observe decorator
@observe(name="process-request", user_id="user-456")
def process_user_request(input_text: str) -> str:
    """Example function with automatic tracing."""
    # This function is automatically traced
    result = f"Processed: {input_text}"
    return result


def example_decorator():
    """Example using the @observe decorator."""
    print("=== Example 4: Decorator Usage ===\n")

    # Initialize client
    client = get_client()

    # Call decorated function (automatically traced)
    result = process_user_request("Hello, world!")
    print(f"Result: {result}")

    # Flush
    client.flush()
    print("Data flushed successfully\n")


# Example 5: Nested spans (parent-child hierarchy)
def example_nested_spans():
    """Example creating nested spans."""
    print("=== Example 5: Nested Spans ===\n")

    client = get_client()

    # Parent span
    with client.start_as_current_span("parent-operation") as parent:
        parent.set_attribute("operation.type", "parent")
        print("Created parent span")

        # Child span 1
        with client.start_as_current_span("child-operation-1") as child1:
            child1.set_attribute("operation.type", "child")
            child1.set_attribute("child.index", 1)
            print("  Created child span 1")

        # Child span 2
        with client.start_as_current_span("child-operation-2") as child2:
            child2.set_attribute("operation.type", "child")
            child2.set_attribute("child.index", 2)
            print("  Created child span 2")

        print("Parent span ending")

    client.flush()
    print("Data flushed successfully\n")


# Example 6: Error handling
def example_error_handling():
    """Example demonstrating error handling."""
    print("=== Example 6: Error Handling ===\n")

    client = get_client()

    try:
        with client.start_as_current_span("operation-with-error") as span:
            span.set_attribute("will_fail", True)
            print("Creating span that will fail")

            # Simulate an error
            raise ValueError("This is a simulated error")

    except ValueError as e:
        print(f"Caught error: {e}")
        print("Error was automatically recorded in span")

    client.flush()
    print("Data flushed successfully\n")


# Main execution
if __name__ == "__main__":
    print("Brokle OpenTelemetry SDK - Basic Usage Examples\n")
    print("=" * 60)
    print()

    # Run examples
    try:
        example_explicit_client()
        example_singleton_client()
        example_llm_generation()
        example_decorator()
        example_nested_spans()
        example_error_handling()

        print("=" * 60)
        print("\nAll examples completed successfully!")
        print("\nNote: These examples use dummy API keys.")
        print("For real usage, set BROKLE_API_KEY environment variable")
        print("or pass api_key parameter to Brokle() constructor.")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback

        traceback.print_exc()
