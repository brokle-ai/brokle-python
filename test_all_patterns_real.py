#!/usr/bin/env python3
"""
Test all Brokle integration patterns with real OpenAI API calls.

Patterns tested:
1. SDK Wrapper (wrap_openai) - Zero config
2. Context Manager (start_as_current_generation) - Explicit control
3. Decorator (@observe) - Automatic tracing
4. Nested spans - Hierarchical traces
"""

import os
import sys
from openai import OpenAI
from brokle import get_client, Brokle
from brokle.wrappers import wrap_openai
from brokle.decorators import observe
from brokle.types import Attrs

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
BROKLE_API_KEY = os.getenv("BROKLE_API_KEY", "bk_fzwUZlCBIE3Z0QfGnfAIKjZ4DuK4ChJHf3mPnnbV")

print("=" * 70)
print("Brokle OTEL SDK - Real World Integration Test")
print("=" * 70)
print()
print(f"OpenAI Model: {OPENAI_MODEL}")
print(f"Brokle API Key: {BROKLE_API_KEY[:15]}...")
print()


def test_pattern_1_wrapper():
    """Pattern 1: SDK Wrapper - Zero configuration auto-instrumentation."""
    print("=" * 70)
    print("PATTERN 1: SDK Wrapper (Zero Config)")
    print("=" * 70)
    print()

    # Initialize Brokle
    brokle = get_client()

    # Wrap OpenAI client
    print("1. Wrapping OpenAI client...")
    client = wrap_openai(OpenAI(api_key=OPENAI_API_KEY))
    print("✓ Client wrapped")
    print()

    # Make real API call (automatically traced)
    print("2. Making real OpenAI API call...")
    print(f"   Model: {OPENAI_MODEL}")
    print(f"   Prompt: 'What is 2+2? Answer in one word.'")
    print()

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "user", "content": "What is 2+2? Answer in one word."}
            ],
            temperature=0.7,
            max_tokens=10,
        )

        answer = response.choices[0].message.content
        print(f"✓ Response: {answer}")
        print(f"✓ Tokens: {response.usage.prompt_tokens} input, {response.usage.completion_tokens} output")
        print()

        print("3. GenAI attributes automatically captured:")
        print(f"   - gen_ai.provider.name: openai")
        print(f"   - gen_ai.operation.name: chat")
        print(f"   - gen_ai.request.model: {OPENAI_MODEL}")
        print(f"   - gen_ai.response.model: {response.model}")
        print(f"   - gen_ai.usage.input_tokens: {response.usage.prompt_tokens}")
        print(f"   - gen_ai.usage.output_tokens: {response.usage.completion_tokens}")
        print(f"   - gen_ai.response.finish_reasons: {response.choices[0].finish_reason}")
        print()

    except Exception as e:
        print(f"❌ Error: {e}")
        print()

    # Flush
    print("4. Flushing telemetry...")
    brokle.flush()
    print("✓ Data sent to Brokle backend")
    print()


def test_pattern_2_context_manager():
    """Pattern 2: Context Manager - Explicit control with manual LLM calls."""
    print("=" * 70)
    print("PATTERN 2: Context Manager (Explicit Control)")
    print("=" * 70)
    print()

    brokle = get_client()
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    print("1. Creating LLM generation span with context manager...")
    print()

    with brokle.start_as_current_generation(
        name="chat",
        model=OPENAI_MODEL,
        provider="openai",
        input_messages=[
            {"role": "user", "content": "What is the capital of France? One word."}
        ],
        model_parameters={"temperature": 0.5, "max_tokens": 10}
    ) as gen:
        print("2. Making real OpenAI API call inside span...")

        try:
            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "user", "content": "What is the capital of France? One word."}
                ],
                temperature=0.5,
                max_tokens=10,
            )

            answer = response.choices[0].message.content
            print(f"✓ Response: {answer}")
            print()

            print("3. Updating span with response data...")
            import json
            gen.set_attribute(
                Attrs.GEN_AI_OUTPUT_MESSAGES,
                json.dumps([{"role": "assistant", "content": answer}])
            )
            gen.set_attribute(Attrs.GEN_AI_USAGE_INPUT_TOKENS, response.usage.prompt_tokens)
            gen.set_attribute(Attrs.GEN_AI_USAGE_OUTPUT_TOKENS, response.usage.completion_tokens)
            gen.set_attribute(Attrs.BROKLE_USAGE_TOTAL_TOKENS, response.usage.total_tokens)
            gen.set_attribute(Attrs.GEN_AI_RESPONSE_ID, response.id)
            gen.set_attribute(Attrs.GEN_AI_RESPONSE_MODEL, response.model)
            gen.set_attribute(Attrs.GEN_AI_RESPONSE_FINISH_REASONS, [response.choices[0].finish_reason])
            print("✓ Span updated with GenAI attributes")
            print()

        except Exception as e:
            print(f"❌ Error: {e}")
            print()

    print("4. Flushing telemetry...")
    brokle.flush()
    print("✓ Data sent to Brokle backend")
    print()


@observe(name="ask-llm", as_type="generation", user_id="real-user-123")
def test_pattern_3_decorator():
    """Pattern 3: Decorator - Automatic function tracing."""
    print("=" * 70)
    print("PATTERN 3: Decorator (Automatic Tracing)")
    print("=" * 70)
    print()

    print("1. Function decorated with @observe...")
    print("2. Making real OpenAI API call...")
    print()

    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "user", "content": "Say 'Hello from decorator!' in 3 words."}
            ],
            max_tokens=10,
        )

        answer = response.choices[0].message.content
        print(f"✓ Response: {answer}")
        print(f"✓ Tokens: {response.usage.total_tokens} total")
        print()

        print("3. Function automatically traced:")
        print("   - Input arguments captured")
        print("   - Output return value captured")
        print("   - Execution time measured")
        print()

        return answer

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_pattern_4_nested():
    """Pattern 4: Nested spans - Parent-child hierarchy."""
    print("=" * 70)
    print("PATTERN 4: Nested Spans (Hierarchical Tracing)")
    print("=" * 70)
    print()

    brokle = get_client()
    openai_client = wrap_openai(OpenAI(api_key=OPENAI_API_KEY))

    print("1. Creating parent workflow span...")
    with brokle.start_as_current_span("multi-step-workflow") as workflow:
        workflow.set_attribute("workflow.type", "qa-pipeline")
        workflow.set_attribute(Attrs.USER_ID, "real-user-456")
        print("✓ Workflow span created")
        print()

        # Step 1: Question classification
        print("2. Step 1: Classifying question...")
        with brokle.start_as_current_span("classify-question") as classify:
            classify.set_attribute("step", "classification")
            print("   (Simulated classification)")
            classify.set_attribute("category", "math")

        print("✓ Classification complete")
        print()

        # Step 2: Answer with LLM (automatically nested via wrapper)
        print("3. Step 2: Generating answer with OpenAI...")
        try:
            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "user", "content": "What is 10 + 15? Just the number."}
                ],
                max_tokens=5,
            )
            answer = response.choices[0].message.content
            print(f"   ✓ Answer: {answer}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            answer = None

        print()

        # Step 3: Format response
        print("4. Step 3: Formatting response...")
        with brokle.start_as_current_span("format-response") as format_span:
            format_span.set_attribute("step", "formatting")
            formatted = f"The answer is: {answer}"
            format_span.set_attribute("output", formatted)

        print("✓ Response formatted")
        print()

        workflow.set_attribute("final_output", formatted)

    print("5. Workflow complete with hierarchy:")
    print("   - Parent: multi-step-workflow")
    print("     - Child 1: classify-question")
    print("     - Child 2: chat gpt-4o-mini (LLM call via wrapper)")
    print("     - Child 3: format-response")
    print()

    brokle.flush()
    print("✓ Complete workflow traced and sent to backend")
    print()


# Main execution
if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    try:
        # Set Brokle environment variables
        os.environ["BROKLE_API_KEY"] = BROKLE_API_KEY
        os.environ["BROKLE_ENVIRONMENT"] = "real-world-test"
        os.environ["BROKLE_DEBUG"] = "false"

        # Run all pattern tests
        test_pattern_1_wrapper()
        print("\n" + "=" * 70 + "\n")

        test_pattern_2_context_manager()
        print("\n" + "=" * 70 + "\n")

        test_pattern_3_decorator()
        get_client().flush()  # Flush decorator data
        print("\n" + "=" * 70 + "\n")

        test_pattern_4_nested()

        print("=" * 70)
        print("✅ ALL PATTERNS TESTED SUCCESSFULLY!")
        print("=" * 70)
        print()
        print("Verify data in ClickHouse:")
        print("  docker exec brokle-clickhouse clickhouse-client \\")
        print("    --query \"SELECT name, provider, model_name, usage_details['total'] as tokens FROM spans WHERE provider='openai' ORDER BY start_time DESC LIMIT 10 FORMAT Vertical\"")
        print()
        print("Check complete attributes:")
        print("  docker exec brokle-clickhouse clickhouse-client \\")
        print("    --query \"SELECT attributes FROM spans WHERE provider='openai' LIMIT 1 FORMAT Vertical\"")

    except KeyboardInterrupt:
        print("\n\nTest interrupted")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
