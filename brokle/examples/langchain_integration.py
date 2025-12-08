"""
LangChain integration example for Brokle OpenTelemetry SDK.

This example demonstrates automatic tracing of LangChain applications
using the BrokleLangChainCallback handler.

Features demonstrated:
1. Simple LLM call tracing
2. Chain tracing with automatic hierarchy
3. Tool usage tracing
4. Error handling
"""

import os

from brokle import get_client
from brokle.integrations import BrokleLangChainCallback


def example_simple_llm():
    """Example 1: Simple LLM call with automatic tracing."""
    print("=== Example 1: Simple LLM Call ===\n")

    try:
        from langchain.chat_models import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate
    except ImportError:
        print(
            "LangChain not installed. Install with: pip install langchain langchain-openai"
        )
        return

    # Initialize Brokle callback
    callback = BrokleLangChainCallback(
        user_id="user-123",
        session_id="session-456",
        tags=["example", "simple-llm"],
    )

    # Create LLM with callback
    llm = ChatOpenAI(
        temperature=0.7,
        callbacks=[callback],
        api_key=os.getenv("OPENAI_API_KEY", "sk-test-key"),
    )

    # Make LLM call
    try:
        response = llm.predict("Say 'Hello from LangChain!'")
        print(f"Response: {response}")
        print("✓ LLM call automatically traced\n")
    except Exception as e:
        print(f"LLM call failed (expected if no API key): {e}")
        print("✓ Error automatically recorded in span\n")

    # Flush telemetry
    get_client().flush()


def example_chain_tracing():
    """Example 2: LangChain chain with automatic parent-child hierarchy."""
    print("=== Example 2: Chain Tracing ===\n")

    try:
        from langchain.chains import LLMChain
        from langchain.chat_models import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate
    except ImportError:
        print(
            "LangChain not installed. Install with: pip install langchain langchain-openai"
        )
        return

    # Initialize Brokle callback
    callback = BrokleLangChainCallback(
        user_id="user-123",
        session_id="session-456",
        metadata={"example": "chain_tracing"},
    )

    # Create LLM and chain
    llm = ChatOpenAI(
        temperature=0.7,
        callbacks=[callback],
        api_key=os.getenv("OPENAI_API_KEY", "sk-test-key"),
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that answers questions concisely."),
            ("user", "{question}"),
        ]
    )

    chain = LLMChain(llm=llm, prompt=prompt, callbacks=[callback])

    # Run chain
    try:
        result = chain.run(question="What is the capital of France?")
        print(f"Result: {result}")
        print("✓ Chain automatically traced with hierarchy:")
        print("  - Parent: chain:LLMChain")
        print("  - Child: chat gpt-3.5-turbo (LLM call)\n")
    except Exception as e:
        print(f"Chain failed (expected if no API key): {e}")
        print("✓ Error automatically recorded\n")

    # Flush telemetry
    get_client().flush()


def example_multi_step_chain():
    """Example 3: Multi-step chain with tools."""
    print("=== Example 3: Multi-Step Chain with Tools ===\n")

    try:
        from langchain.agents import AgentType, Tool, initialize_agent
        from langchain.chat_models import ChatOpenAI
    except ImportError:
        print(
            "LangChain not installed. Install with: pip install langchain langchain-openai"
        )
        return

    # Initialize Brokle callback
    callback = BrokleLangChainCallback(
        user_id="user-123",
        session_id="session-456",
        tags=["agent", "tools"],
    )

    # Define tools
    def get_weather(location: str) -> str:
        """Fake weather tool for demonstration."""
        return f"The weather in {location} is sunny and 72°F"

    def calculate(expression: str) -> str:
        """Fake calculator for demonstration."""
        try:
            result = eval(expression)
            return f"Result: {result}"
        except:
            return "Invalid expression"

    tools = [
        Tool(
            name="Weather", func=get_weather, description="Get weather for a location"
        ),
        Tool(
            name="Calculator",
            func=calculate,
            description="Calculate mathematical expressions",
        ),
    ]

    # Create agent
    llm = ChatOpenAI(
        temperature=0.7,
        callbacks=[callback],
        api_key=os.getenv("OPENAI_API_KEY", "sk-test-key"),
    )

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        callbacks=[callback],
        verbose=True,
    )

    # Run agent
    try:
        result = agent.run("What's the weather in San Francisco?")
        print(f"\nResult: {result}")
        print("\n✓ Agent execution traced with:")
        print("  - Parent: chain:AgentExecutor")
        print("  - Children: LLM calls + tool executions\n")
    except Exception as e:
        print(f"\nAgent failed (expected if no API key): {e}")
        print("✓ Error automatically recorded\n")

    # Flush telemetry
    get_client().flush()


def example_streaming():
    """Example 4: Streaming LLM calls."""
    print("=== Example 4: Streaming LLM Calls ===\n")

    try:
        from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
        from langchain.chat_models import ChatOpenAI
    except ImportError:
        print(
            "LangChain not installed. Install with: pip install langchain langchain-openai"
        )
        return

    # Initialize Brokle callback
    callback = BrokleLangChainCallback(
        user_id="user-123",
        session_id="session-456",
    )

    # Create LLM with both streaming and Brokle callbacks
    llm = ChatOpenAI(
        temperature=0.7,
        streaming=True,
        callbacks=[
            StreamingStdOutCallbackHandler(),  # Stream to stdout
            callback,  # Also trace with Brokle
        ],
        api_key=os.getenv("OPENAI_API_KEY", "sk-test-key"),
    )

    # Make streaming call
    try:
        print("Streaming response:")
        response = llm.predict("Count to 5")
        print(f"\n\nFull response: {response}")
        print("✓ Streaming call traced (complete response captured)\n")
    except Exception as e:
        print(f"Streaming failed (expected if no API key): {e}")
        print("✓ Error automatically recorded\n")

    # Flush telemetry
    get_client().flush()


def example_error_handling():
    """Example 5: Error handling in chains."""
    print("=== Example 5: Error Handling ===\n")

    try:
        from langchain.chains import LLMChain
        from langchain.chat_models import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate
    except ImportError:
        print(
            "LangChain not installed. Install with: pip install langchain langchain-openai"
        )
        return

    # Initialize Brokle callback
    callback = BrokleLangChainCallback(
        user_id="user-123",
        session_id="session-456",
    )

    # Create chain with invalid API key (will fail)
    llm = ChatOpenAI(
        temperature=0.7,
        callbacks=[callback],
        api_key="invalid-key",  # Intentionally invalid
    )

    prompt = ChatPromptTemplate.from_template("Say {text}")
    chain = LLMChain(llm=llm, prompt=prompt, callbacks=[callback])

    # Try to run (will fail)
    try:
        result = chain.run(text="hello")
        print(f"Result: {result}")
    except Exception as e:
        print(f"Chain failed (expected): {type(e).__name__}")
        print("✓ Error automatically captured in span:")
        print("  - Span status set to ERROR")
        print("  - Exception recorded with traceback")
        print("  - Parent chain span also marked as error\n")

    # Flush telemetry
    get_client().flush()


# Main execution
if __name__ == "__main__":
    print("Brokle OpenTelemetry SDK - LangChain Integration Examples\n")
    print("=" * 60)
    print()

    # Set environment variables for Brokle
    os.environ["BROKLE_API_KEY"] = os.getenv("BROKLE_API_KEY", "bk_test_key")
    os.environ["BROKLE_ENVIRONMENT"] = "development"
    os.environ["BROKLE_DEBUG"] = "true"

    # Run examples
    try:
        example_simple_llm()
        example_chain_tracing()
        example_multi_step_chain()
        example_streaming()
        example_error_handling()

        print("=" * 60)
        print("\nAll LangChain integration examples completed!")
        print("\nNote: Examples may fail if you don't have valid API keys.")
        print("Set environment variables:")
        print("  - BROKLE_API_KEY (required)")
        print("  - OPENAI_API_KEY (for OpenAI examples)")
        print("\nInstall dependencies:")
        print("  pip install brokle-otel[langchain]")
        print("  # or")
        print("  pip install langchain langchain-openai")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback

        traceback.print_exc()
