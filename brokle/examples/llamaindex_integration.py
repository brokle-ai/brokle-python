"""
LlamaIndex integration example for Brokle OpenTelemetry SDK.

This example demonstrates automatic tracing of LlamaIndex applications
using the set_global_handler function.

Features demonstrated:
1. Global handler registration
2. Document indexing tracing
3. Query execution tracing
4. Retrieval tracing
5. LLM call tracing within queries
"""

import os

from brokle import get_client
from brokle.integrations import set_global_handler


def example_global_handler():
    """Example 1: Set up global handler for automatic tracing."""
    print("=== Example 1: Global Handler Setup ===\n")

    # Set Brokle as global handler
    handler = set_global_handler(
        "brokle",
        user_id="user-123",
        session_id="session-456",
        tags=["llamaindex", "example"],
    )

    print("✓ Brokle set as global LlamaIndex handler")
    print("  All LlamaIndex operations will now be automatically traced\n")

    return handler


def example_simple_query():
    """Example 2: Simple query with automatic tracing."""
    print("=== Example 2: Simple Query ===\n")

    try:
        from llama_index import Document, VectorStoreIndex
    except ImportError:
        print("LlamaIndex not installed. Install with: pip install llama-index")
        return

    # Set global handler
    set_global_handler("brokle", user_id="user-123")

    # Create documents
    documents = [
        Document(text="Paris is the capital of France."),
        Document(text="Berlin is the capital of Germany."),
        Document(text="Rome is the capital of Italy."),
    ]

    try:
        # Build index - automatically traced
        print("Building index...")
        index = VectorStoreIndex.from_documents(documents)
        print("✓ Index building traced (includes embedding generation)\n")

        # Query - automatically traced
        print("Querying index...")
        query_engine = index.as_query_engine()
        response = query_engine.query("What is the capital of France?")

        print(f"Response: {response}")
        print("\n✓ Query execution traced with:")
        print("  - query span (parent)")
        print("  - retrieve span (document retrieval)")
        print("  - chat span (LLM generation)")
        print()

    except Exception as e:
        print(f"Query failed (expected if no API key): {e}")
        print("✓ Error automatically recorded\n")

    # Flush telemetry
    get_client().flush()


def example_with_retriever():
    """Example 3: Retrieval tracing."""
    print("=== Example 3: Retrieval Tracing ===\n")

    try:
        from llama_index import Document, VectorStoreIndex
    except ImportError:
        print("LlamaIndex not installed. Install with: pip install llama-index")
        return

    # Set global handler
    set_global_handler("brokle", user_id="user-123", tags=["retrieval"])

    # Create documents with more content
    documents = [
        Document(text="Machine learning is a subset of artificial intelligence."),
        Document(text="Deep learning uses neural networks with multiple layers."),
        Document(text="Natural language processing deals with text and speech."),
        Document(
            text="Computer vision enables machines to interpret visual information."
        ),
    ]

    try:
        # Build index
        print("Building index...")
        index = VectorStoreIndex.from_documents(documents)

        # Create retriever
        retriever = index.as_retriever(similarity_top_k=2)

        # Retrieve - automatically traced
        print("Retrieving relevant documents...")
        nodes = retriever.retrieve("What is deep learning?")

        print(f"Retrieved {len(nodes)} documents")
        for i, node in enumerate(nodes):
            print(f"  {i+1}. Score: {node.score:.3f}")

        print("\n✓ Retrieval traced with:")
        print("  - retrieve span")
        print("  - embedding span (query embedding)")
        print("  - node scores captured")
        print()

    except Exception as e:
        print(f"Retrieval failed (expected if no API key): {e}")
        print("✓ Error automatically recorded\n")

    # Flush telemetry
    get_client().flush()


def example_chat_engine():
    """Example 4: Chat engine with conversation history."""
    print("=== Example 4: Chat Engine ===\n")

    try:
        from llama_index import Document, VectorStoreIndex
    except ImportError:
        print("LlamaIndex not installed. Install with: pip install llama-index")
        return

    # Set global handler
    set_global_handler("brokle", user_id="user-123", session_id="chat-session-789")

    # Create knowledge base
    documents = [
        Document(text="Brokle is an open-source AI observability platform."),
        Document(text="Brokle provides LLM tracing, metrics, and analytics."),
        Document(text="Brokle uses OpenTelemetry for standardized telemetry."),
    ]

    try:
        # Build index
        print("Building knowledge base...")
        index = VectorStoreIndex.from_documents(documents)

        # Create chat engine
        chat_engine = index.as_chat_engine(chat_mode="best")

        # Multi-turn conversation
        print("Starting conversation...")

        response1 = chat_engine.chat("What is Brokle?")
        print(f"Q: What is Brokle?")
        print(f"A: {response1}\n")

        response2 = chat_engine.chat("What does it provide?")
        print(f"Q: What does it provide?")
        print(f"A: {response2}\n")

        print("✓ Chat conversation traced with:")
        print("  - Each turn as separate query span")
        print("  - Retrieval for each question")
        print("  - LLM generation with context")
        print("  - Session ID groups all turns")
        print()

    except Exception as e:
        print(f"Chat failed (expected if no API key): {e}")
        print("✓ Error automatically recorded\n")

    # Flush telemetry
    get_client().flush()


def example_custom_llm():
    """Example 5: Custom LLM with specific parameters."""
    print("=== Example 5: Custom LLM Configuration ===\n")

    try:
        from llama_index import Document, ServiceContext, VectorStoreIndex
        from llama_index.llms import OpenAI
    except ImportError:
        print("LlamaIndex not installed. Install with: pip install llama-index")
        return

    # Set global handler
    set_global_handler("brokle", user_id="user-123")

    # Create custom LLM
    llm = OpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=100,
        api_key=os.getenv("OPENAI_API_KEY", "sk-test-key"),
    )

    # Create service context
    service_context = ServiceContext.from_defaults(llm=llm)

    # Create documents
    documents = [
        Document(text="The sky is blue during the day."),
        Document(text="The sun is a star at the center of our solar system."),
    ]

    try:
        # Build index with custom LLM
        print("Building index with custom LLM...")
        index = VectorStoreIndex.from_documents(
            documents, service_context=service_context
        )

        # Query
        query_engine = index.as_query_engine()
        response = query_engine.query("Why is the sky blue?")

        print(f"Response: {response}\n")
        print("✓ Custom LLM traced with:")
        print("  - Model: gpt-3.5-turbo")
        print("  - Temperature: 0.7")
        print("  - Max tokens: 100")
        print("  - All GenAI 1.28+ attributes captured")
        print()

    except Exception as e:
        print(f"Query failed (expected if no API key): {e}")
        print("✓ Error automatically recorded\n")

    # Flush telemetry
    get_client().flush()


def example_from_documents():
    """Example 6: Loading from actual files (simulated)."""
    print("=== Example 6: Document Loading ===\n")

    try:
        from llama_index import Document, VectorStoreIndex
    except ImportError:
        print("LlamaIndex not installed. Install with: pip install llama-index")
        return

    # Set global handler
    set_global_handler("brokle", user_id="user-123", tags=["document-loading"])

    # Simulate loading documents (in real scenario, use SimpleDirectoryReader)
    print("Loading documents...")
    documents = [
        Document(
            text="Document 1 content about AI and machine learning.",
            metadata={"source": "doc1.txt", "page": 1},
        ),
        Document(
            text="Document 2 content about natural language processing.",
            metadata={"source": "doc2.txt", "page": 1},
        ),
    ]

    try:
        # Build index - document parsing and chunking are traced
        print("Processing documents...")
        index = VectorStoreIndex.from_documents(documents)

        print("\n✓ Document processing traced:")
        print("  - node_parsing span (document parsing)")
        print("  - chunking span (text splitting)")
        print("  - embedding spans (chunk embeddings)")
        print()

    except Exception as e:
        print(f"Document processing failed: {e}")
        print("✓ Error automatically recorded\n")

    # Flush telemetry
    get_client().flush()


# Main execution
if __name__ == "__main__":
    print("Brokle OpenTelemetry SDK - LlamaIndex Integration Examples\n")
    print("=" * 60)
    print()

    # Set environment variables for Brokle
    os.environ["BROKLE_API_KEY"] = os.getenv("BROKLE_API_KEY", "bk_test_key")
    os.environ["BROKLE_ENVIRONMENT"] = "development"
    os.environ["BROKLE_DEBUG"] = "true"

    # Set OpenAI API key for LlamaIndex
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "sk-test-key")

    # Run examples
    try:
        example_global_handler()
        example_simple_query()
        example_with_retriever()
        example_chat_engine()
        example_custom_llm()
        example_from_documents()

        print("=" * 60)
        print("\nAll LlamaIndex integration examples completed!")
        print("\nNote: Examples may fail if you don't have valid API keys.")
        print("Set environment variables:")
        print("  - BROKLE_API_KEY (required)")
        print("  - OPENAI_API_KEY (required for LlamaIndex LLM)")
        print("\nInstall dependencies:")
        print("  pip install brokle-otel[llamaindex]")
        print("  # or")
        print("  pip install llama-index")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback

        traceback.print_exc()
