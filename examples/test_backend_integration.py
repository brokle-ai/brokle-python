#!/usr/bin/env python3
"""
Manual End-to-End Test Script for SDK ↔ Backend Communication

This script provides an interactive way to test the SDK against a running backend.
It can be used for manual verification, debugging, and demonstration purposes.

Usage:
    # Start the backend first
    cd /path/to/brokle && make dev-backend

    # Export your API key
    export BROKLE_API_KEY=bk_your_api_key_here

    # Run this script
    python examples/test_backend_integration.py

    # Or with custom host
    BROKLE_HOST=http://localhost:8080 python examples/test_backend_integration.py

Requirements:
    - Running Brokle backend (make dev-backend)
    - Valid API key in BROKLE_API_KEY environment variable
    - Python with brokle SDK installed (pip install -e .)
"""

import os
import sys
import time
import traceback
from typing import Any, Dict, Optional


def print_banner(text: str, char: str = "=") -> None:
    """Print a banner with the given text."""
    width = max(60, len(text) + 4)
    print(f"\n{char * width}")
    print(f"{text:^{width}}")
    print(f"{char * width}\n")


def print_section(text: str) -> None:
    """Print a section header."""
    print(f"\n{'─' * 50}")
    print(f"🧪 {text}")
    print("─" * 50)


def print_success(text: str) -> None:
    """Print success message."""
    print(f"✅ {text}")


def print_error(text: str) -> None:
    """Print error message."""
    print(f"❌ {text}")


def print_warning(text: str) -> None:
    """Print warning message."""
    print(f"⚠️  {text}")


def print_info(text: str) -> None:
    """Print info message."""
    print(f"ℹ️  {text}")


def check_environment() -> tuple[str, str]:
    """Check environment variables and return host and API key."""
    host = os.getenv("BROKLE_HOST", "http://localhost:8080")
    api_key = os.getenv("BROKLE_API_KEY")

    print_section("Environment Check")
    print(f"Host: {host}")

    if api_key:
        # Show first 10 and last 4 characters of API key
        masked_key = (
            api_key[:10] + "..." + api_key[-4:]
            if len(api_key) > 14
            else api_key[:6] + "..."
        )
        print(f"API Key: {masked_key}")
        print_success("Environment variables configured")
    else:
        print_error("BROKLE_API_KEY environment variable not set")
        print("Please set your API key:")
        print("  export BROKLE_API_KEY=bk_your_api_key_here")
        sys.exit(1)

    return host, api_key


def check_backend_health(host: str) -> bool:
    """Check if backend is healthy."""
    print_section("Backend Health Check")

    try:
        import httpx

        # Check basic health endpoint
        response = httpx.get(f"{host}/health", timeout=5.0)
        if response.status_code == 200:
            print_success(f"Backend is healthy at {host}")
        else:
            print_error(f"Backend health check failed: {response.status_code}")
            return False

        # Check database health
        try:
            response = httpx.get(f"{host}/health/db", timeout=5.0)
            if response.status_code == 200:
                print_success("Database connections are healthy")
            else:
                print_warning(f"Database health check returned: {response.status_code}")
        except Exception as e:
            print_warning(f"Database health check failed: {e}")

        return True

    except Exception as e:
        print_error(f"Failed to connect to backend: {e}")
        print("Make sure the backend is running:")
        print(f"  cd /path/to/brokle && make dev-backend")
        return False


def test_sdk_initialization(api_key: str, host: str) -> Optional[Any]:
    """Test SDK initialization."""
    print_section("SDK Initialization")

    try:
        from brokle import Brokle

        client = Brokle(
            api_key=api_key, host=host, environment="manual-test", timeout=30.0
        )

        print_success("SDK client initialized successfully")
        print(f"Client disabled: {client.is_disabled}")
        print(f"Host: {client.config.host}")
        print(f"Environment: {client.config.environment}")
        print(f"Timeout: {client.config.timeout}s")

        return client

    except Exception as e:
        print_error(f"SDK initialization failed: {e}")
        traceback.print_exc()
        return None


def test_authentication(client: Any) -> bool:
    """Test authentication by making a lightweight API call."""
    print_section("Authentication Test")

    try:
        # Use models.list() as a lightweight auth test
        models = client.models.list()
        if models is not None:
            print_success("Authentication successful")
            print(f"Found {len(models.data)} available models")
            return True
        else:
            print_error("Authentication failed - got None response")
            return False

    except Exception as e:
        print_error(f"Authentication failed: {e}")
        if "401" in str(e) or "authentication" in str(e).lower():
            print("This usually means your API key is invalid or expired")
        return False


def test_models_api(client: Any) -> bool:
    """Test models API endpoints."""
    print_section("Models API Test")

    try:
        # Test models list
        print("Testing GET /v1/models...")
        models_response = client.models.list()

        if models_response is None:
            print_error("Models list returned None")
            return False

        print_success(f"Retrieved {len(models_response.data)} models")

        # Show first few models
        for i, model in enumerate(models_response.data[:3]):
            print(
                f"  {i+1}. {model.id} (provider: {getattr(model, 'provider', 'N/A')})"
            )

        if len(models_response.data) > 3:
            print(f"  ... and {len(models_response.data) - 3} more")

        # Test filters
        print("\nTesting filters...")
        filtered = client.models.list(available_only=True)
        available_count = len(filtered.data) if filtered else 0
        print(f"Available models only: {available_count}")

        # Test individual model retrieval if models exist
        if len(models_response.data) > 0:
            first_model = models_response.data[0]
            print(f"\nRetrieving individual model: {first_model.id}")

            model_detail = client.models.retrieve(first_model.id)
            if model_detail:
                print_success(f"Retrieved model details for {model_detail.id}")
            else:
                print_warning("Model retrieval returned None")

        return True

    except Exception as e:
        print_error(f"Models API test failed: {e}")
        traceback.print_exc()
        return False


def test_chat_completions_api(client: Any) -> bool:
    """Test chat completions API."""
    print_section("Chat Completions API Test")

    try:
        print("Testing POST /v1/chat/completions...")

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant for testing.",
                },
                {
                    "role": "user",
                    "content": "Hello! Please respond with exactly: 'Test successful!'",
                },
            ],
            max_tokens=50,
            temperature=0.1,  # Low temperature for consistent response
        )

        if response is None:
            print_error("Chat completions returned None")
            return False

        print_success("Chat completion request successful")

        # Print response details
        if hasattr(response, "choices") and len(response.choices) > 0:
            content = response.choices[0].message.content
            print(f"Response: {content}")
            print(f"Model: {getattr(response, 'model', 'N/A')}")

            if hasattr(response, "usage") and response.usage:
                print(f"Tokens used: {response.usage.total_tokens}")

            # Check for Brokle metadata
            if hasattr(response, "brokle") and response.brokle:
                print(f"Provider: {response.brokle.provider}")
                print(f"Latency: {response.brokle.latency_ms}ms")
                if response.brokle.cost_usd:
                    print(f"Cost: ${response.brokle.cost_usd:.6f}")
                if response.brokle.cache_hit:
                    print("🚀 Cache hit!")

        # Test with Brokle extensions
        print("\nTesting with Brokle extensions...")
        response2 = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'Extensions work!'"}],
            max_tokens=20,
            routing_strategy="cost_optimized",
            cache_strategy="semantic",
            tags=["manual-test", "extensions"],
        )

        if response2:
            print_success("Brokle extensions accepted")

        return True

    except Exception as e:
        print_error(f"Chat completions test failed: {e}")
        if "model" in str(e).lower() and "available" in str(e).lower():
            print_warning(
                "The test model (gpt-3.5-turbo) may not be configured on this backend"
            )
        traceback.print_exc()
        return False


def test_embeddings_api(client: Any) -> bool:
    """Test embeddings API."""
    print_section("Embeddings API Test")

    try:
        print("Testing POST /v1/embeddings...")

        response = client.embeddings.create(
            input="This is a test sentence for embedding generation.",
            model="text-embedding-ada-002",
        )

        if response is None:
            print_error("Embeddings returned None")
            return False

        print_success("Embeddings request successful")

        if hasattr(response, "data") and len(response.data) > 0:
            embedding = response.data[0].embedding
            print(f"Embedding dimensions: {len(embedding)}")
            print(f"First 5 values: {embedding[:5]}")

            if hasattr(response, "usage") and response.usage:
                print(f"Tokens used: {response.usage.total_tokens}")

            # Check for Brokle metadata
            if hasattr(response, "brokle") and response.brokle:
                print(f"Provider: {response.brokle.provider}")
                print(f"Latency: {response.brokle.latency_ms}ms")

        # Test multiple inputs
        print("\nTesting multiple inputs...")
        response2 = client.embeddings.create(
            input=["First sentence", "Second sentence"], model="text-embedding-ada-002"
        )

        if response2 and len(response2.data) == 2:
            print_success("Multiple input embeddings successful")

        return True

    except Exception as e:
        print_error(f"Embeddings test failed: {e}")
        if "model" in str(e).lower():
            print_warning(
                "The test model (text-embedding-ada-002) may not be configured"
            )
        traceback.print_exc()
        return False


def test_telemetry_system(client: Any) -> bool:
    """Test telemetry system."""
    print_section("Telemetry System Test")

    try:
        print("Testing telemetry submission...")

        # Check processor health
        if not client.is_processor_healthy():
            print_warning("Background processor is not healthy")
        else:
            print_success("Background processor is healthy")

        # Get initial metrics
        initial_metrics = client.get_processor_metrics()
        print(f"Queue depth: {initial_metrics.get('queue_depth', 0)}")
        print(f"Items processed: {initial_metrics.get('items_processed', 0)}")

        # Submit some telemetry
        client.submit_telemetry(
            {
                "test_type": "manual_integration",
                "timestamp": time.time(),
                "user": "manual_tester",
            }
        )

        # Submit batch event
        event_id = client.submit_batch_event(
            "manual_test",
            {
                "action": "test_telemetry_system",
                "success": True,
                "timestamp": time.time(),
            },
        )

        print(f"Submitted batch event: {event_id}")

        # Wait a moment for processing
        print("Waiting for telemetry processing...")
        time.sleep(2)

        # Flush and check metrics
        print("Flushing telemetry queue...")
        flushed = client.flush_processor(timeout=10.0)

        if flushed:
            print_success("Telemetry flush completed")
        else:
            print_warning("Telemetry flush timed out")

        # Get final metrics
        final_metrics = client.get_processor_metrics()
        processed_items = final_metrics.get("items_processed", 0)

        if processed_items > initial_metrics.get("items_processed", 0):
            print_success("Telemetry items were processed")
        else:
            print_warning("No telemetry items processed")

        return True

    except Exception as e:
        print_error(f"Telemetry test failed: {e}")
        traceback.print_exc()
        return False


def test_error_handling(client: Any) -> bool:
    """Test error handling scenarios."""
    print_section("Error Handling Test")

    try:
        # Test 404 error
        print("Testing 404 error handling...")
        try:
            client.models.retrieve("non-existent-model-12345")
            print_error("Expected 404 error but got successful response")
            return False
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                print_success("404 error handled correctly")
            else:
                print_warning(f"Got unexpected error: {e}")

        # Test invalid model for chat
        print("Testing invalid model error...")
        try:
            client.chat.completions.create(
                model="definitely-not-a-real-model",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=10,
            )
            print_warning("Invalid model request succeeded unexpectedly")
        except Exception as e:
            print_success(f"Invalid model error handled: {type(e).__name__}")

        return True

    except Exception as e:
        print_error(f"Error handling test failed: {e}")
        return False


def run_performance_test(client: Any) -> bool:
    """Run a simple performance test."""
    print_section("Performance Test")

    try:
        print("Running 5 concurrent models.list() requests...")

        import concurrent.futures
        import time

        def make_request():
            start_time = time.time()
            result = client.models.list()
            end_time = time.time()
            return result, (end_time - start_time) * 1000  # latency in ms

        start_total = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = []

            for future in concurrent.futures.as_completed(futures):
                result, latency = future.result()
                results.append((result, latency))

        end_total = time.time()
        total_time = (end_total - start_total) * 1000

        # Analyze results
        successful_requests = sum(1 for result, _ in results if result is not None)
        latencies = [latency for _, latency in results if latency > 0]

        print(f"Successful requests: {successful_requests}/5")
        print(f"Total time: {total_time:.1f}ms")

        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)

            print(f"Average latency: {avg_latency:.1f}ms")
            print(f"Min latency: {min_latency:.1f}ms")
            print(f"Max latency: {max_latency:.1f}ms")

        if successful_requests == 5:
            print_success("Performance test completed successfully")
            return True
        else:
            print_warning(f"Only {successful_requests}/5 requests succeeded")
            return False

    except Exception as e:
        print_error(f"Performance test failed: {e}")
        return False


def main():
    """Run the complete integration test suite."""
    print_banner("Brokle SDK ↔ Backend Integration Test", "🚀")

    # Check environment
    host, api_key = check_environment()

    # Check backend health
    if not check_backend_health(host):
        print_error("Cannot continue without healthy backend")
        sys.exit(1)

    # Initialize SDK
    client = test_sdk_initialization(api_key, host)
    if not client:
        print_error("Cannot continue without SDK client")
        sys.exit(1)

    # Run tests
    test_results = []

    with client:
        test_results.append(("Authentication", test_authentication(client)))
        test_results.append(("Models API", test_models_api(client)))
        test_results.append(("Chat Completions API", test_chat_completions_api(client)))
        test_results.append(("Embeddings API", test_embeddings_api(client)))
        test_results.append(("Telemetry System", test_telemetry_system(client)))
        test_results.append(("Error Handling", test_error_handling(client)))
        test_results.append(("Performance", run_performance_test(client)))

    # Print summary
    print_banner("Test Results Summary")

    passed = 0
    failed = 0

    for test_name, result in test_results:
        if result:
            print_success(f"{test_name}")
            passed += 1
        else:
            print_error(f"{test_name}")
            failed += 1

    print(f"\n📊 Results: {passed} passed, {failed} failed")

    if failed == 0:
        print_banner("🎉 All Tests Passed! SDK ↔ Backend Communication Working!", "🎉")
        print("Your SDK is successfully communicating with the backend.")
        print("You can now use the SDK in your applications with confidence.")
    else:
        print_banner("❌ Some Tests Failed", "❌")
        print("Please check the error messages above and verify:")
        print("1. Backend is running and healthy")
        print("2. API key is valid and has proper permissions")
        print("3. Required models are configured on the backend")
        sys.exit(1)


if __name__ == "__main__":
    main()
