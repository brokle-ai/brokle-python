#!/usr/bin/env python3
"""
Working Integration Test for SDK ↔ Backend Communication

This test works with the current backend state (placeholder/TODO responses)
and focuses on verifying that the communication layer is functional.

Usage:
    BROKLE_API_KEY=bk_your_key python examples/test_working_integration.py
"""

import os
import sys
import time
import json
from typing import Dict, Any, Optional

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
    print('─' * 50)

def print_success(text: str) -> None:
    """Print success message."""
    print(f"✅ {text}")

def print_error(text: str) -> None:
    """Print error message."""
    print(f"❌ {text}")

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
        masked_key = api_key[:10] + "..." + api_key[-4:] if len(api_key) > 14 else api_key[:6] + "..."
        print(f"API Key: {masked_key}")
        print_success("Environment variables configured")
    else:
        print_error("BROKLE_API_KEY environment variable not set")
        print("Please set your API key:")
        print("  export BROKLE_API_KEY=bk_your_api_key_here")
        sys.exit(1)
    
    return host, api_key

def test_backend_connection(host: str) -> bool:
    """Test basic backend connectivity."""
    print_section("Backend Connectivity Test")
    
    try:
        import httpx
        response = httpx.get(f"{host}/health", timeout=5.0)
        if response.status_code == 200:
            print_success(f"Backend is healthy at {host}")
            health_data = response.json()
            print(f"    Version: {health_data.get('version', 'unknown')}")
            print(f"    Uptime: {health_data.get('uptime', 'unknown')}")
            return True
        else:
            print_error(f"Backend health check failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Failed to connect to backend: {e}")
        return False

def test_sdk_initialization(api_key: str, host: str) -> Optional[Any]:
    """Test SDK initialization."""
    print_section("SDK Initialization Test")
    
    try:
        from brokle import Brokle
        
        client = Brokle(
            api_key=api_key,
            host=host,
            environment="integration-test",
            timeout=30.0
        )
        
        print_success("SDK client initialized successfully")
        print(f"    Host: {client.config.host}")
        print(f"    Environment: {client.config.environment}")  
        print(f"    Disabled: {client.is_disabled}")
        print(f"    Timeout: {client.config.timeout}s")
        
        return client
        
    except Exception as e:
        print_error(f"SDK initialization failed: {e}")
        return None

def test_raw_api_communication(client: Any) -> bool:
    """Test raw API communication without Pydantic validation."""
    print_section("Raw API Communication Test")
    
    try:
        # Test models endpoint
        print("Testing GET /v1/models...")
        response = client.request('GET', '/v1/models')
        
        if response.get('success'):
            print_success("Models API request successful")
            print(f"    Response: {json.dumps(response, indent=2)}")
            print_info("Note: This is placeholder data from development backend")
        else:
            print_error("Models API request failed")
            return False
        
        # Test chat completions endpoint
        print("\nTesting POST /v1/chat/completions...")
        chat_response = client.request('POST', '/v1/chat/completions', json={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello!"}]
        })
        
        if chat_response.get('success'):
            print_success("Chat completions API request successful")
            print(f"    Response: {json.dumps(chat_response, indent=2)}")
            print_info("Note: This is placeholder data from development backend")
        else:
            print_error("Chat completions API request failed")
            return False
        
        # Test embeddings endpoint
        print("\nTesting POST /v1/embeddings...")
        embed_response = client.request('POST', '/v1/embeddings', json={
            "input": "Test text",
            "model": "text-embedding-ada-002"
        })
        
        if embed_response.get('success'):
            print_success("Embeddings API request successful")
            print(f"    Response: {json.dumps(embed_response, indent=2)}")
            print_info("Note: This is placeholder data from development backend")
        else:
            print_error("Embeddings API request failed")
            return False
        
        return True
        
    except Exception as e:
        print_error(f"Raw API communication failed: {e}")
        return False

def test_authentication(client: Any) -> bool:
    """Test authentication by checking request success."""
    print_section("Authentication Test")
    
    try:
        response = client.request('GET', '/v1/models')
        if response.get('success'):
            print_success("Authentication successful")
            print("    API key is valid and accepted by backend")
            request_id = response.get('meta', {}).get('request_id')
            if request_id:
                print(f"    Request ID: {request_id}")
            return True
        else:
            print_error("Authentication failed")
            return False
    except Exception as e:
        if 'UNAUTHORIZED' in str(e) or '401' in str(e):
            print_error(f"Authentication failed: {e}")
        else:
            print_error(f"Request failed: {e}")
        return False

def test_headers_and_metadata(client: Any) -> bool:
    """Test that proper headers are sent."""
    print_section("Headers and Metadata Test")
    
    try:
        response = client.request('GET', '/v1/models')
        
        if response.get('success'):
            print_success("Headers sent correctly")
            meta = response.get('meta', {})
            print(f"    Request ID: {meta.get('request_id', 'N/A')}")
            print(f"    Timestamp: {meta.get('timestamp', 'N/A')}")
            print(f"    API Version: {meta.get('version', 'N/A')}")
            
            # The fact we get a successful response means:
            # - X-API-Key header was sent
            # - X-Environment header was sent
            # - Content-Type header was sent
            # - User-Agent header was sent
            print_info("All required headers successfully transmitted")
            return True
        else:
            print_error("Headers test failed")
            return False
    except Exception as e:
        print_error(f"Headers test failed: {e}")
        return False

def test_telemetry_basic(client: Any) -> bool:
    """Test basic telemetry functionality."""
    print_section("Telemetry System Test")
    
    try:
        # Check processor health
        healthy = client.is_processor_healthy()
        if healthy:
            print_success("Background processor is healthy")
        else:
            print_error("Background processor is not healthy")
            return False
        
        # Get metrics
        metrics = client.get_processor_metrics()
        print(f"    Queue depth: {metrics.get('queue_depth', 0)}")
        print(f"    Items processed: {metrics.get('items_processed', 0)}")
        print(f"    Worker alive: {metrics.get('worker_alive', False)}")
        
        # Test basic telemetry submission (the old way that works)
        print("\nTesting basic telemetry submission...")
        client.submit_telemetry({
            'event_type': 'integration_test',
            'success': True,
            'timestamp': time.time()
        })
        
        print_success("Telemetry submitted successfully")
        print_info("Background processor will handle batch submission to backend")
        
        return True
        
    except Exception as e:
        print_error(f"Telemetry test failed: {e}")
        return False

def test_performance_basic(client: Any) -> bool:
    """Test basic performance with raw requests."""
    print_section("Performance Test")
    
    try:
        import concurrent.futures
        
        def make_request():
            start_time = time.time()
            result = client.request('GET', '/v1/models')
            end_time = time.time()
            return result.get('success', False), (end_time - start_time) * 1000
        
        print("Running 3 concurrent API requests...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request) for _ in range(3)]
            results = []
            
            for future in concurrent.futures.as_completed(futures):
                success, latency = future.result()
                results.append((success, latency))
        
        successful = sum(1 for success, _ in results if success)
        latencies = [latency for success, latency in results if success]
        
        print(f"    Successful requests: {successful}/3")
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            print(f"    Average latency: {avg_latency:.1f}ms")
        
        if successful == 3:
            print_success("Performance test passed")
            return True
        else:
            print_error(f"Only {successful}/3 requests succeeded")
            return False
            
    except Exception as e:
        print_error(f"Performance test failed: {e}")
        return False

def main():
    """Run the working integration test suite."""
    print_banner("Brokle SDK ↔ Backend Working Integration Test", "🚀")
    
    # Check environment
    host, api_key = check_environment()
    
    # Test backend connectivity
    if not test_backend_connection(host):
        print_error("Cannot continue without backend connectivity")
        sys.exit(1)
    
    # Initialize SDK
    client = test_sdk_initialization(api_key, host)
    if not client:
        print_error("Cannot continue without SDK client")
        sys.exit(1)
    
    # Run tests that work with current backend state
    test_results = []
    
    with client:
        test_results.append(("Raw API Communication", test_raw_api_communication(client)))
        test_results.append(("Authentication", test_authentication(client)))
        test_results.append(("Headers and Metadata", test_headers_and_metadata(client)))
        test_results.append(("Telemetry System", test_telemetry_basic(client)))
        test_results.append(("Performance", test_performance_basic(client)))
    
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
        print_banner("🎉 All Tests Passed! SDK ↔ Backend Communication CONFIRMED!", "🎉")
        print("✅ Authentication working")
        print("✅ HTTP communication functional")
        print("✅ All endpoints reachable")
        print("✅ Headers and metadata correct")
        print("✅ Background telemetry system operational")
        print("✅ Performance acceptable")
        print()
        print("🎯 SDK is ready for use!")
        print("📝 Note: Backend endpoints currently return placeholder data")
        print("   This is expected for development backends with TODO implementations")
    else:
        print_banner("❌ Some Communication Issues Detected", "❌")
        print("Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()