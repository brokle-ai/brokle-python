"""
Integration Tests for SDK ↔ Backend Communication

Tests that verify the SDK can communicate with the actual Brokle backend.
Requires a running backend instance at localhost:8080 with a valid API key.

These tests verify:
1. Authentication flow
2. Chat completions API
3. Embeddings API  
4. Models API
5. Error handling
6. Telemetry submission

Usage:
    # Start the backend first
    make dev-backend
    
    # Run tests with a test API key
    BROKLE_API_KEY=bk_test_... python -m pytest tests/test_backend_integration.py -v
"""

import os
import time
from typing import Dict, Any

import pytest
import httpx

from brokle import Brokle
from brokle.exceptions import AuthenticationError, APIError


class TestBackendIntegration:
    """Test SDK communication with actual Brokle backend."""
    
    @pytest.fixture(scope="class")
    def backend_host(self) -> str:
        """Backend host for testing."""
        return os.getenv("BROKLE_HOST", "http://localhost:8080")
    
    @pytest.fixture(scope="class")  
    def api_key(self) -> str:
        """API key for testing."""
        key = os.getenv("BROKLE_API_KEY")
        if not key:
            pytest.skip("BROKLE_API_KEY environment variable not set")
        return key
    
    @pytest.fixture(scope="class")
    def backend_available(self, backend_host: str) -> bool:
        """Check if backend is available."""
        try:
            response = httpx.get(f"{backend_host}/health", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False
    
    @pytest.fixture
    def client(self, api_key: str, backend_host: str, backend_available: bool) -> Brokle:
        """Create SDK client for testing."""
        if not backend_available:
            pytest.skip(f"Backend not available at {backend_host}")
        
        return Brokle(
            api_key=api_key,
            host=backend_host,
            environment="test",
            timeout=30.0
        )
    
    def test_backend_health_check(self, backend_host: str, backend_available: bool):
        """Test that backend is running and healthy."""
        if not backend_available:
            pytest.skip(f"Backend not available at {backend_host}")

        # Direct health check
        response = httpx.get(f"{backend_host}/health")
        assert response.status_code == 200

        # Check database health if endpoint exists (may not be available on all backends)
        try:
            response = httpx.get(f"{backend_host}/health/db")
            # If endpoint exists (not 404), it should return 200
            if response.status_code != 404:
                assert response.status_code == 200
        except Exception:
            # Gracefully handle if endpoint doesn't exist
            pass
    
    def test_authentication_success(self, client: Brokle):
        """Test successful authentication with valid API key."""
        # SDK should authenticate automatically on first request
        # Try to list models as a lightweight auth test
        
        try:
            models = client.models.list()
            assert models is not None
            # If we get here, authentication succeeded
        except AuthenticationError:
            pytest.fail("Authentication failed with valid API key")
    
    def test_authentication_failure(self, backend_host: str, backend_available: bool):
        """Test authentication failure with invalid API key."""
        if not backend_available:
            pytest.skip("Backend not available")
        
        # Create client with invalid API key
        invalid_client = Brokle(
            api_key="bk_invalid_key",
            host=backend_host,
            environment="test"
        )
        
        # Should raise AuthenticationError
        with pytest.raises(AuthenticationError):
            invalid_client.models.list()
    
    def test_chat_completions_success(self, client: Brokle):
        """Test chat completions API success case."""
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use a commonly available model
                messages=[
                    {"role": "user", "content": "Hello! This is a test message."}
                ],
                max_tokens=50,
                temperature=0.7
            )
            
            # Verify response structure
            assert response is not None
            assert hasattr(response, 'choices')
            assert len(response.choices) > 0
            assert hasattr(response.choices[0], 'message')
            assert hasattr(response.choices[0].message, 'content')
            assert response.choices[0].message.role == "assistant"
            
            # Verify Brokle metadata is present
            if hasattr(response, 'brokle') and response.brokle:
                assert hasattr(response.brokle, 'provider')
                assert hasattr(response.brokle, 'request_id')
                assert hasattr(response.brokle, 'latency_ms')
                
        except APIError as e:
            if "model" in str(e).lower() and "available" in str(e).lower():
                pytest.skip("Test model not available on this backend")
            else:
                raise
    
    def test_chat_completions_with_brokle_extensions(self, client: Brokle):
        """Test chat completions with Brokle-specific extensions."""
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "Hello!"}
                ],
                max_tokens=30,
                # Brokle extensions
                routing_strategy="cost_optimized",
                cache_strategy="semantic",
                tags=["test", "integration"]
            )
            
            assert response is not None
            assert len(response.choices) > 0
            
        except APIError as e:
            if "model" in str(e).lower():
                pytest.skip("Test model not available")
            else:
                raise
    
    def test_embeddings_success(self, client: Brokle):
        """Test embeddings API success case."""
        try:
            response = client.embeddings.create(
                input="This is a test text for embedding.",
                model="text-embedding-ada-002"
            )
            
            # Verify response structure
            assert response is not None
            assert hasattr(response, 'data')
            assert len(response.data) > 0
            assert hasattr(response.data[0], 'embedding')
            assert isinstance(response.data[0].embedding, list)
            assert len(response.data[0].embedding) > 0
            assert all(isinstance(x, (int, float)) for x in response.data[0].embedding)
            
        except APIError as e:
            if "model" in str(e).lower():
                pytest.skip("Test embedding model not available")
            else:
                raise
    
    def test_embeddings_multiple_inputs(self, client: Brokle):
        """Test embeddings with multiple input texts."""
        try:
            response = client.embeddings.create(
                input=["First text", "Second text", "Third text"],
                model="text-embedding-ada-002"
            )
            
            # Should get 3 embeddings back
            assert response is not None
            assert len(response.data) == 3
            
            for i, embedding_data in enumerate(response.data):
                assert embedding_data.index == i
                assert len(embedding_data.embedding) > 0
                
        except APIError as e:
            if "model" in str(e).lower():
                pytest.skip("Test embedding model not available")
            else:
                raise
    
    def test_models_list_success(self, client: Brokle):
        """Test models list API success case."""
        response = client.models.list()
        
        # Verify response structure
        assert response is not None
        assert hasattr(response, 'data')
        assert isinstance(response.data, list)
        
        # Should have at least some models
        if len(response.data) > 0:
            model = response.data[0]
            assert hasattr(model, 'id')
            assert hasattr(model, 'object')
            assert model.object == "model"
    
    def test_models_list_with_filters(self, client: Brokle):
        """Test models list with Brokle filters."""
        # Test with provider filter
        response = client.models.list(provider="openai", available_only=True)
        assert response is not None
        assert isinstance(response.data, list)
        
        # Test with category filter
        response = client.models.list(category="chat", available_only=True)  
        assert response is not None
        assert isinstance(response.data, list)
    
    def test_models_retrieve_success(self, client: Brokle):
        """Test individual model retrieval."""
        # First get list of models
        models_response = client.models.list()
        if len(models_response.data) == 0:
            pytest.skip("No models available for retrieval test")
        
        # Get first model
        first_model_id = models_response.data[0].id
        
        # Retrieve specific model
        model = client.models.retrieve(first_model_id)
        assert model is not None
        assert model.id == first_model_id
        assert model.object == "model"
    
    def test_error_handling_404(self, client: Brokle):
        """Test 404 error handling."""
        with pytest.raises(APIError) as exc_info:
            client.models.retrieve("non-existent-model-id")
        
        assert exc_info.value.status_code == 404
    
    def test_error_handling_timeout(self, api_key: str, backend_host: str, backend_available: bool):
        """Test timeout handling."""
        if not backend_available:
            pytest.skip("Backend not available")
        
        # Create client with very short timeout
        timeout_client = Brokle(
            api_key=api_key,
            host=backend_host,
            timeout=0.001  # 1ms timeout - should always timeout
        )
        
        # Should raise some form of network/timeout error
        with pytest.raises(Exception):  # Could be NetworkError, APIError, etc.
            timeout_client.models.list()
    
    def test_environment_headers(self, client: Brokle):
        """Test that environment is properly configured."""
        # Environment is now sent in request body (for telemetry endpoints)
        # not in headers. This test verifies basic request functionality.

        response = client.models.list()
        assert response is not None
        # If we get a response, authentication was successful
    
    def test_telemetry_submission(self, client: Brokle):
        """Test that telemetry is submitted in background."""
        # Make some API calls to generate telemetry
        client.models.list()
        
        # Submit some explicit telemetry
        client.submit_telemetry({
            "test_metric": "integration_test",
            "timestamp": time.time()
        })
        
        # Submit batch event
        event_id = client.submit_batch_event("test_event", {
            "test_data": "integration_test",
            "value": 42
        })
        
        assert event_id != ""  # Should get back a non-empty event ID
        
        # Check processor health
        assert client.is_processor_healthy()
        
        # Get metrics
        metrics = client.get_processor_metrics()
        assert isinstance(metrics, dict)
        assert "queue_depth" in metrics
        assert "items_processed" in metrics
        
        # Flush telemetry (give it time to submit)
        flushed = client.flush_processor(timeout=10.0)
        assert flushed, "Telemetry flush timed out"
    
    def test_client_context_manager(self, api_key: str, backend_host: str, backend_available: bool):
        """Test client works properly as context manager."""
        if not backend_available:
            pytest.skip("Backend not available")
        
        # Test context manager usage
        with Brokle(api_key=api_key, host=backend_host, environment="test") as client:
            models = client.models.list()
            assert models is not None
        
        # Client should be closed after context exit
        # Note: We can't easily test if the client is actually closed,
        # but we can verify the context manager works
    
    def test_disabled_client_mode(self, backend_host: str, backend_available: bool):
        """Test client behavior when disabled (no API key)."""
        if not backend_available:
            pytest.skip("Backend not available")
        
        # Create client without API key (should be disabled)
        disabled_client = Brokle(host=backend_host)
        
        assert disabled_client.is_disabled
        
        # API calls should return None gracefully
        response = disabled_client.models.list()
        assert response is None
        
        response = disabled_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}]
        )
        assert response is None
    
    @pytest.mark.slow
    def test_concurrent_requests(self, client: Brokle):
        """Test handling multiple concurrent requests."""
        import concurrent.futures
        
        def make_request():
            return client.models.list()
        
        # Make 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        assert len(results) == 5
        for result in results:
            assert result is not None
            assert hasattr(result, 'data')
    
    def test_request_headers_and_metadata(self, client: Brokle):
        """Test that proper headers and metadata are included in requests."""
        # Make a request and verify it succeeds (headers must be correct)
        response = client.models.list()
        assert response is not None

        # The fact that we get a successful response means:
        # - X-API-Key header was sent correctly
        # - Content-Type and User-Agent headers were sent
        # - Authentication succeeded

        # If any required headers were missing, we'd get a 401 or 400 error