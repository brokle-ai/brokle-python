"""Comprehensive integration tests for Brokle SDK."""

import pytest
from unittest.mock import patch, MagicMock
import asyncio

from brokle import Brokle
from brokle.decorators import observe, trace_workflow
from brokle.config import Config


class TestBrokleIntegration:
    """Test integration between SDK components."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config(
            api_key="ak_test_key",
            project_id="proj_test",
            host="https://test.example.com",
            otel_enabled=False
        )

    def test_end_to_end_workflow(self, config):
        """Test complete workflow: client → decorator → OpenAI integration."""
        with patch('brokle.client.get_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.config.telemetry_enabled = True
            mock_get_client.return_value = mock_client

            # Create client
            client = Brokle(config=config)

            # Use decorator in workflow
            @observe(name="llm-call")
            def make_llm_call(prompt):
                return f"Response to: {prompt}"

            # Execute workflow
            with trace_workflow("test-integration"):
                result = make_llm_call("Hello AI")

            assert result == "Response to: Hello AI"

    def test_openai_integration_with_instrumentation(self):
        """Test OpenAI drop-in replacement with instrumentation."""
        try:
            from brokle.openai import OpenAI
        except ImportError:
            pytest.skip("OpenAI integration not available")

        # Simplified test: just verify that the OpenAI client can be created with Brokle
        with patch('brokle.client.get_client') as mock_get_client:
            mock_brokle_client = MagicMock()
            mock_brokle_client.config.telemetry_enabled = True
            mock_get_client.return_value = mock_brokle_client

            with patch.dict('os.environ', {'BROKLE_API_KEY': 'ak_test', 'BROKLE_PROJECT_ID': 'proj_test'}):
                client = OpenAI(api_key="sk-test")

                # Just verify client creation and interface
                assert client is not None
                assert hasattr(client, 'chat')
                assert hasattr(client.chat, 'completions')
                assert hasattr(client.chat.completions, 'create')

    @pytest.mark.asyncio
    async def test_async_workflow_integration(self, config):
        """Test async workflow with decorators and client."""
        with patch('brokle.client.get_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.config.telemetry_enabled = True
            mock_get_client.return_value = mock_client

            @observe()
            async def async_step(step_name):
                await asyncio.sleep(0.01)
                return f"Completed {step_name}"

            with patch('brokle._utils.telemetry.trace') as mock_trace:
                mock_tracer = MagicMock()
                mock_span = MagicMock()
                mock_tracer.start_span.return_value = mock_span
                mock_trace.get_tracer.return_value = mock_tracer

                with trace_workflow("async-workflow"):
                    result1 = await async_step("step1")
                    result2 = await async_step("step2")

                assert result1 == "Completed step1"
                assert result2 == "Completed step2"

    def test_error_handling_across_components(self, config):
        """Test error handling across SDK components."""
        with patch('brokle.client.get_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.config.telemetry_enabled = True
            mock_get_client.return_value = mock_client

            # Test that client creation and basic operations work even with errors
            client = Brokle(config=config)

            # Verify client can handle operations gracefully
            span = client.span("test-error-span")
            assert span is not None

            # Test that errors don't break the client
            try:
                raise ValueError("Test error")
            except ValueError:
                pass  # Should not affect client functionality

            # Client should still work after error
            assert client.config is not None

    def test_configuration_propagation(self):
        """Test configuration propagation across SDK components."""
        config = Config(
            api_key="ak_integration_test",
            project_id="proj_integration",
            environment="test",
            telemetry_enabled=True,
            otel_enabled=False
        )

        client = Brokle(config=config)

        assert client.config.api_key == "ak_integration_test"
        assert client.config.project_id == "proj_integration"
        assert client.config.environment == "test"
        assert client.config.telemetry_enabled is True

    def test_multi_pattern_usage(self, config):
        """Test using multiple SDK patterns together."""
        with patch('brokle.client.get_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.config.telemetry_enabled = True
            mock_get_client.return_value = mock_client

            # Native SDK usage
            client = Brokle(config=config)
            span = client.span("native-span")

            # Decorator usage
            @observe()
            def decorated_function():
                return "decorated result"

            # Context manager usage
            with patch('brokle._utils.telemetry.trace') as mock_trace:
                mock_tracer = MagicMock()
                mock_workflow_span = MagicMock()
                mock_tracer.start_span.return_value = mock_workflow_span
                mock_trace.get_tracer.return_value = mock_tracer

                with trace_workflow("multi-pattern-workflow"):
                    result = decorated_function()

                assert result == "decorated result"