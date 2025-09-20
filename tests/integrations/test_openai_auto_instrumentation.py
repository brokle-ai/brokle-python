"""
Test suite for OpenAI auto-instrumentation.

Tests the automatic instrumentation functionality without requiring a real Brokle backend.
"""

import asyncio

import pytest
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Dict


class TestOpenAIAutoInstrumentation:
    """Test OpenAI auto-instrumentation functionality."""

    def setup_method(self):
        """Set up test environment before each test."""
        # Clear any existing instrumentation
        self._clear_instrumentation()
        # Reset global instrumentation state
        try:
            if 'brokle.integrations.openai' in sys.modules:
                openai_module = sys.modules['brokle.integrations.openai']
                openai_module._instrumented = False
                openai_module._instrumentation_errors = []
        except (ImportError, AttributeError):
            pass

    def teardown_method(self):
        """Clean up after each test."""
        self._clear_instrumentation()
        # Reset global instrumentation state
        try:
            if 'brokle.integrations.openai' in sys.modules:
                openai_module = sys.modules['brokle.integrations.openai']
                openai_module._instrumented = False
                openai_module._instrumentation_errors = []
        except (ImportError, AttributeError):
            pass

    def _clear_instrumentation(self):
        """Clear instrumentation state for clean tests."""
        # Remove instrumentation modules from cache
        modules_to_remove = [
            'brokle.integrations.openai',
            'brokle.openai'
        ]
        for module in modules_to_remove:
            if module in sys.modules:
                del sys.modules[module]

    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI library."""
        with patch.dict('sys.modules', {
            'openai': MagicMock(),
            'openai.resources': MagicMock(),
            'openai.resources.chat': MagicMock(),
            'openai.resources.chat.completions': MagicMock(),
            'openai.resources.completions': MagicMock(),
            'openai.resources.embeddings': MagicMock(),
        }):
            yield sys.modules['openai']

    @pytest.fixture
    def mock_wrapt(self):
        """Mock wrapt library."""
        with patch.dict('sys.modules', {'wrapt': MagicMock()}):
            yield sys.modules['wrapt']

    @pytest.fixture
    def stub_brokle_client(self):
        """Provide a lightweight Brokle client capturing span interactions."""

        class _StubObservationBase:
            def __init__(self, metadata: Dict[str, Any]):
                self.metadata = dict(metadata or {})
                self.started = False
                self.update_calls = []
                self.end_calls = []

            def start(self):
                self.started = True
                return self

            def update(self, *, metadata: Dict[str, Any] = None, **kwargs):
                self.update_calls.append({"metadata": metadata, "kwargs": kwargs})
                if metadata:
                    self.metadata.update(metadata)
                return self

            def end(self, **kwargs):
                self.end_calls.append(kwargs)
                return None

        class _StubGenerationObservation(_StubObservationBase):
            def __init__(self, metadata: Dict[str, Any]):
                super().__init__(metadata)
                self.metric_updates = []

            def update_metrics(self, **kwargs):
                self.metric_updates.append(kwargs)
                return self

        class _StubSpanObservation(_StubObservationBase):
            pass

        class _StubBrokleClient:
            def __init__(self):
                self.generation_invocations = []
                self.span_invocations = []

            def generation(self, **kwargs):
                observation = _StubGenerationObservation(kwargs.get("metadata"))
                observation.model = kwargs.get("model")
                observation.provider = kwargs.get("provider")
                self.generation_invocations.append({
                    "kwargs": kwargs,
                    "observation": observation
                })
                return observation

            def span(self, **kwargs):
                observation = _StubSpanObservation(kwargs.get("metadata"))
                observation.provider = kwargs.get("metadata", {}).get("provider")
                self.span_invocations.append({
                    "kwargs": kwargs,
                    "observation": observation
                })
                return observation

        return _StubBrokleClient()

    def test_import_enables_instrumentation(self, mock_openai, mock_wrapt):
        """Test that importing the module enables auto-instrumentation."""
        # Import the auto-instrumentation module
        import brokle.integrations.openai as openai_auto

        # Check that instrumentation is enabled
        assert openai_auto.is_instrumented() == True
        assert openai_auto.HAS_OPENAI == True
        assert openai_auto.HAS_WRAPT == True

        # Check that wrapt.wrap_function_wrapper was called
        mock_wrapt.wrap_function_wrapper.assert_called()

    def test_instrumentation_status(self, mock_openai, mock_wrapt):
        """Test instrumentation status reporting."""
        with patch('brokle.integrations.openai._get_brokle_client') as mock_get_client:
            mock_get_client.return_value = Mock()

            import brokle.integrations.openai as openai_auto

            status = openai_auto.get_instrumentation_status()

            assert isinstance(status, dict)
            assert 'instrumented' in status
            assert 'openai_available' in status
            assert 'wrapt_available' in status
            assert 'errors' in status
            assert 'client_available' in status

    def test_missing_openai_library(self, mock_wrapt):
        """Test behavior when OpenAI library is not available."""
        # Don't mock OpenAI, simulate missing library
        with patch.dict('sys.modules', {'openai': None}):
            import brokle.integrations.openai as openai_auto

            assert openai_auto.HAS_OPENAI == False
            assert openai_auto.is_instrumented() == False

            status = openai_auto.get_instrumentation_status()
            assert status['openai_available'] == False

    def test_missing_wrapt_library(self, mock_openai):
        """Test behavior when wrapt library is not available."""
        # Don't mock wrapt, simulate missing library
        with patch.dict('sys.modules', {'wrapt': None}):
            import brokle.integrations.openai as openai_auto

            assert openai_auto.HAS_WRAPT == False
            assert openai_auto.is_instrumented() == False

            status = openai_auto.get_instrumentation_status()
            assert status['wrapt_available'] == False

    def test_wrapper_function_creation(self, mock_openai, mock_wrapt):
        """Test that wrapper functions are created correctly."""
        import brokle.integrations.openai as openai_auto

        # Get the number of wrap_function_wrapper calls
        call_count = mock_wrapt.wrap_function_wrapper.call_count

        # Should instrument multiple methods (sync + async for chat, completions, embeddings)
        assert call_count >= 6  # At least 6 methods should be instrumented

        # Check that specific modules are being wrapped
        calls = mock_wrapt.wrap_function_wrapper.call_args_list
        modules_wrapped = [call[0][0] for call in calls]

        expected_modules = [
            "openai.resources.chat.completions",
            "openai.resources.completions",
            "openai.resources.embeddings"
        ]

        for module in expected_modules:
            assert any(module in wrapped for wrapped in modules_wrapped)

    @patch('brokle.integrations.openai._get_brokle_client')
    def test_wrapper_execution_with_client(self, mock_get_client, mock_openai, mock_wrapt, stub_brokle_client):
        """Test wrapper execution when Brokle client is available."""
        mock_get_client.return_value = stub_brokle_client

        import brokle.integrations.openai as openai_auto

        # Get one of the wrapper functions that was created
        calls = mock_wrapt.wrap_function_wrapper.call_args_list
        if calls:
            # Execute the wrapper with mock data
            wrapper_func = calls[0][0][2]  # The wrapper function

            # Mock wrapped function
            mock_wrapped = Mock()
            mock_wrapped.__module__ = "openai.resources.chat.completions"
            mock_wrapped.__qualname__ = "Completions.create"
            mock_wrapped.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Test response"))],
                model="gpt-3.5-turbo",
                usage={"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25}
            )

            # Execute wrapper
            result = wrapper_func(
                mock_wrapped,
                Mock(),  # instance
                (),  # args
                {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "test"}]}
            )

            assert result.choices[0].message.content == "Test response"

            # Verify observability calls were made with new API
            assert len(stub_brokle_client.generation_invocations) == 1
            generation_call = stub_brokle_client.generation_invocations[0]
            observation = generation_call["observation"]

            assert observation.started is True
            assert observation.end_calls, "Observation should be ended"
            # Successful completion should set status_message="success"
            assert observation.end_calls[-1].get("status_message") == "success"

            # Request metadata should be captured
            assert "request" in observation.metadata
            assert observation.metadata["request"]["model"] == "gpt-3.5-turbo"

            # Response metadata should be stored
            assert "response" in observation.metadata
            assert observation.metadata["response"]["model"] == "gpt-3.5-turbo"
            assert observation.metadata["usage"]["total_tokens"] == 25

            # Metrics should capture token counts, cost, and latency
            assert observation.metric_updates, "Expected metric updates for generation spans"
            metrics = observation.metric_updates[-1]
            assert metrics["input_tokens"] == 10
            assert metrics["output_tokens"] == 15
            assert metrics["total_tokens"] == 25
            assert metrics["cost_usd"] == pytest.approx(0.00004, rel=1e-6)
            assert metrics["latency_ms"] >= 0

    @patch('brokle.integrations.openai._get_brokle_client')
    def test_wrapper_execution_without_client(self, mock_get_client, mock_openai, mock_wrapt):
        """Test wrapper execution when Brokle client is not available."""
        mock_get_client.return_value = None

        import brokle.integrations.openai as openai_auto

        # Get one of the wrapper functions that was created
        calls = mock_wrapt.wrap_function_wrapper.call_args_list
        if calls:
            wrapper_func = calls[0][0][2]

            # Mock wrapped function
            mock_wrapped = Mock()
            mock_wrapped.__module__ = "openai.resources.chat.completions"
            mock_wrapped.__qualname__ = "Completions.create"
            expected_result = Mock()
            mock_wrapped.return_value = expected_result

            # Execute wrapper
            result = wrapper_func(
                mock_wrapped,
                Mock(),  # instance
                (),  # args
                {"model": "gpt-3.5-turbo"}
            )

            # Should still return the original result
            assert result == expected_result

            # Should call the original wrapped function
            mock_wrapped.assert_called_once()

    @patch('brokle.integrations.openai._get_brokle_client')
    def test_wrapper_handles_missing_usage(self, mock_get_client, mock_openai, mock_wrapt, stub_brokle_client):
        """Ensure instrumentation handles responses without usage data."""
        mock_get_client.return_value = stub_brokle_client

        import brokle.integrations.openai as openai_auto

        calls = mock_wrapt.wrap_function_wrapper.call_args_list
        if calls:
            wrapper_func = calls[0][0][2]

            mock_wrapped = Mock()
            mock_wrapped.__module__ = "openai.resources.chat.completions"
            mock_wrapped.__qualname__ = "Completions.create"
            mock_wrapped.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="No usage"))],
                model="gpt-3.5-turbo",
                usage=None
            )

            result = wrapper_func(
                mock_wrapped,
                Mock(),
                (),
                {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "test"}]}
            )

            assert result.choices[0].message.content == "No usage"

            observation = stub_brokle_client.generation_invocations[0]["observation"]
            assert observation.metadata["usage"] == {}
            assert observation.metric_updates
            metrics = observation.metric_updates[-1]
            assert metrics["input_tokens"] is None
            assert metrics["output_tokens"] is None
            assert metrics["total_tokens"] is None
            assert metrics["cost_usd"] in (0, None)
            assert observation.end_calls[-1].get("status_message") == "success"

    @pytest.mark.asyncio
    async def test_async_wrapper_records_latency(self, mock_openai, mock_wrapt, stub_brokle_client):
        """Async wrappers should finalize after the awaited call completes."""
        with patch('brokle.integrations.openai._get_brokle_client') as mock_get_client:
            mock_get_client.return_value = stub_brokle_client

            import brokle.integrations.openai as openai_auto  # noqa: F401 - ensure instrumentation runs

            calls = mock_wrapt.wrap_function_wrapper.call_args_list
            async_wrapper = None
            for call in calls:
                if "Async" in call[0][1]:
                    async_wrapper = call[0][2]
                    break

            assert async_wrapper is not None, "Expected async wrapper to be instrumented"

            async def fake_async_method(*args, **kwargs):
                await asyncio.sleep(0)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="Async response"))],
                    model="gpt-3.5-turbo",
                    usage={"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50}
                )

            coroutine = async_wrapper(
                fake_async_method,
                Mock(),
                (),
                {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "async"}]}
            )

            awaited_result = await coroutine
            assert awaited_result.choices[0].message.content == "Async response"

            observation = stub_brokle_client.generation_invocations[0]["observation"]
            assert observation.metric_updates
            metrics = observation.metric_updates[-1]
            assert metrics["input_tokens"] == 20
            assert metrics["output_tokens"] == 30
            assert metrics["total_tokens"] == 50
            assert metrics["latency_ms"] >= 0
            assert observation.end_calls[-1].get("status_message") == "success"

    def test_request_data_extraction(self, mock_openai, mock_wrapt):
        """Test request data extraction function."""
        import brokle.integrations.openai as openai_auto

        # Test with typical chat completion args
        args = (Mock(), "gpt-4", [{"role": "user", "content": "test"}])
        kwargs = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "max_tokens": 100
        }

        result = openai_auto._extract_request_data(args, kwargs)

        assert isinstance(result, dict)
        assert "model" in result
        assert "messages" in result
        assert "temperature" in result
        assert "max_tokens" in result

        # Should not include sensitive or irrelevant fields
        assert "api_key" not in result

    def test_response_data_extraction(self, mock_openai, mock_wrapt):
        """Test response data extraction function."""
        import brokle.integrations.openai as openai_auto

        # Mock OpenAI response object
        mock_response = Mock()
        mock_response.model_dump.return_value = {
            "id": "chatcmpl-123",
            "model": "gpt-3.5-turbo",
            "choices": [{"message": {"content": "Hello there!"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        }

        result = openai_auto._extract_response_data(mock_response)

        assert isinstance(result, dict)
        assert "id" in result
        assert "model" in result
        assert "choices" in result
        assert "usage" in result

    def test_cost_calculation(self, mock_openai, mock_wrapt):
        """Test cost calculation function."""
        import brokle.integrations.openai as openai_auto

        # Test with known model and usage
        usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500
        }

        # Test with gpt-3.5-turbo
        cost = openai_auto._calculate_cost("gpt-3.5-turbo", usage)
        assert isinstance(cost, float)
        assert cost > 0

        # Test with gpt-4
        cost_gpt4 = openai_auto._calculate_cost("gpt-4", usage)
        assert isinstance(cost_gpt4, float)
        assert cost_gpt4 > cost  # GPT-4 should be more expensive

        # Test with unknown model (should use fallback pricing)
        cost_unknown = openai_auto._calculate_cost("unknown-model", usage)
        assert isinstance(cost_unknown, float)
        assert cost_unknown > 0

    def test_error_handling_in_wrapper(self, mock_openai, mock_wrapt, stub_brokle_client):
        """Test error handling when wrapped function raises exception."""
        with patch('brokle.integrations.openai._get_brokle_client') as mock_get_client:
            mock_get_client.return_value = stub_brokle_client

            import brokle.integrations.openai as openai_auto

            # Get wrapper function
            calls = mock_wrapt.wrap_function_wrapper.call_args_list
            if calls:
                wrapper_func = calls[0][0][2]

                # Mock wrapped function that raises an exception
                mock_wrapped = Mock()
                mock_wrapped.__module__ = "openai.resources.chat.completions"
                mock_wrapped.__qualname__ = "Completions.create"
                mock_wrapped.side_effect = Exception("API Error")

                # Execute wrapper and expect exception to be re-raised
                with pytest.raises(Exception, match="API Error"):
                    wrapper_func(
                        mock_wrapped,
                        Mock(),
                        (),
                        {"model": "gpt-3.5-turbo"}
                    )

                # Verify that observation was created and error was recorded
                assert len(stub_brokle_client.generation_invocations) == 1
                observation = stub_brokle_client.generation_invocations[0]["observation"]
                assert observation.started is True
                assert observation.end_calls, "Expected observation to end even on error"
                end_kwargs = observation.end_calls[-1]
                assert "status_message" in end_kwargs
                assert "error:" in end_kwargs["status_message"]

    def test_integration_with_brokle_openai_module(self, mock_openai, mock_wrapt):
        """Test integration through brokle.openai module."""
        # Import through the convenience module
        import brokle.openai as openai_module

        # Check that auto-instrumentation functions are available
        assert hasattr(openai_module, 'is_instrumented')
        assert hasattr(openai_module, 'get_instrumentation_status')
        assert hasattr(openai_module, 'HAS_OPENAI')
        assert hasattr(openai_module, 'HAS_WRAPT')

        # Check that instrumentation is active
        assert openai_module.is_instrumented() == True

    def test_instrumentation_registry_integration(self, mock_openai, mock_wrapt):
        """Test integration with the main integrations registry."""
        # Force a fresh import cycle to ensure instrumentation runs with mocked dependencies
        # Clear both the integrations registry and the openai module
        modules_to_clear = ['brokle.integrations', 'brokle.integrations.openai']
        cleared_modules = {}

        for module_name in modules_to_clear:
            if module_name in sys.modules:
                cleared_modules[module_name] = sys.modules[module_name]
                del sys.modules[module_name]

        try:
            # Import fresh - this should trigger instrumentation with our mocks
            import brokle.integrations as integrations

            # Check that auto-instrumentation functions are available
            assert hasattr(integrations, 'openai_is_instrumented')
            assert hasattr(integrations, 'openai_get_status')
            assert hasattr(integrations, 'HAS_OPENAI')
            assert hasattr(integrations, 'HAS_WRAPT')
            assert hasattr(integrations, 'OPENAI_AUTO_AVAILABLE')

            # Check that the registry properly imported the auto-instrumentation module
            assert integrations.OPENAI_AUTO_AVAILABLE == True

            # P1: Keep the boolean assertion to guard against regressions
            # With fresh import and mocked dependencies, instrumentation should succeed
            assert integrations.openai_is_instrumented() == True

        finally:
            # Restore cleared modules to avoid affecting other tests
            for module_name, module_obj in cleared_modules.items():
                sys.modules[module_name] = module_obj

    @pytest.mark.parametrize("method_config", [
        {
            "module": "openai.resources.chat.completions",
            "object": "Completions.create",
            "name": "chat_completions_create"
        },
        {
            "module": "openai.resources.completions",
            "object": "Completions.create",
            "name": "completions_create"
        },
        {
            "module": "openai.resources.embeddings",
            "object": "Embeddings.create",
            "name": "embeddings_create"
        }
    ])
    def test_specific_method_instrumentation(self, method_config, mock_openai, mock_wrapt):
        """Test that specific OpenAI methods are properly instrumented."""
        import brokle.integrations.openai as openai_auto

        # Check that wrap_function_wrapper was called for this specific method
        calls = mock_wrapt.wrap_function_wrapper.call_args_list
        method_calls = [
            call for call in calls
            if call[0][0] == method_config["module"] and call[0][1] == method_config["object"]
        ]

        assert len(method_calls) > 0, f"Method {method_config['object']} in {method_config['module']} was not instrumented"


class TestAutoInstrumentationIntegration:
    """Test integration scenarios for auto-instrumentation."""

    def _clear_modules(self, *module_names: str) -> None:
        for module_name in module_names:
            if module_name in sys.modules:
                del sys.modules[module_name]

    def test_multiple_imports_are_safe(self):
        """Test that multiple imports of the same module are safe."""
        with patch.dict('sys.modules', {
            'openai': MagicMock(),
            'openai.resources': MagicMock(),
            'openai.resources.chat': MagicMock(),
            'openai.resources.chat.completions': MagicMock(),
            'openai.resources.completions': MagicMock(),
            'openai.resources.embeddings': MagicMock(),
            'wrapt': MagicMock()
        }):
            self._clear_modules('brokle.integrations', 'brokle.integrations.openai', 'brokle.openai')

            import brokle.integrations.openai as auto1
            import brokle.integrations.openai as auto2
            import brokle.openai as convenience

            assert auto1.is_instrumented() is True
            assert auto1.is_instrumented() == auto2.is_instrumented() == convenience.is_instrumented()

    def test_import_order_independence(self):
        """Test that import order doesn't matter."""
        with patch.dict('sys.modules', {
            'openai': MagicMock(),
            'openai.resources': MagicMock(),
            'openai.resources.chat': MagicMock(),
            'openai.resources.chat.completions': MagicMock(),
            'openai.resources.completions': MagicMock(),
            'openai.resources.embeddings': MagicMock(),
            'wrapt': MagicMock()
        }):
            self._clear_modules('brokle.integrations', 'brokle.integrations.openai', 'brokle.openai')

            import brokle.openai as convenience_first
            import brokle.integrations.openai as direct_second

            assert convenience_first.is_instrumented() == direct_second.is_instrumented() == True

    def test_graceful_degradation(self):
        """Test graceful degradation when dependencies are missing."""
        self._clear_modules('brokle.integrations', 'brokle.integrations.openai', 'brokle.openai', 'openai', 'wrapt')

        import builtins

        original_import = builtins.__import__

        def failing_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == 'wrapt':
                raise ImportError("wrapt missing")
            return original_import(name, globals, locals, fromlist, level)

        with patch('builtins.__import__', side_effect=failing_import):
            import brokle.integrations.openai as openai_auto

            assert openai_auto.HAS_WRAPT is False
            assert openai_auto.is_instrumented() is False

            status = openai_auto.get_instrumentation_status()
            assert status['instrumented'] is False
            assert status['wrapt_available'] is False

        self._clear_modules('brokle.integrations', 'brokle.integrations.openai', 'brokle.openai')
