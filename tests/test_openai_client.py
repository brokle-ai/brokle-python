"""Tests for OpenAI drop-in replacement functionality."""

import pytest
from unittest.mock import patch, MagicMock

# Import brokle.openai (our drop-in replacement)
try:
    from brokle.openai import OpenAI
    HAS_BROKLE_OPENAI = True
except ImportError:
    HAS_BROKLE_OPENAI = False

# Mock OpenAI SDK types if not available
try:
    from openai.types.chat import ChatCompletion
    HAS_OPENAI_SDK = True
except ImportError:
    HAS_OPENAI_SDK = False
    class ChatCompletion:
        pass


@pytest.mark.skipif(not HAS_BROKLE_OPENAI, reason="brokle.openai not available")
class TestOpenAIDropInReplacement:
    """Test essential OpenAI drop-in replacement functionality."""

    def test_openai_import_and_creation(self):
        """Test OpenAI can be imported and created."""
        with patch.dict('os.environ', {'BROKLE_API_KEY': 'ak_test', 'BROKLE_PROJECT_ID': 'proj_test'}):
            client = OpenAI(api_key="sk-test-key")
            assert client is not None

    def test_openai_client_interface_compatibility(self):
        """Test that Brokle OpenAI client has same interface as original."""
        with patch.dict('os.environ', {'BROKLE_API_KEY': 'ak_test', 'BROKLE_PROJECT_ID': 'proj_test'}):
            client = OpenAI(api_key="sk-test-key")

            # Should have the same interface as real OpenAI client
            assert hasattr(client, 'chat')
            assert hasattr(client.chat, 'completions')
            assert hasattr(client.chat.completions, 'create')
            assert hasattr(client, 'completions')
            assert hasattr(client.completions, 'create')
            assert hasattr(client, 'embeddings')
            assert hasattr(client.embeddings, 'create')

    @patch('brokle.openai._original_openai')
    def test_chat_completions_passthrough(self, mock_openai):
        """Test that chat completions calls pass through to original OpenAI."""
        mock_response = {
            "id": "chatcmpl-test123",
            "object": "chat.completion",
            "model": "gpt-4",
            "choices": [{"message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        with patch('brokle.client.get_client') as mock_get_client:
            mock_brokle_client = MagicMock()
            mock_brokle_client.config.telemetry_enabled = True
            mock_get_client.return_value = mock_brokle_client

            with patch.dict('os.environ', {'BROKLE_API_KEY': 'ak_test', 'BROKLE_PROJECT_ID': 'proj_test'}):
                client = OpenAI(api_key="sk-test-key")

                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Hello"}]
                )

                assert response == mock_response
                mock_client.chat.completions.create.assert_called_once()

    @patch('brokle.openai._original_openai')
    def test_completions_passthrough(self, mock_openai):
        """Test that completions calls pass through to original OpenAI."""
        mock_response = {
            "id": "cmpl-test123",
            "object": "text_completion",
            "model": "gpt-3.5-turbo-instruct",
            "choices": [{"text": "Hello world!", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
        }

        mock_client = MagicMock()
        mock_client.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        with patch('brokle.client.get_client') as mock_get_client:
            mock_brokle_client = MagicMock()
            mock_brokle_client.config.telemetry_enabled = True
            mock_get_client.return_value = mock_brokle_client

            with patch.dict('os.environ', {'BROKLE_API_KEY': 'ak_test', 'BROKLE_PROJECT_ID': 'proj_test'}):
                client = OpenAI(api_key="sk-test-key")

                response = client.completions.create(
                    model="gpt-3.5-turbo-instruct",
                    prompt="Hello"
                )

                assert response == mock_response
                mock_client.completions.create.assert_called_once()

    @patch('brokle.openai._original_openai')
    def test_embeddings_passthrough(self, mock_openai):
        """Test that embeddings calls pass through to original OpenAI."""
        mock_response = {
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "text-embedding-ada-002",
            "usage": {"prompt_tokens": 3, "total_tokens": 3}
        }

        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        with patch('brokle.client.get_client') as mock_get_client:
            mock_brokle_client = MagicMock()
            mock_brokle_client.config.telemetry_enabled = True
            mock_get_client.return_value = mock_brokle_client

            with patch.dict('os.environ', {'BROKLE_API_KEY': 'ak_test', 'BROKLE_PROJECT_ID': 'proj_test'}):
                client = OpenAI(api_key="sk-test-key")

                response = client.embeddings.create(
                    model="text-embedding-ada-002",
                    input="Hello world"
                )

                assert response == mock_response
                mock_client.embeddings.create.assert_called_once()

    def test_instrumentation_disabled_without_brokle_client(self):
        """Test that instrumentation is disabled when Brokle client not available."""
        with patch('brokle.client.get_client') as mock_get_client:
            mock_get_client.side_effect = Exception("No client available")

            with patch('brokle.openai._original_openai') as mock_openai:
                mock_client = MagicMock()
                mock_client.chat.completions.create.return_value = {"id": "test"}
                mock_openai.OpenAI.return_value = mock_client

                with patch.dict('os.environ', {}, clear=True):
                    client = OpenAI(api_key="sk-test-key")
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": "Hello"}]
                    )

                    # Should still work, just without instrumentation
                    assert response == {"id": "test"}

    def test_telemetry_disabled_passthrough(self):
        """Test passthrough when telemetry is disabled."""
        with patch('brokle.client.get_client') as mock_get_client:
            mock_brokle_client = MagicMock()
            mock_brokle_client.config.telemetry_enabled = False
            mock_get_client.return_value = mock_brokle_client

            with patch('brokle.openai._original_openai') as mock_openai:
                mock_client = MagicMock()
                mock_client.chat.completions.create.return_value = {"id": "passthrough-test"}
                mock_openai.OpenAI.return_value = mock_client

                with patch.dict('os.environ', {'BROKLE_API_KEY': 'ak_test', 'BROKLE_PROJECT_ID': 'proj_test'}):
                    client = OpenAI(api_key="sk-test-key")
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": "Hello"}]
                    )

                    assert response == {"id": "passthrough-test"}
                    mock_client.chat.completions.create.assert_called()

    @patch('brokle.openai._original_openai')
    def test_error_propagation(self, mock_openai):
        """Test that OpenAI errors are properly propagated."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("OpenAI API Error")
        mock_openai.OpenAI.return_value = mock_client

        with patch('brokle.client.get_client') as mock_get_client:
            mock_brokle_client = MagicMock()
            mock_brokle_client.config.telemetry_enabled = True
            mock_get_client.return_value = mock_brokle_client

            with patch.dict('os.environ', {'BROKLE_API_KEY': 'ak_test', 'BROKLE_PROJECT_ID': 'proj_test'}):
                client = OpenAI(api_key="sk-test-key")

                with pytest.raises(Exception, match="OpenAI API Error"):
                    client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": "Hello"}]
                    )

    @patch('brokle.openai._original_openai')
    def test_parameter_preservation(self, mock_openai):
        """Test that all parameters are correctly preserved."""
        mock_response = {"id": "param-test"}
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        with patch('brokle.client.get_client') as mock_get_client:
            mock_brokle_client = MagicMock()
            mock_brokle_client.config.telemetry_enabled = True
            mock_get_client.return_value = mock_brokle_client

            with patch.dict('os.environ', {'BROKLE_API_KEY': 'ak_test', 'BROKLE_PROJECT_ID': 'proj_test'}):
                client = OpenAI(api_key="sk-test-key")

                client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Hello"}],
                    temperature=0.7,
                    max_tokens=100,
                    top_p=0.9,
                    stream=False
                )

                # Verify all parameters were forwarded
                call_args = mock_client.chat.completions.create.call_args
                assert call_args.kwargs["model"] == "gpt-4"
                assert call_args.kwargs["temperature"] == 0.7
                assert call_args.kwargs["max_tokens"] == 100
                assert call_args.kwargs["top_p"] == 0.9
                assert call_args.kwargs["stream"] is False

    @pytest.mark.asyncio
    async def test_async_client_basic_functionality(self):
        """Test async OpenAI client basic functionality."""
        try:
            from brokle.openai import AsyncOpenAI
            with patch.dict('os.environ', {'BROKLE_API_KEY': 'ak_test', 'BROKLE_PROJECT_ID': 'proj_test'}):
                async_client = AsyncOpenAI(api_key="sk-test-key")
                assert async_client is not None
                assert hasattr(async_client, 'chat')
                assert hasattr(async_client.chat, 'completions')
        except ImportError:
            pytest.skip("AsyncOpenAI not available")


class TestOpenAIStubMode:
    """Test OpenAI stub mode when SDK not available."""

    @patch('brokle.openai.HAS_OPENAI', False)
    def test_stub_openai_raises_helpful_error(self):
        """Test that stub OpenAI raises helpful error when SDK not available."""
        try:
            from brokle.openai import OpenAI
            with pytest.raises(Exception) as exc_info:
                client = OpenAI(api_key="sk-test")
                client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Hello"}]
                )

            error_msg = str(exc_info.value)
            assert "OpenAI SDK not installed" in error_msg or "pip install openai" in error_msg
        except ImportError:
            pytest.skip("brokle.openai not available")


class TestOpenAIInstrumentation:
    """Test OpenAI instrumentation functionality."""

    @patch('brokle.openai._original_openai')
    def test_span_creation_during_openai_call(self, mock_openai):
        """Test that spans are created during OpenAI calls."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = {"id": "test"}
        mock_openai.OpenAI.return_value = mock_client

        with patch('brokle.client.get_client') as mock_get_client:
            mock_brokle_client = MagicMock()
            mock_brokle_client.config.telemetry_enabled = True
            mock_brokle_client.generation.return_value.__enter__ = MagicMock()
            mock_brokle_client.generation.return_value.__exit__ = MagicMock()
            mock_get_client.return_value = mock_brokle_client

            with patch.dict('os.environ', {'BROKLE_API_KEY': 'ak_test', 'BROKLE_PROJECT_ID': 'proj_test'}):
                try:
                    from brokle.openai import OpenAI
                    client = OpenAI(api_key="sk-test")
                    client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": "Hello"}]
                    )

                    # Should have called generation to create span
                    mock_brokle_client.generation.assert_called()
                except ImportError:
                    pytest.skip("brokle.openai not available")