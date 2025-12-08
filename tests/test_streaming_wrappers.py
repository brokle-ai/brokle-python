"""
Test suite for streaming wrapper exception handling.

Tests that BrokleStreamWrapper and BrokleAsyncStreamWrapper properly handle
exceptions during stream iteration, ensuring spans are always closed with
appropriate status and error information.
"""

import time
from unittest.mock import MagicMock, Mock, patch

import pytest
from opentelemetry.trace import Status, StatusCode

from brokle.streaming.accumulator import StreamingAccumulator
from brokle.streaming.wrappers import BrokleAsyncStreamWrapper, BrokleStreamWrapper


class MockChunk:
    """Mock streaming chunk for testing."""

    def __init__(self, content: str = "test"):
        self.choices = [Mock()]
        self.choices[0].delta = Mock()
        self.choices[0].delta.content = content


class MockStreamSuccess:
    """Mock stream that yields chunks successfully then stops."""

    def __init__(self, chunks):
        self._chunks = iter(chunks)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._chunks)


class MockStreamError:
    """Mock stream that raises an error mid-stream."""

    def __init__(self, chunks_before_error, error):
        self._chunks = iter(chunks_before_error)
        self._error = error
        self._yielded_all = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            if not self._yielded_all:
                self._yielded_all = True
                raise self._error
            raise


class MockAsyncStreamSuccess:
    """Mock async stream that yields chunks successfully then stops."""

    def __init__(self, chunks):
        self._chunks = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration


class MockAsyncStreamError:
    """Mock async stream that raises an error mid-stream."""

    def __init__(self, chunks_before_error, error):
        self._chunks = iter(chunks_before_error)
        self._error = error
        self._yielded_all = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            if not self._yielded_all:
                self._yielded_all = True
                raise self._error
            raise StopAsyncIteration


class TestBrokleStreamWrapperNormalCompletion:
    """Test normal stream completion (StopIteration)."""

    def test_successful_iteration_completes_span(self):
        """Test that successful iteration completes span normally."""
        mock_span = MagicMock()
        chunks = [MockChunk("Hello"), MockChunk(" World")]
        mock_stream = MockStreamSuccess(chunks)

        accumulator = StreamingAccumulator(time.perf_counter())
        wrapper = BrokleStreamWrapper(mock_stream, mock_span, accumulator)

        # Iterate through stream
        collected = []
        for chunk in wrapper:
            collected.append(chunk)

        # Verify span was ended
        mock_span.end.assert_called_once()

        # Verify success status was set (consistent with non-streaming paths)
        mock_span.set_status.assert_called_once()
        status_call = mock_span.set_status.call_args[0][0]
        assert status_call.status_code == StatusCode.OK
        mock_span.record_exception.assert_not_called()

        # Verify wrapper is completed
        assert wrapper.is_completed is True
        assert wrapper.result is not None

    def test_empty_stream_completes_span(self):
        """Test that empty stream still completes span."""
        mock_span = MagicMock()
        mock_stream = MockStreamSuccess([])

        accumulator = StreamingAccumulator(time.perf_counter())
        wrapper = BrokleStreamWrapper(mock_stream, mock_span, accumulator)

        # Iterate (should immediately stop)
        collected = list(wrapper)

        assert collected == []
        mock_span.end.assert_called_once()
        assert wrapper.is_completed is True


class TestBrokleStreamWrapperErrorHandling:
    """Test error handling during stream iteration."""

    def test_runtime_error_records_error_status(self):
        """Test that RuntimeError during iteration records error status."""
        mock_span = MagicMock()
        chunks = [MockChunk("Hello")]
        error = RuntimeError("Provider API error")
        mock_stream = MockStreamError(chunks, error)

        accumulator = StreamingAccumulator(time.perf_counter())
        wrapper = BrokleStreamWrapper(mock_stream, mock_span, accumulator)

        # Iterate and expect error
        collected = []
        with pytest.raises(RuntimeError, match="Provider API error"):
            for chunk in wrapper:
                collected.append(chunk)

        # Verify we got the chunk before the error
        assert len(collected) == 1

        # Verify span was ended with error status
        mock_span.end.assert_called_once()
        mock_span.set_status.assert_called_once()
        status_call = mock_span.set_status.call_args[0][0]
        assert status_call.status_code == StatusCode.ERROR
        assert "Provider API error" in status_call.description

        # Verify exception was recorded
        mock_span.record_exception.assert_called_once_with(error)

        # Verify wrapper is completed
        assert wrapper.is_completed is True

    def test_connection_error_records_error_status(self):
        """Test that ConnectionError during iteration records error status."""
        mock_span = MagicMock()
        error = ConnectionError("Network connection lost")
        mock_stream = MockStreamError([], error)

        accumulator = StreamingAccumulator(time.perf_counter())
        wrapper = BrokleStreamWrapper(mock_stream, mock_span, accumulator)

        with pytest.raises(ConnectionError, match="Network connection lost"):
            list(wrapper)

        mock_span.end.assert_called_once()
        mock_span.set_status.assert_called_once()
        mock_span.record_exception.assert_called_once_with(error)

    def test_value_error_records_error_status(self):
        """Test that ValueError during iteration records error status."""
        mock_span = MagicMock()
        error = ValueError("Invalid response format")
        mock_stream = MockStreamError([MockChunk("a"), MockChunk("b")], error)

        accumulator = StreamingAccumulator(time.perf_counter())
        wrapper = BrokleStreamWrapper(mock_stream, mock_span, accumulator)

        collected = []
        with pytest.raises(ValueError, match="Invalid response format"):
            for chunk in wrapper:
                collected.append(chunk)

        # Should have collected chunks before error
        assert len(collected) == 2

        # Span should be closed with error
        mock_span.end.assert_called_once()
        mock_span.set_status.assert_called_once()

    def test_error_does_not_complete_twice(self):
        """Test that error handling is idempotent."""
        mock_span = MagicMock()
        error = RuntimeError("Test error")
        mock_stream = MockStreamError([], error)

        accumulator = StreamingAccumulator(time.perf_counter())
        wrapper = BrokleStreamWrapper(mock_stream, mock_span, accumulator)

        # First iteration triggers error
        with pytest.raises(RuntimeError):
            list(wrapper)

        # Manually try to complete again (should be no-op)
        wrapper._complete_instrumentation()
        wrapper._complete_instrumentation_with_error(RuntimeError("Another"))

        # Span should only be ended once
        assert mock_span.end.call_count == 1


class TestBrokleStreamWrapperContextManager:
    """Test context manager behavior with errors."""

    def test_context_manager_with_error(self):
        """Test that context manager properly handles errors."""
        mock_span = MagicMock()
        chunks = [MockChunk("Hello")]
        error = RuntimeError("Error during iteration")
        mock_stream = MockStreamError(chunks, error)

        accumulator = StreamingAccumulator(time.perf_counter())
        wrapper = BrokleStreamWrapper(mock_stream, mock_span, accumulator)

        with pytest.raises(RuntimeError):
            with wrapper as stream:
                for chunk in stream:
                    pass

        # Span should be ended
        mock_span.end.assert_called()


class TestBrokleAsyncStreamWrapperNormalCompletion:
    """Test normal async stream completion (StopAsyncIteration)."""

    @pytest.mark.asyncio
    async def test_successful_iteration_completes_span(self):
        """Test that successful async iteration completes span normally."""
        mock_span = MagicMock()
        chunks = [MockChunk("Hello"), MockChunk(" World")]
        mock_stream = MockAsyncStreamSuccess(chunks)

        accumulator = StreamingAccumulator(time.perf_counter())
        wrapper = BrokleAsyncStreamWrapper(mock_stream, mock_span, accumulator)

        # Iterate through stream
        collected = []
        async for chunk in wrapper:
            collected.append(chunk)

        # Verify span was ended
        mock_span.end.assert_called_once()

        # Verify success status was set (consistent with non-streaming paths)
        mock_span.set_status.assert_called_once()
        status_call = mock_span.set_status.call_args[0][0]
        assert status_call.status_code == StatusCode.OK
        mock_span.record_exception.assert_not_called()

        # Verify wrapper is completed
        assert wrapper.is_completed is True
        assert wrapper.result is not None

    @pytest.mark.asyncio
    async def test_empty_stream_completes_span(self):
        """Test that empty async stream still completes span."""
        mock_span = MagicMock()
        mock_stream = MockAsyncStreamSuccess([])

        accumulator = StreamingAccumulator(time.perf_counter())
        wrapper = BrokleAsyncStreamWrapper(mock_stream, mock_span, accumulator)

        # Iterate (should immediately stop)
        collected = []
        async for chunk in wrapper:
            collected.append(chunk)

        assert collected == []
        mock_span.end.assert_called_once()
        assert wrapper.is_completed is True


class TestBrokleAsyncStreamWrapperErrorHandling:
    """Test error handling during async stream iteration."""

    @pytest.mark.asyncio
    async def test_runtime_error_records_error_status(self):
        """Test that RuntimeError during async iteration records error status."""
        mock_span = MagicMock()
        chunks = [MockChunk("Hello")]
        error = RuntimeError("Provider API error")
        mock_stream = MockAsyncStreamError(chunks, error)

        accumulator = StreamingAccumulator(time.perf_counter())
        wrapper = BrokleAsyncStreamWrapper(mock_stream, mock_span, accumulator)

        # Iterate and expect error
        collected = []
        with pytest.raises(RuntimeError, match="Provider API error"):
            async for chunk in wrapper:
                collected.append(chunk)

        # Verify we got the chunk before the error
        assert len(collected) == 1

        # Verify span was ended with error status
        mock_span.end.assert_called_once()
        mock_span.set_status.assert_called_once()
        status_call = mock_span.set_status.call_args[0][0]
        assert status_call.status_code == StatusCode.ERROR
        assert "Provider API error" in status_call.description

        # Verify exception was recorded
        mock_span.record_exception.assert_called_once_with(error)

        # Verify wrapper is completed
        assert wrapper.is_completed is True

    @pytest.mark.asyncio
    async def test_connection_error_records_error_status(self):
        """Test that ConnectionError during async iteration records error status."""
        mock_span = MagicMock()
        error = ConnectionError("Network connection lost")
        mock_stream = MockAsyncStreamError([], error)

        accumulator = StreamingAccumulator(time.perf_counter())
        wrapper = BrokleAsyncStreamWrapper(mock_stream, mock_span, accumulator)

        with pytest.raises(ConnectionError, match="Network connection lost"):
            async for _ in wrapper:
                pass

        mock_span.end.assert_called_once()
        mock_span.set_status.assert_called_once()
        mock_span.record_exception.assert_called_once_with(error)

    @pytest.mark.asyncio
    async def test_error_does_not_complete_twice(self):
        """Test that async error handling is idempotent."""
        mock_span = MagicMock()
        error = RuntimeError("Test error")
        mock_stream = MockAsyncStreamError([], error)

        accumulator = StreamingAccumulator(time.perf_counter())
        wrapper = BrokleAsyncStreamWrapper(mock_stream, mock_span, accumulator)

        # First iteration triggers error
        with pytest.raises(RuntimeError):
            async for _ in wrapper:
                pass

        # Manually try to complete again (should be no-op)
        wrapper._complete_instrumentation()
        wrapper._complete_instrumentation_with_error(RuntimeError("Another"))

        # Span should only be ended once
        assert mock_span.end.call_count == 1


class MockAsyncStreamErrorWithContextManager:
    """Mock async stream with context manager that raises an error mid-stream."""

    def __init__(self, chunks_before_error, error):
        self._chunks = iter(chunks_before_error)
        self._error = error
        self._yielded_all = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            if not self._yielded_all:
                self._yielded_all = True
                raise self._error
            raise StopAsyncIteration

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False


class TestBrokleAsyncStreamWrapperContextManager:
    """Test async context manager behavior with errors."""

    @pytest.mark.asyncio
    async def test_context_manager_with_error(self):
        """Test that async context manager properly handles errors."""
        mock_span = MagicMock()
        chunks = [MockChunk("Hello")]
        error = RuntimeError("Error during iteration")
        mock_stream = MockAsyncStreamErrorWithContextManager(chunks, error)

        accumulator = StreamingAccumulator(time.perf_counter())
        wrapper = BrokleAsyncStreamWrapper(mock_stream, mock_span, accumulator)

        with pytest.raises(RuntimeError):
            async with wrapper as stream:
                async for chunk in stream:
                    pass

        # Span should be ended
        mock_span.end.assert_called()


class TestPartialDataCapture:
    """Test that partial streaming data is captured on errors."""

    def test_sync_partial_chunks_captured(self):
        """Test that chunks received before error are captured in result."""
        mock_span = MagicMock()
        chunks = [MockChunk("Hello"), MockChunk(" "), MockChunk("World")]
        error = RuntimeError("Mid-stream error")
        mock_stream = MockStreamError(chunks, error)

        accumulator = StreamingAccumulator(time.perf_counter())
        wrapper = BrokleStreamWrapper(mock_stream, mock_span, accumulator)

        collected = []
        with pytest.raises(RuntimeError):
            for chunk in wrapper:
                collected.append(chunk)

        # All chunks before error should be collected
        assert len(collected) == 3

        # Result should exist with partial data
        assert wrapper.result is not None

        # Span should have attributes set (from partial data)
        assert mock_span.set_attribute.called

    @pytest.mark.asyncio
    async def test_async_partial_chunks_captured(self):
        """Test that async chunks received before error are captured in result."""
        mock_span = MagicMock()
        chunks = [MockChunk("Async"), MockChunk(" test")]
        error = RuntimeError("Mid-stream error")
        mock_stream = MockAsyncStreamError(chunks, error)

        accumulator = StreamingAccumulator(time.perf_counter())
        wrapper = BrokleAsyncStreamWrapper(mock_stream, mock_span, accumulator)

        collected = []
        with pytest.raises(RuntimeError):
            async for chunk in wrapper:
                collected.append(chunk)

        # All chunks before error should be collected
        assert len(collected) == 2

        # Result should exist with partial data
        assert wrapper.result is not None


class TestExceptionTypePreservation:
    """Test that original exception types are preserved."""

    def test_sync_preserves_exception_type(self):
        """Test that sync wrapper preserves original exception type."""
        mock_span = MagicMock()

        # Test various exception types
        for exc_type in [RuntimeError, ValueError, ConnectionError, TimeoutError]:
            mock_stream = MockStreamError([], exc_type("test"))
            accumulator = StreamingAccumulator(time.perf_counter())
            wrapper = BrokleStreamWrapper(mock_stream, mock_span, accumulator)

            with pytest.raises(exc_type):
                list(wrapper)

    @pytest.mark.asyncio
    async def test_async_preserves_exception_type(self):
        """Test that async wrapper preserves original exception type."""
        mock_span = MagicMock()

        # Test various exception types
        for exc_type in [RuntimeError, ValueError, ConnectionError, TimeoutError]:
            mock_stream = MockAsyncStreamError([], exc_type("test"))
            accumulator = StreamingAccumulator(time.perf_counter())
            wrapper = BrokleAsyncStreamWrapper(mock_stream, mock_span, accumulator)

            with pytest.raises(exc_type):
                async for _ in wrapper:
                    pass


class TestBrokleStreamWrapperBaseExceptionHandling:
    """Test BaseException handling during sync stream iteration."""

    def test_generator_exit_completes_span_normally(self):
        """Test that GeneratorExit completes span with success status (not error)."""
        mock_span = MagicMock()

        class GeneratorExitStream:
            def __iter__(self):
                return self

            def __next__(self):
                raise GeneratorExit()

        accumulator = StreamingAccumulator(time.perf_counter())
        wrapper = BrokleStreamWrapper(GeneratorExitStream(), mock_span, accumulator)

        with pytest.raises(GeneratorExit):
            next(wrapper)

        # Span should be ended with success status (early termination is OK, not error)
        mock_span.end.assert_called_once()
        mock_span.set_status.assert_called_once()
        status_call = mock_span.set_status.call_args[0][0]
        assert status_call.status_code == StatusCode.OK  # Success, not ERROR
        mock_span.record_exception.assert_not_called()
        assert wrapper.is_completed is True

    def test_keyboard_interrupt_records_error_and_completes(self):
        """Test that KeyboardInterrupt records error but still completes span."""
        mock_span = MagicMock()

        class KeyboardInterruptStream:
            def __iter__(self):
                return self

            def __next__(self):
                raise KeyboardInterrupt()

        accumulator = StreamingAccumulator(time.perf_counter())
        wrapper = BrokleStreamWrapper(KeyboardInterruptStream(), mock_span, accumulator)

        with pytest.raises(KeyboardInterrupt):
            next(wrapper)

        # Span should be ended WITH error status
        mock_span.end.assert_called_once()
        mock_span.set_status.assert_called_once()
        mock_span.record_exception.assert_called_once()
        assert wrapper.is_completed is True

    def test_system_exit_records_error_and_completes(self):
        """Test that SystemExit records error but still completes span."""
        mock_span = MagicMock()

        class SystemExitStream:
            def __iter__(self):
                return self

            def __next__(self):
                raise SystemExit(1)

        accumulator = StreamingAccumulator(time.perf_counter())
        wrapper = BrokleStreamWrapper(SystemExitStream(), mock_span, accumulator)

        with pytest.raises(SystemExit):
            next(wrapper)

        # Span should be ended WITH error status
        mock_span.end.assert_called_once()
        mock_span.set_status.assert_called_once()
        assert wrapper.is_completed is True

    def test_generator_exit_mid_stream_completes_normally(self):
        """Test GeneratorExit after some chunks still completes without error."""
        mock_span = MagicMock()

        class MidStreamGeneratorExitStream:
            def __init__(self):
                self._count = 0

            def __iter__(self):
                return self

            def __next__(self):
                self._count += 1
                if self._count > 2:
                    raise GeneratorExit()
                return MockChunk(f"chunk{self._count}")

        accumulator = StreamingAccumulator(time.perf_counter())
        wrapper = BrokleStreamWrapper(
            MidStreamGeneratorExitStream(), mock_span, accumulator
        )

        collected = []
        with pytest.raises(GeneratorExit):
            for chunk in wrapper:
                collected.append(chunk)

        # Should have collected chunks before GeneratorExit
        assert len(collected) == 2
        # Span should be ended with success status (early termination is OK)
        mock_span.end.assert_called_once()
        mock_span.set_status.assert_called_once()
        status_call = mock_span.set_status.call_args[0][0]
        assert status_call.status_code == StatusCode.OK


class TestBrokleAsyncStreamWrapperBaseExceptionHandling:
    """Test BaseException handling during async stream iteration."""

    @pytest.mark.asyncio
    async def test_cancelled_error_completes_span_normally(self):
        """Test that CancelledError completes span with success status (not error)."""
        import asyncio

        mock_span = MagicMock()

        class CancelledStream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise asyncio.CancelledError()

        accumulator = StreamingAccumulator(time.perf_counter())
        wrapper = BrokleAsyncStreamWrapper(CancelledStream(), mock_span, accumulator)

        with pytest.raises(asyncio.CancelledError):
            await wrapper.__anext__()

        # Span should be ended with success status (cancellation is OK, not error)
        mock_span.end.assert_called_once()
        mock_span.set_status.assert_called_once()
        status_call = mock_span.set_status.call_args[0][0]
        assert status_call.status_code == StatusCode.OK  # Success, not ERROR
        mock_span.record_exception.assert_not_called()
        assert wrapper.is_completed is True

    @pytest.mark.asyncio
    async def test_keyboard_interrupt_records_error_and_completes(self):
        """Test that KeyboardInterrupt records error but still completes span."""
        mock_span = MagicMock()

        class KeyboardInterruptStream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise KeyboardInterrupt()

        accumulator = StreamingAccumulator(time.perf_counter())
        wrapper = BrokleAsyncStreamWrapper(
            KeyboardInterruptStream(), mock_span, accumulator
        )

        with pytest.raises(KeyboardInterrupt):
            await wrapper.__anext__()

        # Span should be ended WITH error status
        mock_span.end.assert_called_once()
        mock_span.set_status.assert_called_once()
        mock_span.record_exception.assert_called_once()
        assert wrapper.is_completed is True

    @pytest.mark.asyncio
    async def test_system_exit_records_error_and_completes(self):
        """Test that SystemExit records error but still completes span."""
        mock_span = MagicMock()

        class SystemExitStream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise SystemExit(1)

        accumulator = StreamingAccumulator(time.perf_counter())
        wrapper = BrokleAsyncStreamWrapper(SystemExitStream(), mock_span, accumulator)

        with pytest.raises(SystemExit):
            await wrapper.__anext__()

        # Span should be ended WITH error status
        mock_span.end.assert_called_once()
        mock_span.set_status.assert_called_once()
        assert wrapper.is_completed is True

    @pytest.mark.asyncio
    async def test_cancelled_error_mid_stream_completes_normally(self):
        """Test CancelledError after some chunks still completes without error."""
        import asyncio

        mock_span = MagicMock()

        class MidStreamCancelledStream:
            def __init__(self):
                self._count = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                self._count += 1
                if self._count > 2:
                    raise asyncio.CancelledError()
                return MockChunk(f"chunk{self._count}")

        accumulator = StreamingAccumulator(time.perf_counter())
        wrapper = BrokleAsyncStreamWrapper(
            MidStreamCancelledStream(), mock_span, accumulator
        )

        collected = []
        with pytest.raises(asyncio.CancelledError):
            async for chunk in wrapper:
                collected.append(chunk)

        # Should have collected chunks before CancelledError
        assert len(collected) == 2
        # Span should be ended with success status (CancelledError is non-error)
        mock_span.end.assert_called_once()
        # Verify success status was set (consistent with non-streaming paths)
        mock_span.set_status.assert_called_once()
        status_call = mock_span.set_status.call_args[0][0]
        assert status_call.status_code == StatusCode.OK
        mock_span.record_exception.assert_not_called()
