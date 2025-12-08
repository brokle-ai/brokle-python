"""
Transparent stream wrappers for automatic instrumentation.

These wrappers implement the iterator protocol and delegate to the
underlying stream while collecting metrics via StreamingAccumulator.
Users iterate streams normally - instrumentation is completely automatic.

Example:
    # Users iterate normally - no manual accumulator needed
    for chunk in wrapped_stream:
        print(chunk.choices[0].delta.content)
    # Metrics recorded automatically when loop ends

Based on patterns from:
- OpenLLMetry: ObjectProxy wrapper pattern
- OpenLIT: Timestamp collection for comprehensive analytics
- Langfuse: Generator wrapper with auto-finalization
"""

import asyncio
from typing import TYPE_CHECKING, Any, Optional

from opentelemetry.trace import Status, StatusCode
from wrapt import ObjectProxy

from .accumulator import StreamingAccumulator, StreamingResult

if TYPE_CHECKING:
    from opentelemetry.trace import Span


class BrokleStreamWrapper(ObjectProxy):
    """
    Transparent wrapper for synchronous streaming responses.

    Implements iterator protocol, accumulates content, and records
    metrics when stream completes. All original stream methods and
    attributes are transparently accessible.

    Example:
        # Users iterate normally - instrumentation is automatic
        for chunk in wrapped_stream:
            print(chunk.choices[0].delta.content)
        # Metrics recorded automatically when loop ends

        # Context manager also works
        with wrapped_stream as stream:
            for chunk in stream:
                print(chunk.choices[0].delta.content)
    """

    def __init__(
        self,
        response: Any,
        span: "Span",
        accumulator: StreamingAccumulator,
    ):
        """
        Initialize stream wrapper.

        Args:
            response: Original streaming response from LLM provider
            span: OpenTelemetry span to record attributes on
            accumulator: Pre-configured accumulator with start_time
        """
        super().__init__(response)
        self._self_span = span
        self._self_accumulator = accumulator
        self._self_completed = False
        self._self_result: Optional[StreamingResult] = None

    def __iter__(self):
        """Return iterator (self)."""
        return self

    def __next__(self):
        """Get next chunk from stream and process through accumulator."""
        try:
            chunk = self.__wrapped__.__next__()
        except BaseException as e:
            if isinstance(e, StopIteration):
                self._complete_instrumentation()
            elif isinstance(e, GeneratorExit):
                self._complete_instrumentation()
            else:
                self._complete_instrumentation_with_error(e)
            raise
        else:
            self._self_accumulator.on_chunk(chunk)
            return chunk

    def __enter__(self):
        """Context manager entry."""
        if hasattr(self.__wrapped__, "__enter__"):
            self.__wrapped__.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._complete_instrumentation()
        if hasattr(self.__wrapped__, "__exit__"):
            return self.__wrapped__.__exit__(exc_type, exc_val, exc_tb)
        return False

    def __del__(self):
        """Destructor - attempt to complete instrumentation if not done."""
        if not self._self_completed:
            try:
                self._complete_instrumentation()
            except Exception:
                # Ignore errors during cleanup
                pass

    def _complete_instrumentation(self):
        """Finalize accumulator and record span attributes."""
        if self._self_completed:
            return
        self._self_completed = True

        self._self_result = self._self_accumulator.finalize()

        # Set span attributes from streaming result
        attrs = self._self_result.to_attributes()
        for key, value in attrs.items():
            if value is not None:
                self._self_span.set_attribute(key, value)

        # Mark span as successful (consistent with non-streaming paths)
        self._self_span.set_status(Status(StatusCode.OK))

        # End the span (it was started without a context manager)
        self._self_span.end()

    def _complete_instrumentation_with_error(self, exception: BaseException):
        """Finalize accumulator with error status for failed streams."""
        if self._self_completed:
            return
        self._self_completed = True

        self._self_result = self._self_accumulator.finalize()

        attrs = self._self_result.to_attributes()
        for key, value in attrs.items():
            if value is not None:
                self._self_span.set_attribute(key, value)

        self._self_span.set_status(Status(StatusCode.ERROR, str(exception)))
        self._self_span.record_exception(exception)
        self._self_span.end()

    @property
    def result(self) -> Optional[StreamingResult]:
        """
        Access streaming result after iteration completes.

        Returns:
            StreamingResult with metrics and accumulated content,
            or None if stream hasn't completed yet
        """
        return self._self_result

    @property
    def is_completed(self) -> bool:
        """Check if stream has completed and instrumentation finalized."""
        return self._self_completed


class BrokleAsyncStreamWrapper(ObjectProxy):
    """
    Transparent wrapper for asynchronous streaming responses.

    Same as BrokleStreamWrapper but implements async iterator protocol.
    All original async stream methods and attributes are transparently accessible.

    Example:
        # Users iterate normally with async for
        async for chunk in wrapped_stream:
            print(chunk.choices[0].delta.content)
        # Metrics recorded automatically when loop ends

        # Async context manager also works
        async with wrapped_stream as stream:
            async for chunk in stream:
                print(chunk.choices[0].delta.content)
    """

    def __init__(
        self,
        response: Any,
        span: "Span",
        accumulator: StreamingAccumulator,
    ):
        """
        Initialize async stream wrapper.

        Args:
            response: Original async streaming response from LLM provider
            span: OpenTelemetry span to record attributes on
            accumulator: Pre-configured accumulator with start_time
        """
        super().__init__(response)
        self._self_span = span
        self._self_accumulator = accumulator
        self._self_completed = False
        self._self_result: Optional[StreamingResult] = None

    def __aiter__(self):
        """Return async iterator (self)."""
        return self

    async def __anext__(self):
        """Get next chunk from async stream and process through accumulator."""
        try:
            chunk = await self.__wrapped__.__anext__()
        except BaseException as e:
            if isinstance(e, StopAsyncIteration):
                self._complete_instrumentation()
            elif isinstance(e, asyncio.CancelledError):
                self._complete_instrumentation()
            else:
                self._complete_instrumentation_with_error(e)
            raise
        else:
            self._self_accumulator.on_chunk(chunk)
            return chunk

    async def __aenter__(self):
        """Async context manager entry."""
        if hasattr(self.__wrapped__, "__aenter__"):
            await self.__wrapped__.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self._complete_instrumentation()
        if hasattr(self.__wrapped__, "__aexit__"):
            return await self.__wrapped__.__aexit__(exc_type, exc_val, exc_tb)
        return False

    def __del__(self):
        """Destructor - attempt to complete instrumentation if not done."""
        if not self._self_completed:
            try:
                self._complete_instrumentation()
            except Exception:
                # Ignore errors during cleanup
                pass

    def _complete_instrumentation(self):
        """Finalize accumulator and record span attributes."""
        if self._self_completed:
            return
        self._self_completed = True

        self._self_result = self._self_accumulator.finalize()

        # Set span attributes from streaming result
        attrs = self._self_result.to_attributes()
        for key, value in attrs.items():
            if value is not None:
                self._self_span.set_attribute(key, value)

        # Mark span as successful (consistent with non-streaming paths)
        self._self_span.set_status(Status(StatusCode.OK))

        # End the span (it was started without a context manager)
        self._self_span.end()

    def _complete_instrumentation_with_error(self, exception: BaseException):
        """Finalize accumulator with error status for failed streams."""
        if self._self_completed:
            return
        self._self_completed = True

        self._self_result = self._self_accumulator.finalize()

        attrs = self._self_result.to_attributes()
        for key, value in attrs.items():
            if value is not None:
                self._self_span.set_attribute(key, value)

        self._self_span.set_status(Status(StatusCode.ERROR, str(exception)))
        self._self_span.record_exception(exception)
        self._self_span.end()

    @property
    def result(self) -> Optional[StreamingResult]:
        """
        Access streaming result after iteration completes.

        Returns:
            StreamingResult with metrics and accumulated content,
            or None if stream hasn't completed yet
        """
        return self._self_result

    @property
    def is_completed(self) -> bool:
        """Check if stream has completed and instrumentation finalized."""
        return self._self_completed
