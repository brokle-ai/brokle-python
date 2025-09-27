"""
Test retry logic with exponential backoff.
"""

import time
from unittest.mock import Mock, patch

import httpx
import pytest

from brokle._utils.retry import (
    RetryConfig,
    async_retry_with_backoff,
    extract_retry_after,
    is_retryable_error,
    retry_with_backoff,
)
from brokle.exceptions import RateLimitError


class TestRetryConfig:
    """Test retry configuration."""

    def test_default_config(self):
        """Test default retry configuration."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True

    def test_custom_config(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_retries=5,
            base_delay=0.5,
            max_delay=30.0,
            exponential_base=1.5,
            jitter=False,
        )
        assert config.max_retries == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 1.5
        assert config.jitter is False

    def test_calculate_delay(self):
        """Test delay calculation."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)

        # Test exponential backoff
        assert config.calculate_delay(1) == 1.0  # 1.0 * 2^0
        assert config.calculate_delay(2) == 2.0  # 1.0 * 2^1
        assert config.calculate_delay(3) == 4.0  # 1.0 * 2^2

    def test_calculate_delay_with_max(self):
        """Test delay calculation with max delay."""
        config = RetryConfig(
            base_delay=1.0, exponential_base=2.0, max_delay=3.0, jitter=False
        )

        assert config.calculate_delay(1) == 1.0
        assert config.calculate_delay(2) == 2.0
        assert config.calculate_delay(3) == 3.0  # Capped at max_delay
        assert config.calculate_delay(4) == 3.0  # Still capped

    def test_calculate_delay_with_jitter(self):
        """Test delay calculation with jitter."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=True)

        # Jitter should add variability (±25%)
        delay1 = config.calculate_delay(2)
        delay2 = config.calculate_delay(2)

        # Both should be around 2.0 but with some variance
        assert 1.5 <= delay1 <= 2.5
        assert 1.5 <= delay2 <= 2.5

        # Minimum delay should be 0.1
        assert delay1 >= 0.1
        assert delay2 >= 0.1


class TestRetryableErrors:
    """Test retryable error detection."""

    def test_network_errors_retryable(self):
        """Test that network errors are retryable."""
        assert is_retryable_error(httpx.ConnectError("Connection failed"))
        assert is_retryable_error(httpx.TimeoutException("Request timeout"))

    def test_server_errors_retryable(self):
        """Test that server errors are retryable."""
        response = Mock()
        response.status_code = 500
        error = httpx.HTTPStatusError("Server error", request=Mock(), response=response)
        assert is_retryable_error(error)

        response.status_code = 502
        assert is_retryable_error(error)

        response.status_code = 503
        assert is_retryable_error(error)

    def test_rate_limit_errors_retryable(self):
        """Test that rate limit errors are retryable."""
        response = Mock()
        response.status_code = 429
        error = httpx.HTTPStatusError("Rate limit", request=Mock(), response=response)
        assert is_retryable_error(error)

        rate_limit_error = RateLimitError("Rate limited")
        assert is_retryable_error(rate_limit_error)

    def test_client_errors_not_retryable(self):
        """Test that most client errors are not retryable."""
        response = Mock()
        response.status_code = 400
        error = httpx.HTTPStatusError("Bad request", request=Mock(), response=response)
        assert not is_retryable_error(error)

        response.status_code = 401
        assert not is_retryable_error(error)

        response.status_code = 404
        assert not is_retryable_error(error)

    def test_generic_errors_not_retryable(self):
        """Test that generic errors are not retryable."""
        assert not is_retryable_error(ValueError("Invalid value"))
        assert not is_retryable_error(TypeError("Type error"))


class TestRetryAfterExtraction:
    """Test Retry-After header extraction."""

    def test_extract_retry_after_header(self):
        """Test extracting Retry-After header."""
        response = Mock()
        response.headers = {"Retry-After": "5"}
        error = Mock()
        error.response = response

        retry_after = extract_retry_after(error)
        assert retry_after == 5.0

    def test_extract_retry_after_missing(self):
        """Test when Retry-After header is missing."""
        response = Mock()
        response.headers = {}
        error = Mock()
        error.response = response

        retry_after = extract_retry_after(error)
        assert retry_after is None

    def test_extract_retry_after_invalid(self):
        """Test when Retry-After header is invalid."""
        response = Mock()
        response.headers = {"Retry-After": "invalid"}
        error = Mock()
        error.response = response

        retry_after = extract_retry_after(error)
        assert retry_after is None

    def test_extract_retry_after_no_response(self):
        """Test when error has no response."""
        error = Mock()
        error.response = None

        retry_after = extract_retry_after(error)
        assert retry_after is None


class TestRetryDecorator:
    """Test retry decorator."""

    def test_no_retry_on_success(self):
        """Test that successful calls don't retry."""
        call_count = 0

        @retry_with_backoff(max_retries=3)
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_func()
        assert result == "success"
        assert call_count == 1

    def test_retry_on_retryable_error(self):
        """Test retrying on retryable errors."""
        call_count = 0

        @retry_with_backoff(max_retries=2, base_delay=0.1)
        def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.ConnectError("Connection failed")
            return "success"

        result = failing_then_success()
        assert result == "success"
        assert call_count == 3

    def test_no_retry_on_non_retryable_error(self):
        """Test not retrying on non-retryable errors."""
        call_count = 0

        @retry_with_backoff(max_retries=3)
        def non_retryable_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retryable")

        with pytest.raises(ValueError):
            non_retryable_error()

        assert call_count == 1

    def test_max_retries_exceeded(self):
        """Test behavior when max retries exceeded."""
        call_count = 0

        @retry_with_backoff(max_retries=2, base_delay=0.1)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise httpx.ConnectError("Always fails")

        with pytest.raises(httpx.ConnectError):
            always_fails()

        assert call_count == 3  # Initial call + 2 retries

    def test_delay_timing(self):
        """Test that delays are approximately correct."""
        call_times = []

        @retry_with_backoff(max_retries=2, base_delay=0.1, jitter=False)
        def timing_test():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise httpx.ConnectError("Connection failed")
            return "success"

        result = timing_test()
        assert result == "success"
        assert len(call_times) == 3

        # Check approximate delays (allowing some tolerance)
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]

        assert 0.08 <= delay1 <= 0.15  # ~0.1s ±50ms tolerance
        assert 0.18 <= delay2 <= 0.25  # ~0.2s ±50ms tolerance


@pytest.mark.asyncio
class TestAsyncRetryDecorator:
    """Test async retry decorator."""

    async def test_async_no_retry_on_success(self):
        """Test that successful async calls don't retry."""
        call_count = 0

        @async_retry_with_backoff(max_retries=3)
        async def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await successful_func()
        assert result == "success"
        assert call_count == 1

    async def test_async_retry_on_retryable_error(self):
        """Test retrying async calls on retryable errors."""
        call_count = 0

        @async_retry_with_backoff(max_retries=2, base_delay=0.1)
        async def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.ConnectError("Connection failed")
            return "success"

        result = await failing_then_success()
        assert result == "success"
        assert call_count == 3

    async def test_async_max_retries_exceeded(self):
        """Test async behavior when max retries exceeded."""
        call_count = 0

        @async_retry_with_backoff(max_retries=2, base_delay=0.1)
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise httpx.ConnectError("Always fails")

        with pytest.raises(httpx.ConnectError):
            await always_fails()

        assert call_count == 3  # Initial call + 2 retries


class TestIntegration:
    """Integration tests for retry logic."""

    @patch("time.sleep")  # Mock sleep to speed up tests
    def test_retry_with_rate_limiting(self, mock_sleep):
        """Test retry behavior with rate limiting."""
        call_count = 0

        @retry_with_backoff(max_retries=2, base_delay=1.0)
        def rate_limited_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                response = Mock()
                response.headers = {"Retry-After": "2"}
                response.status_code = 429
                error = httpx.HTTPStatusError(
                    "Rate limited", request=Mock(), response=response
                )
                error.response = response
                raise error
            return "success"

        result = rate_limited_func()
        assert result == "success"
        assert call_count == 3

        # Should have used Retry-After header value
        mock_sleep.assert_called()
        # Last call should use the Retry-After value (2 seconds)
        assert any(call.args[0] == 2.0 for call in mock_sleep.call_args_list)
