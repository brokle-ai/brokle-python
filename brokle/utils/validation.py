"""
Validation utilities for Brokle SDK configuration.

Provides validation functions for API keys, environment names, and other
configuration values.
"""

import re
from typing import Optional


def validate_api_key(api_key: str) -> bool:
    """
    Validate Brokle API key format.

    API keys must:
    - Start with 'bk_'
    - Be at least 10 characters long
    - Contain only alphanumeric characters after prefix

    Args:
        api_key: API key to validate

    Returns:
        True if valid

    Raises:
        ValueError: If API key is invalid

    Example:
        >>> validate_api_key("bk_abc123def456")
        True
        >>> validate_api_key("invalid")  # doctest: +SKIP
        Traceback (most recent call last):
        ValueError: API key must start with 'bk_'
    """
    if not api_key:
        raise ValueError("API key cannot be empty")

    if not api_key.startswith("bk_"):
        raise ValueError("API key must start with 'bk_'")

    if len(api_key) < 10:
        raise ValueError("API key is too short (minimum 10 characters)")

    # Check if the part after 'bk_' contains only alphanumeric characters
    key_body = api_key[3:]
    if not key_body.replace("_", "").isalnum():
        raise ValueError("API key contains invalid characters")

    return True


def validate_environment(environment: str) -> bool:
    """
    Validate environment tag format.

    Environment tags must:
    - Be 1-40 characters long
    - Contain only alphanumeric characters, hyphens, and underscores
    - Not start or end with hyphen or underscore

    Args:
        environment: Environment tag to validate

    Returns:
        True if valid

    Raises:
        ValueError: If environment tag is invalid

    Example:
        >>> validate_environment("production")
        True
        >>> validate_environment("dev-env_1")
        True
        >>> validate_environment("")  # doctest: +SKIP
        Traceback (most recent call last):
        ValueError: Environment cannot be empty
    """
    if not environment:
        raise ValueError("Environment cannot be empty")

    if len(environment) > 40:
        raise ValueError("Environment must be 40 characters or less")

    # Check valid characters
    if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*[a-zA-Z0-9]$", environment):
        # Handle single character case
        if len(environment) == 1 and environment.isalnum():
            return True
        raise ValueError(
            "Environment must contain only alphanumeric characters, hyphens, "
            "and underscores, and cannot start or end with hyphen or underscore"
        )

    return True


def validate_sample_rate(sample_rate: float) -> bool:
    """
    Validate sampling rate.

    Args:
        sample_rate: Sampling rate (0.0 to 1.0)

    Returns:
        True if valid

    Raises:
        ValueError: If sample rate is invalid

    Example:
        >>> validate_sample_rate(0.5)
        True
        >>> validate_sample_rate(1.5)  # doctest: +SKIP
        Traceback (most recent call last):
        ValueError: Sample rate must be between 0.0 and 1.0
    """
    if not isinstance(sample_rate, (int, float)):
        raise ValueError("Sample rate must be a number")

    if not 0.0 <= sample_rate <= 1.0:
        raise ValueError("Sample rate must be between 0.0 and 1.0")

    return True


def validate_flush_config(
    flush_at: Optional[int] = None,
    flush_interval: Optional[float] = None,
) -> bool:
    """
    Validate flush configuration.

    Args:
        flush_at: Maximum batch size before flush (1-1000)
        flush_interval: Maximum delay before flush in seconds (0.1-60.0)

    Returns:
        True if valid

    Raises:
        ValueError: If flush configuration is invalid

    Example:
        >>> validate_flush_config(flush_at=100, flush_interval=5.0)
        True
    """
    if flush_at is not None:
        if not isinstance(flush_at, int):
            raise ValueError("flush_at must be an integer")
        if not 1 <= flush_at <= 1000:
            raise ValueError("flush_at must be between 1 and 1000")

    if flush_interval is not None:
        if not isinstance(flush_interval, (int, float)):
            raise ValueError("flush_interval must be a number")
        if not 0.1 <= flush_interval <= 60.0:
            raise ValueError("flush_interval must be between 0.1 and 60.0 seconds")

    return True


def validate_base_url(base_url: str) -> bool:
    """
    Validate Brokle API base URL.

    Args:
        base_url: Base URL to validate

    Returns:
        True if valid

    Raises:
        ValueError: If base URL is invalid

    Example:
        >>> validate_base_url("https://api.brokle.ai")
        True
        >>> validate_base_url("http://localhost:8080")
        True
        >>> validate_base_url("invalid")  # doctest: +SKIP
        Traceback (most recent call last):
        ValueError: Base URL must start with http:// or https://
    """
    if not base_url:
        raise ValueError("Base URL cannot be empty")

    if not base_url.startswith(("http://", "https://")):
        raise ValueError("Base URL must start with http:// or https://")

    # Basic URL format validation
    if not re.match(r"^https?://[a-zA-Z0-9][a-zA-Z0-9.-]+(:[0-9]+)?(/.*)?$", base_url):
        raise ValueError("Base URL format is invalid")

    return True
