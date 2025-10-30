"""
Validation utilities for Brokle SDK.

Simple validation functions for common use cases.
"""

import re
from typing import Any


def validate_environment(environment: str) -> bool:
    """
    Validate environment tag format.

    Args:
        environment: Environment tag to validate

    Returns:
        True if valid

    Raises:
        ValueError: If environment tag is invalid
    """
    if not environment:
        raise ValueError("Environment tag cannot be empty")

    if len(environment) > 40:
        raise ValueError("Environment tag cannot exceed 40 characters")

    if not re.match(r"^[a-z0-9_-]+$", environment):
        raise ValueError(
            "Environment tag must contain only lowercase letters, numbers, hyphens, and underscores"
        )

    if environment.startswith("brokle"):
        raise ValueError("Environment tag cannot start with 'brokle' prefix")

    return True


def validate_api_key(api_key: str) -> bool:
    """
    Validate API key format.

    Args:
        api_key: API key to validate

    Returns:
        True if valid

    Raises:
        ValueError: If API key is invalid
    """
    if not api_key:
        raise ValueError("API key cannot be empty")

    if not api_key.startswith("bk_"):
        raise ValueError("API key must start with 'bk_' prefix")

    if len(api_key) < 10:
        raise ValueError("API key is too short")

    return True
