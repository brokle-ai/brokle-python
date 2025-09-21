"""
Miscellaneous utilities for Brokle SDK.

Simple miscellaneous utilities extracted and cleaned from the old integration framework.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def safe_json_serialize(obj: Any, max_depth: int = 10) -> str:
    """
    Safely serialize an object to JSON, handling complex types.

    Args:
        obj: Object to serialize
        max_depth: Maximum recursion depth

    Returns:
        JSON string representation
    """
    def _serialize_helper(item, depth=0):
        if depth > max_depth:
            return "[MAX_DEPTH_EXCEEDED]"

        if isinstance(item, (str, int, float, bool)) or item is None:
            return item
        elif isinstance(item, dict):
            return {k: _serialize_helper(v, depth + 1) for k, v in item.items()}
        elif isinstance(item, (list, tuple)):
            return [_serialize_helper(i, depth + 1) for i in item]
        else:
            return str(item)

    try:
        serializable = _serialize_helper(obj)
        return json.dumps(serializable, ensure_ascii=False, separators=(',', ':'))
    except Exception as e:
        logger.debug(f"Failed to serialize object: {e}")
        return json.dumps({"serialization_error": str(e)})


def truncate_string(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """
    Truncate a string to a maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated

    Returns:
        Truncated string
    """
    if not text or len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix


def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.

    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)

    Returns:
        Merged dictionary
    """
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


def get_timestamp_ms() -> int:
    """
    Get current timestamp in milliseconds.

    Returns:
        Current timestamp in milliseconds
    """
    return int(time.time() * 1000)


def sanitize_metadata(metadata: Dict[str, Any], max_items: int = 50) -> Dict[str, Any]:
    """
    Sanitize metadata by limiting size and removing sensitive data.

    Args:
        metadata: Metadata dictionary to sanitize
        max_items: Maximum number of items to keep

    Returns:
        Sanitized metadata
    """
    if not isinstance(metadata, dict):
        return {}

    # Remove sensitive keys
    sensitive_keys = {
        'password', 'secret', 'token', 'key', 'api_key',
        'auth', 'authorization', 'credentials'
    }

    sanitized = {}
    for k, v in metadata.items():
        # Skip sensitive keys
        if any(sensitive in k.lower() for sensitive in sensitive_keys):
            continue

        # Limit string length
        if isinstance(v, str):
            v = truncate_string(v, 500)
        elif isinstance(v, dict):
            v = sanitize_metadata(v, max_items // 2)
        elif isinstance(v, (list, tuple)):
            v = v[:10]  # Limit list size

        sanitized[k] = v

        # Limit total items
        if len(sanitized) >= max_items:
            break

    return sanitized


def extract_error_details(exc: Exception) -> Dict[str, Any]:
    """
    Extract useful details from an exception.

    Args:
        exc: Exception to analyze

    Returns:
        Dictionary with error details
    """
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "module": getattr(type(exc), '__module__', 'unknown'),
        "args": getattr(exc, 'args', [])[:3]  # Limit args
    }


def normalize_provider_name(provider: str) -> str:
    """
    Normalize provider name to standard format.

    Args:
        provider: Provider name to normalize

    Returns:
        Normalized provider name
    """
    if not provider:
        return "unknown"

    return provider.lower().strip().replace(' ', '_').replace('-', '_')


def calculate_rate_limit_delay(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True
) -> float:
    """
    Calculate delay for rate limit retry with exponential backoff.

    Args:
        attempt: Attempt number (0-based)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        jitter: Whether to add random jitter

    Returns:
        Delay in seconds
    """
    import random

    delay = min(base_delay * (2 ** attempt), max_delay)

    if jitter:
        delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter

    return delay


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of specified size.

    Args:
        items: List to chunk
        chunk_size: Size of each chunk

    Returns:
        List of chunked lists
    """
    if chunk_size <= 0:
        return [items]

    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]