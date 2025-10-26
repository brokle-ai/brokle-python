"""
ULID utilities for event deduplication.

ULID (Universally Unique Lexicographically Sortable Identifier) provides:
- 128-bit identifier
- Lexicographically sortable
- Timestamp-based (first 48 bits)
- Cryptographically secure random (last 80 bits)
"""

from typing import Optional

from ulid import ULID


def generate_ulid(timestamp: Optional[float] = None) -> str:
    """
    Generate a ULID string.

    Args:
        timestamp: Optional timestamp (defaults to current time)

    Returns:
        26-character ULID string (uppercase)

    Example:
        >>> ulid = generate_ulid()
        >>> len(ulid)
        26
        >>> ulid = generate_ulid(timestamp=1677610602.123)
        >>> ulid[:10]  # First 10 chars encode timestamp
    """
    # ULID objects must be converted to strings for Pydantic compatibility
    if timestamp is None:
        return str(ULID())  # Use ULID() constructor for current time
    return str(ULID.from_timestamp(timestamp))  # Custom timestamp when needed


def extract_timestamp(ulid_str: str) -> Optional[float]:
    """
    Extract timestamp from ULID string.

    Args:
        ulid_str: ULID string

    Returns:
        Unix timestamp or None if invalid

    Example:
        >>> ulid = generate_ulid(timestamp=1677610602.0)
        >>> ts = extract_timestamp(ulid)
        >>> abs(ts - 1677610602.0) < 1  # Within 1 second
        True
    """
    try:
        ulid_obj = ULID.parse(ulid_str)
        return ulid_obj.timestamp
    except ValueError:
        return None


def is_valid_ulid(ulid_str: str) -> bool:
    """
    Validate ULID string format.

    Args:
        ulid_str: String to validate

    Returns:
        True if valid ULID format

    Example:
        >>> is_valid_ulid("01ARZ3NDEKTSV4RRFFQ69G5FAV")
        True
        >>> is_valid_ulid("invalid")
        False
    """
    if not isinstance(ulid_str, str):
        return False

    if len(ulid_str) != 26:
        return False

    # ULID uses Crockford's base32 encoding
    valid_chars = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    return all(c in valid_chars for c in ulid_str.upper())


def generate_event_id() -> str:
    """
    Generate a unique event ID using ULID.

    This is the primary function for generating event IDs in the batch telemetry system.

    Returns:
        26-character ULID string

    Example:
        >>> event_id = generate_event_id()
        >>> len(event_id)
        26
        >>> is_valid_ulid(event_id)
        True
    """
    return generate_ulid()


__all__ = [
    "generate_ulid",
    "generate_event_id",
    "extract_timestamp",
    "is_valid_ulid",
]
