"""
ULID utilities for event deduplication.

ULID (Universally Unique Lexicographically Sortable Identifier) provides:
- 128-bit identifier
- Lexicographically sortable
- Timestamp-based (first 48 bits)
- Cryptographically secure random (last 80 bits)
"""

import time
from typing import Optional

try:
    from ulid import ULID
except ImportError:
    # Fallback implementation if python-ulid not installed
    import uuid

    class ULID:
        """Fallback ULID implementation using UUID."""

        @staticmethod
        def from_timestamp(timestamp: float) -> str:
            """Generate ULID from timestamp (fallback to UUID)."""
            return str(uuid.uuid4()).replace("-", "").upper()[:26]


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
    if timestamp is None:
        timestamp = time.time()

    try:
        # Use python-ulid if available
        from ulid import from_timestamp
        return str(from_timestamp(timestamp))
    except ImportError:
        # Fallback to UUID-based implementation
        return ULID.from_timestamp(timestamp)


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
        from ulid import parse
        ulid_obj = parse(ulid_str)
        return ulid_obj.timestamp().timestamp()
    except (ImportError, ValueError):
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
