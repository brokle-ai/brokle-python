"""
Serialization utilities for Brokle SDK.
"""

import json
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel


def serialize(obj: Any) -> Optional[str]:
    """Serialize object to JSON string for OpenTelemetry attributes."""
    if obj is None:
        return None

    # Handle simple types directly
    if isinstance(obj, (str, int, float, bool)):
        return obj

    try:
        return json.dumps(obj, default=_json_serializer)
    except Exception:
        return str(obj)


def deserialize(data: Optional[str]) -> Any:
    """Deserialize JSON string back to object."""
    if not data:
        return None

    if not isinstance(data, str):
        return data

    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return data


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for complex objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, BaseModel):
        return obj.model_dump()
    elif hasattr(obj, "__dict__"):
        return obj.__dict__
    else:
        return str(obj)


def flatten_dict(
    data: Dict[str, Any], prefix: str = "", separator: str = "."
) -> Dict[str, Any]:
    """Flatten nested dictionary for OpenTelemetry attributes."""
    result = {}

    for key, value in data.items():
        new_key = f"{prefix}{separator}{key}" if prefix else key

        if isinstance(value, dict):
            result.update(flatten_dict(value, new_key, separator))
        elif isinstance(value, (list, tuple)):
            result[new_key] = serialize(value)
        else:
            result[new_key] = value

    return result
