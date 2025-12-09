"""
Robust JSON serializer for arbitrary Python objects.

Follows the Langfuse EventSerializer pattern for comprehensive type handling,
including Pydantic models, dataclasses, numpy arrays, circular references, and more.
"""

import enum
import math
from asyncio import Queue
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from json import JSONEncoder
from logging import getLogger
from pathlib import Path
from typing import Any, Optional, Set
from uuid import UUID

logger = getLogger(__name__)

try:
    from pydantic import BaseModel
    HAS_PYDANTIC = True
except ImportError:
    BaseModel = None  # type: ignore
    HAS_PYDANTIC = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False


def serialize_datetime(dt: datetime) -> str:
    """Serialize datetime to ISO 8601 format with timezone awareness."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


class EventSerializer(JSONEncoder):
    """
    Robust JSON serializer for arbitrary Python objects.

    Handles common types and provides fallback for complex objects.
    Based on Langfuse's EventSerializer pattern.

    Features:
    - Pydantic models → model_dump()
    - Dataclasses → asdict()
    - NumPy arrays → tolist()
    - Datetime → ISO 8601 format
    - Enums → .value
    - Circular reference detection
    - Large integer handling (JS safe range)
    - Exception serialization

    Example:
        >>> import json
        >>> from brokle.utils.serializer import EventSerializer
        >>>
        >>> data = {"model": MyPydanticModel(...), "time": datetime.now()}
        >>> json.dumps(data, cls=EventSerializer)
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.seen: Set[int] = set()

    def default(self, obj: Any) -> Any:
        """Convert non-JSON-serializable objects to serializable format."""
        try:
            if isinstance(obj, datetime):
                return serialize_datetime(obj)

            if isinstance(obj, date):
                return obj.isoformat()

            if HAS_NUMPY and np is not None:
                if isinstance(obj, np.generic):
                    return obj.item()
                if isinstance(obj, np.ndarray):
                    return obj.tolist()

            if isinstance(obj, float):
                if math.isnan(obj):
                    return "NaN"
                if math.isinf(obj):
                    return "Infinity" if obj > 0 else "-Infinity"

            if isinstance(obj, (Exception, KeyboardInterrupt)):
                return f"{type(obj).__name__}: {str(obj)}"

            if "Streaming" in type(obj).__name__:
                return str(obj)

            if isinstance(obj, enum.Enum):
                return obj.value

            if isinstance(obj, Queue):
                return f"<{type(obj).__name__}>"

            if is_dataclass(obj) and not isinstance(obj, type):
                return asdict(obj)

            if isinstance(obj, UUID):
                return str(obj)

            if isinstance(obj, bytes):
                try:
                    return obj.decode("utf-8")
                except UnicodeDecodeError:
                    return "<not serializable bytes>"

            if HAS_PYDANTIC and BaseModel is not None and isinstance(obj, BaseModel):
                try:
                    return obj.model_dump()
                except AttributeError:
                    return obj.dict()

            if isinstance(obj, Path):
                return str(obj)

            if isinstance(obj, int):
                return obj if self._is_js_safe_integer(obj) else str(obj)

            if isinstance(obj, (str, float, type(None))):
                return obj

            if isinstance(obj, (tuple, set, frozenset)):
                return list(obj)

            if isinstance(obj, dict):
                return {self.default(k): self.default(v) for k, v in obj.items()}

            if isinstance(obj, list):
                return [self.default(item) for item in obj]

            if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
                return [self.default(item) for item in obj]

            if hasattr(obj, "__slots__"):
                return self.default(
                    {slot: getattr(obj, slot, None) for slot in obj.__slots__}
                )

            if hasattr(obj, "__dict__"):
                obj_id = id(obj)
                if obj_id in self.seen:
                    return f"<{type(obj).__name__}>"

                self.seen.add(obj_id)
                try:
                    result = {k: self.default(v) for k, v in vars(obj).items()}
                finally:
                    self.seen.discard(obj_id)

                return result

            return f"<{type(obj).__name__}>"

        except Exception as e:
            logger.debug(
                f"Serialization failed for object of type {type(obj).__name__}: {e}"
            )
            return f"<not serializable: {type(obj).__name__}>"

    def encode(self, obj: Any) -> str:
        """Encode object to JSON string with error handling."""
        self.seen.clear()
        try:
            return super().encode(self.default(obj))
        except Exception:
            return f'"<not serializable: {type(obj).__name__}>"'

    @staticmethod
    def _is_js_safe_integer(value: int) -> bool:
        """
        Check if integer is within JavaScript's safe range.

        JavaScript can only safely represent integers in the range
        -(2^53 - 1) to 2^53 - 1.

        https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Number/MAX_SAFE_INTEGER
        """
        max_safe_int = 2**53 - 1
        min_safe_int = -(2**53) + 1
        return min_safe_int <= value <= max_safe_int


def serialize(obj: Any) -> Optional[str]:
    """
    Serialize an object to JSON string.

    Convenience function that uses EventSerializer.

    Args:
        obj: Object to serialize

    Returns:
        JSON string or None if object is None

    Example:
        >>> serialize({"key": "value"})
        '{"key": "value"}'
        >>> serialize(None)
        None
    """
    if obj is None:
        return None

    if isinstance(obj, str):
        return obj

    try:
        import json

        return json.dumps(obj, cls=EventSerializer)
    except Exception as e:
        logger.debug(f"Serialization failed: {e}")
        return f"<serialization failed: {type(obj).__name__}>"


def serialize_value(value: Any) -> Any:
    """
    Recursively serialize a value for JSON encoding.

    This is the core serialization function used by the @observe decorator
    to convert function arguments and return values to JSON-serializable format.

    Args:
        value: Value to serialize

    Returns:
        JSON-serializable value

    Example:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Person:
        ...     name: str
        ...     age: int
        >>> serialize_value(Person("Alice", 30))
        {'name': 'Alice', 'age': 30}
    """
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, int) and not EventSerializer._is_js_safe_integer(value):
            return str(value)
        if isinstance(value, float):
            if math.isnan(value):
                return "NaN"
            if math.isinf(value):
                return "Infinity" if value > 0 else "-Infinity"
        return value

    if isinstance(value, datetime):
        return serialize_datetime(value)

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, UUID):
        return str(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return "<not serializable bytes>"

    if isinstance(value, enum.Enum):
        return value.value

    if HAS_NUMPY and np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()

    if isinstance(value, (list, tuple)):
        return [serialize_value(item) for item in value]

    if isinstance(value, (set, frozenset)):
        return [serialize_value(item) for item in value]

    if isinstance(value, dict):
        return {str(k): serialize_value(v) for k, v in value.items()}

    if HAS_PYDANTIC and BaseModel is not None and isinstance(value, BaseModel):
        try:
            return value.model_dump()
        except AttributeError:
            return value.dict()

    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)

    if isinstance(value, Exception):
        return f"{type(value).__name__}: {str(value)}"

    return str(value)


def serialize_function_args(
    func_args: tuple,
    func_kwargs: dict,
    param_names: Optional[list] = None,
    is_method: bool = False,
) -> dict:
    """
    Serialize function arguments to a dictionary.

    Args:
        func_args: Positional arguments tuple
        func_kwargs: Keyword arguments dict
        param_names: Optional list of parameter names (from inspect.signature)
        is_method: If True, skip first arg (self/cls)

    Returns:
        Dictionary with serialized arguments

    Example:
        >>> def greet(name, greeting="Hello"):
        ...     pass
        >>> serialize_function_args(("World",), {"greeting": "Hi"}, ["name", "greeting"])
        {'name': 'World', 'greeting': 'Hi'}
    """
    args = func_args[1:] if is_method else func_args
    result = {}

    if param_names:
        for i, arg in enumerate(args):
            if i < len(param_names):
                result[param_names[i]] = serialize_value(arg)
            else:
                result[f"arg_{i}"] = serialize_value(arg)
    else:
        for i, arg in enumerate(args):
            result[f"arg_{i}"] = serialize_value(arg)

    for key, value in func_kwargs.items():
        result[key] = serialize_value(value)

    return result
