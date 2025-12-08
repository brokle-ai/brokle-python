"""
Edge case tests for input/output serialization.

Tests defensive programming for: None values, bytes, non-serializable objects,
circular references, Pydantic models, dataclasses.
"""

import json
from dataclasses import dataclass

import pytest

from brokle.client import _is_llm_messages_format, _serialize_with_mime


def test_serialize_none_value():
    """Test serialization of None value."""
    result, mime_type = _serialize_with_mime(None)
    assert result == "null"
    assert mime_type == "application/json"


def test_serialize_dict():
    """Test serialization of dict."""
    data = {"key": "value", "number": 42}
    result, mime_type = _serialize_with_mime(data)
    assert json.loads(result) == data
    assert mime_type == "application/json"


def test_serialize_list():
    """Test serialization of list."""
    data = ["item1", "item2", 123]
    result, mime_type = _serialize_with_mime(data)
    assert json.loads(result) == data
    assert mime_type == "application/json"


def test_serialize_string():
    """Test serialization of plain string."""
    result, mime_type = _serialize_with_mime("Hello world")
    assert result == "Hello world"
    assert mime_type == "text/plain"


def test_serialize_bytes_utf8():
    """Test serialization of bytes (UTF-8)."""
    result, mime_type = _serialize_with_mime(b"Hello bytes")
    assert result == "Hello bytes"
    assert mime_type == "text/plain"


def test_serialize_bytes_invalid_utf8():
    """Test serialization of invalid UTF-8 bytes."""
    invalid_bytes = b"\xff\xfe Invalid UTF-8"
    result, mime_type = _serialize_with_mime(invalid_bytes)
    # Should decode with error replacement
    assert isinstance(result, str)
    assert mime_type == "text/plain"


def test_serialize_pydantic_model():
    """Test serialization of Pydantic model."""
    try:
        from pydantic import BaseModel

        class User(BaseModel):
            name: str
            age: int

        user = User(name="John", age=30)
        result, mime_type = _serialize_with_mime(user)
        assert json.loads(result) == {"name": "John", "age": 30}
        assert mime_type == "application/json"
    except ImportError:
        pytest.skip("Pydantic not installed")


def test_serialize_dataclass():
    """Test serialization of dataclass."""

    @dataclass
    class Point:
        x: int
        y: int

    point = Point(x=10, y=20)
    result, mime_type = _serialize_with_mime(point)
    assert json.loads(result) == {"x": 10, "y": 20}
    assert mime_type == "application/json"


def test_serialize_non_serializable_object():
    """Test serialization of non-serializable custom object."""

    class CustomObject:
        def __init__(self):
            self.value = "test"

        def __str__(self):
            return f"CustomObject(value={self.value})"

    obj = CustomObject()
    result, mime_type = _serialize_with_mime(obj)
    # Should fallback to str()
    assert "CustomObject" in result
    assert mime_type == "text/plain"


def test_serialize_circular_reference():
    """Test serialization of dict with circular reference."""

    data = {"key": "value"}
    data["self"] = data  # Circular reference

    result, mime_type = _serialize_with_mime(data)
    # json.dumps with default=str should handle this
    # If it fails, should return error message
    assert isinstance(result, str)
    # Either succeeds or returns error message
    if not result.startswith("<serialization failed"):
        # If successful, should be valid JSON (with str() of circular ref)
        parsed = json.loads(result)
        assert "key" in parsed


def test_serialize_complex_nested_structure():
    """Test serialization of deeply nested structure."""
    data = {
        "level1": {
            "level2": {
                "level3": {
                    "values": [1, 2, 3],
                    "nested_list": [{"item": 1}, {"item": 2}],
                }
            }
        }
    }
    result, mime_type = _serialize_with_mime(data)
    assert json.loads(result) == data
    assert mime_type == "application/json"


def test_is_llm_messages_format_valid():
    """Test ChatML format detection - valid cases."""
    # Valid ChatML
    assert _is_llm_messages_format([{"role": "user", "content": "Hello"}])
    assert _is_llm_messages_format(
        [{"role": "system", "content": "System"}, {"role": "user", "content": "Hello"}]
    )
    assert _is_llm_messages_format(
        [{"role": "assistant", "content": "Hi", "tool_calls": []}]
    )


def test_is_llm_messages_format_invalid():
    """Test ChatML format detection - invalid cases."""
    # Not a list
    assert not _is_llm_messages_format({"role": "user", "content": "Hello"})

    # Empty list
    assert not _is_llm_messages_format([])

    # List but missing role
    assert not _is_llm_messages_format([{"content": "Hello"}])

    # List but not all dicts
    assert not _is_llm_messages_format(["string", "items"])

    # None
    assert not _is_llm_messages_format(None)

    # String
    assert not _is_llm_messages_format("not messages")


def test_serialize_with_special_characters():
    """Test serialization of strings with special characters."""
    special_str = 'Hello\nWorld\t"Quoted"\r\n'
    result, mime_type = _serialize_with_mime(special_str)
    assert result == special_str
    assert mime_type == "text/plain"


def test_serialize_with_unicode():
    """Test serialization of Unicode strings."""
    unicode_str = "Hello 世界 🌍"
    result, mime_type = _serialize_with_mime(unicode_str)
    assert result == unicode_str
    assert mime_type == "text/plain"


def test_serialize_empty_dict():
    """Test serialization of empty dict."""
    result, mime_type = _serialize_with_mime({})
    assert result == "{}"
    assert mime_type == "application/json"


def test_serialize_empty_list():
    """Test serialization of empty list."""
    result, mime_type = _serialize_with_mime([])
    assert result == "[]"
    assert mime_type == "application/json"
