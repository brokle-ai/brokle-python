"""
Tests for data masking functionality.

Comprehensive test suite for PII masking in the Brokle SDK, covering:
- Core masking functionality
- Error handling and fallbacks
- Performance characteristics
- Integration with span processing
"""

import logging
import re
import time
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
from opentelemetry.sdk.trace import ReadableSpan

from brokle.config import BrokleConfig
from brokle.processor import MASKABLE_ATTRIBUTES, BrokleSpanProcessor
from brokle.types.attributes import BrokleOtelSpanAttributes as Attrs


class TestCoreMasking:
    """Test core masking functionality."""

    def test_masking_disabled_by_default(self):
        """Verify no masking when not configured."""
        config = BrokleConfig(api_key="bk_test_key_12345")
        assert config.mask is None

    def test_simple_string_masking(self):
        """Test basic string replacement."""

        def simple_mask(data):
            if isinstance(data, str):
                return data.replace("secret", "***")
            return data

        config = BrokleConfig(api_key="bk_test_key_12345", mask=simple_mask)
        exporter = Mock()
        processor = BrokleSpanProcessor(exporter, config)

        # Create mock span with sensitive input
        span = Mock(spec=ReadableSpan)
        span.attributes = {
            Attrs.INPUT_VALUE: "This is a secret message",
            Attrs.OUTPUT_VALUE: "No confidential data here",
            Attrs.GEN_AI_REQUEST_MODEL: "gpt-4",  # Should not be masked
        }
        span._attributes = span.attributes

        processor._apply_masking(span)

        # Verify masking was applied to INPUT_VALUE
        assert span._attributes[Attrs.INPUT_VALUE] == "This is a *** message"
        # Verify OUTPUT_VALUE unchanged (no "secret" in it)
        assert span._attributes[Attrs.OUTPUT_VALUE] == "No confidential data here"
        # Verify non-maskable attribute unchanged
        assert span._attributes[Attrs.GEN_AI_REQUEST_MODEL] == "gpt-4"

    def test_nested_dict_masking(self):
        """Test masking in nested dictionaries."""

        def nested_mask(data):
            if isinstance(data, dict):
                return {k: nested_mask(v) for k, v in data.items()}
            elif isinstance(data, str):
                return data.replace("@example.com", "@[MASKED]")
            return data

        config = BrokleConfig(api_key="bk_test_key_12345", mask=nested_mask)
        exporter = Mock()
        processor = BrokleSpanProcessor(exporter, config)

        span = Mock(spec=ReadableSpan)
        span.attributes = {
            Attrs.METADATA: {
                "user": {"email": "john@example.com", "name": "John"},
                "admin": {"email": "admin@example.com"},
            }
        }
        span._attributes = span.attributes

        processor._apply_masking(span)

        result = span._attributes[Attrs.METADATA]
        assert result["user"]["email"] == "john@[MASKED]"
        assert result["admin"]["email"] == "admin@[MASKED]"
        assert result["user"]["name"] == "John"  # Unchanged

    def test_list_masking(self):
        """Test masking in lists."""

        def list_mask(data):
            if isinstance(data, list):
                return [list_mask(item) for item in data]
            elif isinstance(data, str) and "sensitive" in data:
                return "[REDACTED]"
            return data

        config = BrokleConfig(api_key="bk_test_key_12345", mask=list_mask)
        exporter = Mock()
        processor = BrokleSpanProcessor(exporter, config)

        span = Mock(spec=ReadableSpan)
        span.attributes = {
            Attrs.OUTPUT_VALUE: ["normal", "sensitive data", "also normal"]
        }
        span._attributes = span.attributes

        processor._apply_masking(span)

        result = span._attributes[Attrs.OUTPUT_VALUE]
        assert result == ["normal", "[REDACTED]", "also normal"]

    def test_masking_preserves_structure(self):
        """Verify data structure is preserved after masking."""

        def structure_mask(data):
            if isinstance(data, dict):
                return {k: structure_mask(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [structure_mask(item) for item in data]
            elif isinstance(data, str):
                return data.upper()
            return data

        config = BrokleConfig(api_key="bk_test_key_12345", mask=structure_mask)
        exporter = Mock()
        processor = BrokleSpanProcessor(exporter, config)

        span = Mock(spec=ReadableSpan)
        complex_structure = {
            "nested": {"deep": ["value1", "value2"], "count": 42},
            "list": [1, 2, {"key": "value"}],
        }
        span.attributes = {Attrs.METADATA: complex_structure}
        span._attributes = span.attributes

        processor._apply_masking(span)

        result = span._attributes[Attrs.METADATA]
        # Structure preserved, strings uppercased
        assert isinstance(result, dict)
        assert isinstance(result["nested"], dict)
        assert isinstance(result["nested"]["deep"], list)
        assert result["nested"]["deep"] == ["VALUE1", "VALUE2"]
        assert result["nested"]["count"] == 42  # Non-string unchanged

    def test_masking_only_applies_to_maskable_attributes(self):
        """Verify masking only affects MASKABLE_ATTRIBUTES."""

        def mask_all(data):
            return "MASKED"

        config = BrokleConfig(api_key="bk_test_key_12345", mask=mask_all)
        exporter = Mock()
        processor = BrokleSpanProcessor(exporter, config)

        span = Mock(spec=ReadableSpan)
        span.attributes = {
            # Maskable
            Attrs.INPUT_VALUE: "input",
            Attrs.OUTPUT_VALUE: "output",
            Attrs.GEN_AI_INPUT_MESSAGES: "messages",
            Attrs.METADATA: "metadata",
            # Non-maskable
            Attrs.GEN_AI_REQUEST_MODEL: "gpt-4",
            Attrs.SESSION_ID: "session-123",
            Attrs.GEN_AI_USAGE_INPUT_TOKENS: 100,
        }
        span._attributes = span.attributes

        processor._apply_masking(span)

        # Maskable attributes masked
        assert span._attributes[Attrs.INPUT_VALUE] == "MASKED"
        assert span._attributes[Attrs.OUTPUT_VALUE] == "MASKED"
        assert span._attributes[Attrs.GEN_AI_INPUT_MESSAGES] == "MASKED"
        assert span._attributes[Attrs.METADATA] == "MASKED"

        # Non-maskable attributes unchanged
        assert span._attributes[Attrs.GEN_AI_REQUEST_MODEL] == "gpt-4"
        assert span._attributes[Attrs.SESSION_ID] == "session-123"
        assert span._attributes[Attrs.GEN_AI_USAGE_INPUT_TOKENS] == 100


class TestErrorHandling:
    """Test masking error scenarios."""

    def test_masking_exception_full_mask(self):
        """Verify full masking on exception."""

        def broken_mask(data):
            raise ValueError("Intentional error for testing")

        config = BrokleConfig(api_key="bk_test_key_12345", mask=broken_mask)
        exporter = Mock()
        processor = BrokleSpanProcessor(exporter, config)

        span = Mock(spec=ReadableSpan)
        span.attributes = {
            Attrs.INPUT_VALUE: "This should be fully masked due to error"
        }
        span._attributes = span.attributes

        processor._apply_masking(span)

        # Verify full masking fallback
        assert (
            span._attributes[Attrs.INPUT_VALUE]
            == "<fully masked due to failed mask function>"
        )

    def test_masking_error_logged(self, caplog):
        """Verify errors are logged."""

        def broken_mask(data):
            raise ValueError("Test error")

        config = BrokleConfig(api_key="bk_test_key_12345", mask=broken_mask)
        exporter = Mock()
        processor = BrokleSpanProcessor(exporter, config)

        span = Mock(spec=ReadableSpan)
        span.attributes = {Attrs.INPUT_VALUE: "test"}
        span._attributes = span.attributes

        with caplog.at_level(logging.ERROR):
            processor._apply_masking(span)

        # Verify error was logged
        assert "Masking failed" in caplog.text
        assert "ValueError" in caplog.text

    def test_masking_error_does_not_crash_processor(self):
        """Verify processor continues even if masking fails."""

        def broken_mask(data):
            raise RuntimeError("Crash test")

        config = BrokleConfig(api_key="bk_test_key_12345", mask=broken_mask)
        exporter = Mock()
        processor = BrokleSpanProcessor(exporter, config)

        span = Mock(spec=ReadableSpan)
        span.attributes = {Attrs.INPUT_VALUE: "test", Attrs.OUTPUT_VALUE: "test2"}
        span._attributes = span.attributes

        # Should not raise exception
        processor._apply_masking(span)

        # Both attributes should be fully masked (not crash)
        assert (
            span._attributes[Attrs.INPUT_VALUE]
            == "<fully masked due to failed mask function>"
        )
        assert (
            span._attributes[Attrs.OUTPUT_VALUE]
            == "<fully masked due to failed mask function>"
        )

    def test_partial_masking_failure(self):
        """Test that some attributes can be masked even if others fail."""

        def partial_mask(data):
            if isinstance(data, dict):
                raise ValueError("Cannot mask dicts")
            if isinstance(data, str):
                return data.replace("sensitive", "***")
            return data

        config = BrokleConfig(api_key="bk_test_key_12345", mask=partial_mask)
        exporter = Mock()
        processor = BrokleSpanProcessor(exporter, config)

        span = Mock(spec=ReadableSpan)
        span.attributes = {
            Attrs.INPUT_VALUE: "This is sensitive",
            Attrs.METADATA: {"key": "value"},  # Will fail
        }
        span._attributes = span.attributes

        processor._apply_masking(span)

        # String successfully masked
        assert span._attributes[Attrs.INPUT_VALUE] == "This is ***"
        # Dict masked with fallback
        assert (
            span._attributes[Attrs.METADATA]
            == "<fully masked due to failed mask function>"
        )

    def test_none_attributes_handling(self):
        """Test handling of None attributes."""
        config = BrokleConfig(api_key="bk_test_key_12345", mask=lambda x: "masked")
        exporter = Mock()
        processor = BrokleSpanProcessor(exporter, config)

        span = Mock(spec=ReadableSpan)
        span._attributes = None

        # Should not raise exception
        processor._apply_masking(span)

    def test_empty_attributes_handling(self):
        """Test handling of empty attributes dict."""
        config = BrokleConfig(api_key="bk_test_key_12345", mask=lambda x: "masked")
        exporter = Mock()
        processor = BrokleSpanProcessor(exporter, config)

        span = Mock(spec=ReadableSpan)
        span._attributes = {}

        # Should not raise exception
        processor._apply_masking(span)


class TestMaskingWithRealPatterns:
    """Test with realistic PII patterns."""

    def test_email_masking(self):
        """Test email pattern masking."""

        def mask_emails(data):
            if isinstance(data, str):
                return re.sub(
                    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                    "[EMAIL]",
                    data,
                )
            return data

        config = BrokleConfig(api_key="bk_test_key_12345", mask=mask_emails)
        exporter = Mock()
        processor = BrokleSpanProcessor(exporter, config)

        span = Mock(spec=ReadableSpan)
        span.attributes = {
            Attrs.INPUT_VALUE: "Contact john@example.com or admin@company.org"
        }
        span._attributes = span.attributes

        processor._apply_masking(span)

        assert span._attributes[Attrs.INPUT_VALUE] == "Contact [EMAIL] or [EMAIL]"

    def test_phone_masking(self):
        """Test phone number masking."""

        def mask_phones(data):
            if isinstance(data, str):
                return re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]", data)
            return data

        config = BrokleConfig(api_key="bk_test_key_12345", mask=mask_phones)
        exporter = Mock()
        processor = BrokleSpanProcessor(exporter, config)

        span = Mock(spec=ReadableSpan)
        span.attributes = {
            Attrs.OUTPUT_VALUE: "Call 555-123-4567 or 555.987.6543"
        }
        span._attributes = span.attributes

        processor._apply_masking(span)

        assert span._attributes[Attrs.OUTPUT_VALUE] == "Call [PHONE] or [PHONE]"

    def test_api_key_masking(self):
        """Test API key masking."""

        def mask_api_keys(data):
            if isinstance(data, str):
                # Pattern: (sk|pk|bk)_ followed by 20+ alphanumeric/underscore chars
                return re.sub(r"(sk|pk|bk)_[a-zA-Z0-9_]{20,}", "[API_KEY]", data)
            elif isinstance(data, dict):
                return {k: mask_api_keys(v) for k, v in data.items()}
            return data

        config = BrokleConfig(api_key="bk_test_key_12345", mask=mask_api_keys)
        exporter = Mock()
        processor = BrokleSpanProcessor(exporter, config)

        span = Mock(spec=ReadableSpan)
        span.attributes = {
            Attrs.INPUT_VALUE: "Using key: sk_test_51234567890123456789012345678901234"
        }
        span._attributes = span.attributes

        processor._apply_masking(span)

        assert span._attributes[Attrs.INPUT_VALUE] == "Using key: [API_KEY]"

    def test_field_based_masking(self):
        """Test field name-based masking."""

        def mask_sensitive_fields(data):
            if isinstance(data, dict):
                result = {}
                for k, v in data.items():
                    if k in ["password", "ssn", "credit_card"]:
                        result[k] = "***MASKED***"
                    elif isinstance(v, dict):
                        result[k] = mask_sensitive_fields(v)
                    else:
                        result[k] = v
                return result
            return data

        config = BrokleConfig(api_key="bk_test_key_12345", mask=mask_sensitive_fields)
        exporter = Mock()
        processor = BrokleSpanProcessor(exporter, config)

        span = Mock(spec=ReadableSpan)
        span.attributes = {
            Attrs.METADATA: {
                "user": "john",
                "password": "secret123",
                "ssn": "123-45-6789",
                "data": {"credit_card": "1234-5678-9012-3456"},
            }
        }
        span._attributes = span.attributes

        processor._apply_masking(span)

        result = span._attributes[Attrs.METADATA]
        assert result["user"] == "john"
        assert result["password"] == "***MASKED***"
        assert result["ssn"] == "***MASKED***"
        assert result["data"]["credit_card"] == "***MASKED***"


class TestPerformance:
    """Performance benchmarks for masking."""

    def test_masking_overhead_simple(self):
        """Verify <1ms overhead for simple masking."""

        def simple_mask(data):
            if isinstance(data, str):
                return data.replace("test", "***")
            return data

        config = BrokleConfig(api_key="bk_test_key_12345", mask=simple_mask)
        exporter = Mock()
        processor = BrokleSpanProcessor(exporter, config)

        span = Mock(spec=ReadableSpan)
        span.attributes = {
            Attrs.INPUT_VALUE: "This is a test message",
            Attrs.OUTPUT_VALUE: "Another test",
        }
        span._attributes = span.attributes

        # Measure time
        start = time.perf_counter()
        for _ in range(1000):  # Run 1000 times
            processor._apply_masking(span)
        duration = time.perf_counter() - start

        avg_time_ms = (duration / 1000) * 1000  # Convert to ms
        assert avg_time_ms < 1.0, f"Average masking time {avg_time_ms:.3f}ms exceeds 1ms target"

    def test_masking_overhead_complex(self):
        """Verify reasonable overhead for complex masking."""

        def complex_mask(data):
            if isinstance(data, dict):
                return {k: complex_mask(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [complex_mask(item) for item in data]
            elif isinstance(data, str):
                # Multiple regex operations
                data = re.sub(r"\b[\w.]+@[\w.]+\b", "[EMAIL]", data)
                data = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", data)
                return data
            return data

        config = BrokleConfig(api_key="bk_test_key_12345", mask=complex_mask)
        exporter = Mock()
        processor = BrokleSpanProcessor(exporter, config)

        span = Mock(spec=ReadableSpan)
        span.attributes = {
            Attrs.METADATA: {
                "user": {"email": "john@example.com", "ssn": "123-45-6789"},
                "data": ["value1", "value2", "value3"],
            }
        }
        span._attributes = span.attributes

        # Measure time
        start = time.perf_counter()
        for _ in range(100):  # Run 100 times (more expensive operation)
            processor._apply_masking(span)
        duration = time.perf_counter() - start

        avg_time_ms = (duration / 100) * 1000
        assert avg_time_ms < 5.0, f"Average complex masking time {avg_time_ms:.3f}ms exceeds 5ms"

    def test_disabled_masking_zero_overhead(self):
        """Verify zero overhead when masking is disabled."""
        config = BrokleConfig(api_key="bk_test_key_12345")  # No mask
        exporter = Mock()
        processor = BrokleSpanProcessor(exporter, config)

        span = Mock(spec=ReadableSpan)
        span.attributes = {
            Attrs.INPUT_VALUE: "test",
            Attrs.OUTPUT_VALUE: "test2",
        }
        span._attributes = span.attributes

        # Measure time
        start = time.perf_counter()
        for _ in range(10000):  # Run many times
            processor.on_end(span)  # Full on_end path
        duration = time.perf_counter() - start

        avg_time_us = (duration / 10000) * 1_000_000
        # Should be extremely fast (just a boolean check)
        # Using 15µs threshold to account for system load variance
        assert avg_time_us < 15, f"Disabled masking overhead {avg_time_us:.2f}µs is too high"


class TestMaskableAttributesConstant:
    """Test MASKABLE_ATTRIBUTES constant."""

    def test_maskable_attributes_defined(self):
        """Verify MASKABLE_ATTRIBUTES is properly defined."""
        assert len(MASKABLE_ATTRIBUTES) == 5
        assert Attrs.INPUT_VALUE in MASKABLE_ATTRIBUTES
        assert Attrs.OUTPUT_VALUE in MASKABLE_ATTRIBUTES
        assert Attrs.GEN_AI_INPUT_MESSAGES in MASKABLE_ATTRIBUTES
        assert Attrs.GEN_AI_OUTPUT_MESSAGES in MASKABLE_ATTRIBUTES
        assert Attrs.METADATA in MASKABLE_ATTRIBUTES

    def test_non_maskable_attributes_excluded(self):
        """Verify structural attributes are not in MASKABLE_ATTRIBUTES."""
        assert Attrs.GEN_AI_REQUEST_MODEL not in MASKABLE_ATTRIBUTES
        assert Attrs.SESSION_ID not in MASKABLE_ATTRIBUTES
        assert Attrs.GEN_AI_USAGE_INPUT_TOKENS not in MASKABLE_ATTRIBUTES
        assert Attrs.BROKLE_ENVIRONMENT not in MASKABLE_ATTRIBUTES


from brokle.utils.masking import MaskingHelper


class TestMaskingHelperEmails:
    """Test email masking helpers."""

    def test_mask_emails_simple(self):
        """Test simple email masking."""
        result = MaskingHelper.mask_emails("Contact john@example.com")
        assert result == "Contact [EMAIL]"

    def test_mask_emails_multiple(self):
        """Test multiple emails in one string."""
        result = MaskingHelper.mask_emails("Email john@example.com or admin@company.org")
        assert result == "Email [EMAIL] or [EMAIL]"

    def test_mask_emails_in_dict(self):
        """Test email masking in nested dicts."""
        data = {"user": {"email": "john@example.com", "name": "John"}}
        result = MaskingHelper.mask_emails(data)
        assert result["user"]["email"] == "[EMAIL]"
        assert result["user"]["name"] == "John"

    def test_mask_emails_in_list(self):
        """Test email masking in lists."""
        data = ["john@example.com", "admin@company.org"]
        result = MaskingHelper.mask_emails(data)
        assert result == ["[EMAIL]", "[EMAIL]"]


class TestMaskingHelperPhones:
    """Test phone masking helpers."""

    def test_mask_phones_simple(self):
        """Test simple phone masking."""
        result = MaskingHelper.mask_phones("Call 555-123-4567")
        assert result == "Call [PHONE]"

    def test_mask_phones_multiple_formats(self):
        """Test different phone formats."""
        result = MaskingHelper.mask_phones("Call 555-123-4567 or 555.987.6543 or 5551234567")
        assert result == "Call [PHONE] or [PHONE] or [PHONE]"


class TestMaskingHelperSSN:
    """Test SSN masking helpers."""

    def test_mask_ssn_simple(self):
        """Test SSN masking."""
        result = MaskingHelper.mask_ssn("SSN: 123-45-6789")
        assert result == "SSN: [SSN]"

    def test_mask_ssn_multiple(self):
        """Test multiple SSNs."""
        result = MaskingHelper.mask_ssn("SSN1: 123-45-6789, SSN2: 987-65-4321")
        assert result == "SSN1: [SSN], SSN2: [SSN]"


class TestMaskingHelperCreditCards:
    """Test credit card masking helpers."""

    def test_mask_credit_cards_simple(self):
        """Test credit card masking."""
        result = MaskingHelper.mask_credit_cards("Card: 1234-5678-9012-3456")
        assert result == "Card: [CREDIT_CARD]"

    def test_mask_credit_cards_no_separators(self):
        """Test credit card without separators."""
        result = MaskingHelper.mask_credit_cards("Card: 1234567890123456")
        assert result == "Card: [CREDIT_CARD]"


class TestMaskingHelperAPIKeys:
    """Test API key masking helpers."""

    def test_mask_api_keys_sk(self):
        """Test masking sk_ API keys."""
        result = MaskingHelper.mask_api_keys("Key: sk_test_1234567890abcdefghij")
        assert result == "Key: [API_KEY]"

    def test_mask_api_keys_pk(self):
        """Test masking pk_ API keys."""
        result = MaskingHelper.mask_api_keys("Key: pk_live_1234567890abcdefghij")
        assert result == "Key: [API_KEY]"

    def test_mask_api_keys_bk(self):
        """Test masking bk_ API keys."""
        result = MaskingHelper.mask_api_keys("Key: bk_prod_1234567890abcdefghij")
        assert result == "Key: [API_KEY]"


class TestMaskingHelperPII:
    """Test combined PII masking."""

    def test_mask_pii_all_patterns(self):
        """Test masking all PII patterns at once."""
        text = (
            "Contact john@example.com or call 555-123-4567. "
            "SSN: 123-45-6789, Card: 1234-5678-9012-3456, "
            "Key: sk_test_1234567890abcdefghij"
        )
        result = MaskingHelper.mask_pii(text)

        assert "[EMAIL]" in result
        assert "[PHONE]" in result
        assert "[SSN]" in result
        assert "[CREDIT_CARD]" in result
        assert "[API_KEY]" in result

        # Ensure no PII remains
        assert "john@example.com" not in result
        assert "555-123-4567" not in result
        assert "123-45-6789" not in result
        assert "1234-5678-9012-3456" not in result
        assert "sk_test" not in result

    def test_mask_pii_nested_structure(self):
        """Test PII masking in complex nested structures."""
        data = {
            "user": {
                "email": "john@example.com",
                "phone": "555-123-4567",
                "name": "John",
            },
            "payment": {"card": "1234-5678-9012-3456", "amount": 100},
            "contacts": ["admin@company.org", "support@company.org"],
        }

        result = MaskingHelper.mask_pii(data)

        assert result["user"]["email"] == "[EMAIL]"
        assert result["user"]["phone"] == "[PHONE]"
        assert result["user"]["name"] == "John"  # Not PII
        assert result["payment"]["card"] == "[CREDIT_CARD]"
        assert result["payment"]["amount"] == 100  # Not PII
        assert result["contacts"] == ["[EMAIL]", "[EMAIL]"]


class TestMaskingHelperFieldMask:
    """Test field-based masking."""

    def test_field_mask_simple(self):
        """Test simple field masking."""
        mask_fn = MaskingHelper.field_mask(["password", "ssn"])
        data = {"user": "john", "password": "secret123", "age": 30}
        result = mask_fn(data)

        assert result["user"] == "john"
        assert result["password"] == "***MASKED***"
        assert result["age"] == 30

    def test_field_mask_nested(self):
        """Test field masking in nested dicts."""
        mask_fn = MaskingHelper.field_mask(["password", "api_key"])
        data = {
            "user": "john",
            "credentials": {"password": "secret", "api_key": "key123"},
        }
        result = mask_fn(data)

        assert result["credentials"]["password"] == "***MASKED***"
        assert result["credentials"]["api_key"] == "***MASKED***"

    def test_field_mask_custom_replacement(self):
        """Test field masking with custom replacement."""
        mask_fn = MaskingHelper.field_mask(["secret"], replacement="[REDACTED]")
        data = {"secret": "value", "public": "data"}
        result = mask_fn(data)

        assert result["secret"] == "[REDACTED]"
        assert result["public"] == "data"

    def test_field_mask_case_insensitive(self):
        """Test case-insensitive field masking (default)."""
        mask_fn = MaskingHelper.field_mask(["PASSWORD"])
        data = {"password": "secret", "Password": "secret2", "PASSWORD": "secret3"}
        result = mask_fn(data)

        # All variations should be masked (case-insensitive by default)
        assert result["password"] == "***MASKED***"
        assert result["Password"] == "***MASKED***"
        assert result["PASSWORD"] == "***MASKED***"

    def test_field_mask_case_sensitive(self):
        """Test case-sensitive field masking."""
        mask_fn = MaskingHelper.field_mask(["password"], case_sensitive=True)
        data = {"password": "secret", "Password": "secret2"}
        result = mask_fn(data)

        assert result["password"] == "***MASKED***"
        assert result["Password"] == "secret2"  # Not masked (case-sensitive)


class TestMaskingHelperCombinators:
    """Test advanced masking combinators."""

    def test_combine_masks(self):
        """Test combining multiple mask functions."""
        combined = MaskingHelper.combine_masks(
            MaskingHelper.mask_emails,
            MaskingHelper.mask_phones,
            MaskingHelper.field_mask(["password"]),
        )

        data = {
            "email": "john@example.com",
            "phone": "555-123-4567",
            "password": "secret123",
        }
        result = combined(data)

        assert result["email"] == "[EMAIL]"
        assert result["phone"] == "[PHONE]"
        assert result["password"] == "***MASKED***"

    def test_custom_pattern_mask(self):
        """Test custom regex pattern masking."""
        # Mask IPv4 addresses
        mask_ip = MaskingHelper.custom_pattern_mask(
            r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "[IP_ADDRESS]"
        )

        result = mask_ip("Server at 192.168.1.1 and 10.0.0.1")
        assert result == "Server at [IP_ADDRESS] and [IP_ADDRESS]"

    def test_custom_pattern_mask_case_insensitive(self):
        """Test custom pattern with case-insensitive flag."""
        import re

        mask_secret = MaskingHelper.custom_pattern_mask(
            r"\bsecret\b", "[REDACTED]", flags=re.IGNORECASE
        )

        result = mask_secret("This is Secret and this is SECRET")
        assert result == "This is [REDACTED] and this is [REDACTED]"


class TestMaskingHelperEdgeCases:
    """Test edge cases for MaskingHelper."""

    def test_mask_none_value(self):
        """Test masking None values."""
        result = MaskingHelper.mask_pii(None)
        assert result is None

    def test_mask_empty_string(self):
        """Test masking empty string."""
        result = MaskingHelper.mask_pii("")
        assert result == ""

    def test_mask_empty_dict(self):
        """Test masking empty dict."""
        result = MaskingHelper.mask_pii({})
        assert result == {}

    def test_mask_empty_list(self):
        """Test masking empty list."""
        result = MaskingHelper.mask_pii([])
        assert result == []

    def test_mask_primitives(self):
        """Test masking primitive types."""
        assert MaskingHelper.mask_pii(42) == 42
        assert MaskingHelper.mask_pii(3.14) == 3.14
        assert MaskingHelper.mask_pii(True) is True
        assert MaskingHelper.mask_pii(False) is False

    def test_mask_mixed_types(self):
        """Test masking mixed type structures."""
        data = {
            "string": "john@example.com",
            "number": 42,
            "boolean": True,
            "null": None,
            "list": [1, "admin@company.org", None],
        }
        result = MaskingHelper.mask_pii(data)

        assert result["string"] == "[EMAIL]"
        assert result["number"] == 42
        assert result["boolean"] is True
        assert result["null"] is None
        assert result["list"] == [1, "[EMAIL]", None]

    def test_mask_large_payload(self):
        """Test masking large nested payloads."""
        # Create large nested structure
        large_data = {
            "users": [
                {
                    "id": i,
                    "email": f"user{i}@example.com",
                    "phone": f"555-{i:03d}-{i:04d}",
                    "metadata": {"key": f"value{i}"},
                }
                for i in range(100)
            ]
        }

        result = MaskingHelper.mask_pii(large_data)

        # Verify structure preserved
        assert len(result["users"]) == 100
        # Verify masking applied
        assert result["users"][0]["email"] == "[EMAIL]"
        assert result["users"][0]["phone"] == "[PHONE]"
        assert result["users"][0]["id"] == 0  # Non-PII preserved


class TestMaskingHelperIntegration:
    """Integration tests with real Brokle client setup."""

    def test_masking_with_client_initialization(self):
        """Test that masking can be configured at client initialization."""
        from brokle.config import BrokleConfig

        # This should not raise any errors
        config = BrokleConfig(
            api_key="bk_test_key_12345", mask=MaskingHelper.mask_emails
        )

        assert config.mask is not None
        # Test that the mask function works
        assert config.mask("test@example.com") == "[EMAIL]"

    def test_combining_multiple_helpers(self):
        """Test combining multiple MaskingHelper functions."""
        combined = MaskingHelper.combine_masks(
            MaskingHelper.mask_emails,
            MaskingHelper.mask_phones,
            MaskingHelper.mask_api_keys,
        )

        data = {
            "email": "admin@example.com",
            "phone": "555-123-4567",
            "key": "sk_test_1234567890abcdefghij",
        }

        result = combined(data)

        assert result["email"] == "[EMAIL]"
        assert result["phone"] == "[PHONE]"
        assert result["key"] == "[API_KEY]"

    def test_masking_with_real_brokle_client(self):
        """
        CRITICAL REGRESSION TEST: Verify masking works with real Brokle client.

        This test uses actual OpenTelemetry spans (not mocks) to verify that
        masking works correctly with ReadableSpan's immutable .attributes
        (MappingProxyType).

        The fix accesses span._attributes (internal mutable dict) instead of
        span.attributes (immutable proxy) to apply masking transformations.

        Note: Uses short timeout and catches all errors to avoid test hangs
        in CI environments where localhost may not respond.
        """
        from brokle import Brokle

        # Create real client with masking and short timeout
        client = Brokle(
            api_key="bk_" + "x" * 40,
            base_url="http://localhost:8080",
            mask=MaskingHelper.mask_emails,
            timeout=1,  # 1 second timeout to prevent hangs
        )

        # Create real span (triggers OpenTelemetry span creation)
        with client.start_as_current_span(
            "test-real-masking",
            input="Contact john@example.com",
            output="Sent to admin@company.org"
        ) as span:
            # Set attribute with email (will be masked)
            span.set_attribute("gen_ai.input.messages", "Email: help@example.com")

        # Flush triggers on_end() with real ReadableSpan
        # If masking uses span.attributes (immutable), this raises:
        #   TypeError: 'mappingproxy' object does not support item assignment
        # If masking uses span._attributes (mutable), this works (network error OK)
        try:
            client.flush()
            client.shutdown()
        except TypeError as e:
            # TypeError means the immutability bug exists - FAIL THE TEST
            if "mappingproxy" in str(e) or "does not support item assignment" in str(e):
                pytest.fail(
                    f"CRITICAL BUG: Masking failed due to immutable attributes: {e}\n"
                    f"Fix: Ensure processor._apply_masking uses span._attributes, not span.attributes"
                )
            raise
        except Exception:
            # Network errors (timeout, 404, connection refused, etc.) are expected and OK
            # We only care that masking didn't cause TypeError
            pass

        # If we reach here without TypeError, the _attributes fix works!
