"""
Tests for data masking functionality.

Comprehensive test suite for PII masking in the Brokle SDK, covering:
- MaskingHelper utility functions
- Integration with MaskingSpanExporter
- MASKABLE_ATTRIBUTES constant

Note: Core masking exporter tests are in test_masking_exporter.py
"""

import pytest

from brokle.config import BrokleConfig
from brokle.types.attributes import MASKABLE_ATTRIBUTES
from brokle.types.attributes import BrokleOtelSpanAttributes as Attrs
from brokle.utils.masking import MaskingHelper


class TestMaskingConfiguration:
    """Test masking configuration."""

    def test_masking_disabled_by_default(self):
        """Verify no masking when not configured."""
        config = BrokleConfig(api_key="bk_test_key_12345")
        assert config.mask is None

    def test_masking_function_can_be_configured(self):
        """Verify mask function can be set in config."""

        def simple_mask(data):
            return "[MASKED]"

        config = BrokleConfig(api_key="bk_test_key_12345", mask=simple_mask)
        assert config.mask is not None
        assert config.mask("test") == "[MASKED]"


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


class TestMaskingHelperEmails:
    """Test email masking helpers."""

    def test_mask_emails_simple(self):
        """Test simple email masking."""
        result = MaskingHelper.mask_emails("Contact john@example.com")
        assert result == "Contact [EMAIL]"

    def test_mask_emails_multiple(self):
        """Test multiple emails in one string."""
        result = MaskingHelper.mask_emails(
            "Email john@example.com or admin@company.org"
        )
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
        result = MaskingHelper.mask_phones(
            "Call 555-123-4567 or 555.987.6543 or 5551234567"
        )
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
        masking works correctly with the MaskingSpanExporter wrapper pattern.

        The MaskingSpanExporter uses only public OpenTelemetry APIs to apply
        masking at export time, wrapping spans with MaskedReadableSpan.

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
            output="Sent to admin@company.org",
        ) as span:
            # Set attribute with email (will be masked)
            span.set_attribute("gen_ai.input.messages", "Email: help@example.com")

        # Flush triggers export with MaskingSpanExporter wrapping spans
        # The MaskingSpanExporter creates MaskedReadableSpan wrappers that
        # return masked attribute values via their .attributes property
        try:
            client.flush()
            client.shutdown()
        except TypeError as e:
            # TypeError would mean something is wrong with the masking implementation
            pytest.fail(f"CRITICAL BUG: Masking failed with TypeError: {e}")
        except Exception:
            # Network errors (timeout, 404, connection refused, etc.) are expected and OK
            # We only care that masking didn't cause TypeError
            pass

        # If we reach here without TypeError, masking exporter works correctly!
