"""
MaskingHelper Utilities Example for Brokle Python SDK

This example demonstrates how to use the built-in MaskingHelper utilities
for common PII patterns without writing custom regex logic.

Key concepts:
- Pre-built PII maskers
- Field-based masking
- Combining multiple maskers
- Custom pattern creation
"""

import json
from brokle import Brokle
from brokle.utils.masking import MaskingHelper


def example_1_all_pii():
    """Use the all-in-one PII masker (recommended for most use cases)."""
    print("Example 1: All-in-One PII Masker")
    print("-" * 50)

    client = Brokle(
        api_key="bk_your_api_key",
        mask=MaskingHelper.mask_pii  # Masks emails, phones, SSN, cards, API keys
    )

    with client.start_as_current_span(
        "process-sensitive-data",
        input="""
        Contact: john@example.com
        Phone: 555-123-4567
        SSN: 123-45-6789
        Card: 1234-5678-9012-3456
        API Key: sk_test_1234567890abcdefghij1234567890
        """
    ) as span:
        pass  # Input already set at initialization

    print("✓ All PII patterns automatically masked")
    client.flush()
    print()


def example_2_specific_pii():
    """Use specific PII maskers for targeted protection."""
    print("Example 2: Specific PII Maskers")
    print("-" * 50)

    # Email masking only
    client_emails = Brokle(
        api_key="bk_your_api_key",
        mask=MaskingHelper.mask_emails
    )

    with client_emails.start_as_current_span(
        "email-only",
        input="Contact john@example.com or call 555-123-4567"
    ) as span:
        # Result: "Contact [EMAIL] or call 555-123-4567"
        pass

    print("✓ Email-only masking applied")

    # Phone masking only
    client_phones = Brokle(
        api_key="bk_your_api_key",
        mask=MaskingHelper.mask_phones
    )

    with client_phones.start_as_current_span(
        "phone-only",
        input="Email: admin@company.com, Phone: 555-987-6543"
    ) as span:
        # Result: "Email: admin@company.com, Phone: [PHONE]"
        pass

    print("✓ Phone-only masking applied")

    client_emails.flush()
    client_phones.flush()
    print()


def example_3_field_based():
    """Use field-based masking to target specific dictionary keys."""
    print("Example 3: Field-Based Masking")
    print("-" * 50)

    # Mask by field name
    client = Brokle(
        api_key="bk_your_api_key",
        mask=MaskingHelper.field_mask(['password', 'ssn', 'api_key', 'secret_token'])
    )

    with client.start_as_current_span("process-credentials") as span:
        credentials = {
            "username": "john_doe",  # Not masked
            "password": "super_secret_123",  # Masked
            "email": "john@example.com",  # Not masked (use mask_emails for this)
            "api_key": "sk_1234567890",  # Masked
            "created_at": "2024-01-01"  # Not masked
        }

        span.set_attribute("metadata", json.dumps(credentials))

    print("✓ Field-based masking applied to specific keys")
    client.flush()
    print()


def example_4_combined():
    """Combine multiple masking strategies."""
    print("Example 4: Combined Masking Strategies")
    print("-" * 50)

    # Combine pattern-based + field-based masking
    combined_mask = MaskingHelper.combine_masks(
        MaskingHelper.mask_emails,  # Mask all emails
        MaskingHelper.mask_phones,  # Mask all phones
        MaskingHelper.field_mask(['password', 'secret'])  # Mask specific fields
    )

    client = Brokle(api_key="bk_your_api_key", mask=combined_mask)

    with client.start_as_current_span("multi-strategy") as span:
        data = {
            "contact": "john@example.com or 555-123-4567",  # Email & phone masked
            "password": "my_secret",  # Field masked
            "public_info": "Not sensitive"  # Not masked
        }

        span.set_attribute("metadata", json.dumps(data))

    print("✓ Multiple masking strategies combined successfully")
    client.flush()
    print()


def example_5_custom_pattern():
    """Create custom masking patterns for domain-specific data."""
    print("Example 5: Custom Pattern Masking")
    print("-" * 50)

    # Mask IPv4 addresses
    mask_ip = MaskingHelper.custom_pattern_mask(
        r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
        '[IP_ADDRESS]'
    )

    client = Brokle(api_key="bk_your_api_key", mask=mask_ip)

    with client.start_as_current_span(
        "server-logs",
        input="Request from 192.168.1.1 to server 10.0.0.5"
    ) as span:
        # Result: "Request from [IP_ADDRESS] to server [IP_ADDRESS]"
        pass

    print("✓ Custom IPv4 masking pattern applied")
    client.flush()
    print()


def example_6_real_world():
    """Real-world example with LLM usage."""
    print("Example 6: Real-World LLM Example")
    print("-" * 50)

    # Combine all necessary PII protection
    mask = MaskingHelper.combine_masks(
        MaskingHelper.mask_pii,  # All common PII
        MaskingHelper.field_mask(['api_key', 'secret', 'token'])  # Sensitive fields
    )

    client = Brokle(api_key="bk_your_api_key", mask=mask)

    # Simulate LLM generation with sensitive user data
    with client.start_as_current_generation(
        name="customer-support-response",
        model="gpt-4",
        provider="openai",
        input_messages=[{
            "role": "user",
            "content": "User email: support@customer.com wants help with account 123-45-6789"
        }]
    ) as generation:
        # Simulated LLM response
        output_messages = [{
            "role": "assistant",
            "content": "I'll help you with that. Please verify at support@customer.com"
        }]

        generation.update(
            output_messages=output_messages,
            usage={"input_tokens": 25, "output_tokens": 15}
        )

    print("✓ LLM generation with full PII protection")
    client.flush()
    print()


def main():
    """Run all masking examples."""
    print("=" * 50)
    print("Brokle Masking Examples")
    print("=" * 50)
    print()

    example_1_all_pii()
    example_2_specific_pii()
    example_3_field_based()
    example_4_combined()
    example_5_custom_pattern()
    example_6_real_world()

    print("=" * 50)
    print("All examples completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
