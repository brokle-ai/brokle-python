"""
Basic PII Masking Example for Brokle Python SDK

This example demonstrates how to implement custom masking logic to protect
sensitive information before it's sent to Brokle.

Key concepts:
- Client-side masking (before transmission)
- Custom masking functions
- Recursive masking for nested data
- Error-safe design
"""

import json
import re
from brokle import Brokle


def mask_emails(data):
    """
    Mask email addresses in any data structure.

    Supports strings, dicts, lists, and nested combinations.
    """
    if isinstance(data, str):
        # Replace email addresses with [EMAIL]
        return re.sub(r'\b[\w.]+@[\w.]+\b', '[EMAIL]', data)
    elif isinstance(data, dict):
        # Recursively mask dict values
        return {k: mask_emails(v) for k, v in data.items()}
    elif isinstance(data, list):
        # Recursively mask list items
        return [mask_emails(item) for item in data]
    else:
        # Return primitives unchanged
        return data


def mask_pii_comprehensive(data):
    """
    Mask multiple PII patterns comprehensively.

    Masks:
    - Email addresses
    - Phone numbers
    - SSN
    """
    if isinstance(data, str):
        # Apply multiple regex patterns
        masked = data
        masked = re.sub(r'\b[\w.]+@[\w.]+\b', '[EMAIL]', masked)
        masked = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', masked)
        masked = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', masked)
        return masked
    elif isinstance(data, dict):
        return {k: mask_pii_comprehensive(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [mask_pii_comprehensive(item) for item in data]
    else:
        return data


def main():
    """Demonstrate basic masking with Brokle client."""

    # Example 1: Email masking
    print("Example 1: Email Masking")
    print("-" * 50)

    client = Brokle(api_key="bk_your_api_key", mask=mask_emails)

    with client.start_as_current_span(
        "process-user-request",
        input="Please contact john@example.com for more information"
    ) as span:
        # Simulate processing
        response = "Email sent successfully"
        span.set_attribute("output.value", response)

    print("✓ Span created with masked email addresses")
    print()

    # Example 2: Comprehensive PII masking
    print("Example 2: Comprehensive PII Masking")
    print("-" * 50)

    client_comprehensive = Brokle(
        api_key="bk_your_api_key",
        mask=mask_pii_comprehensive
    )

    with client_comprehensive.start_as_current_span("process-contact") as span:
        contact_info = {
            "email": "admin@company.com",
            "phone": "555-123-4567",
            "ssn": "123-45-6789",
            "name": "John Doe"  # This won't be masked (not a pattern)
        }

        span.set_attribute("metadata", json.dumps(contact_info))

    print("✓ Span created with all PII patterns masked")
    print()

    # Example 3: Nested structure masking
    print("Example 3: Nested Structure Masking")
    print("-" * 50)

    nested_data = {
        "users": [
            {"email": "user1@example.com", "role": "admin"},
            {"email": "user2@example.com", "role": "user"}
        ],
        "admin": {
            "contact": {
                "email": "admin@example.com",
                "phone": "555-987-6543"
            }
        }
    }

    with client_comprehensive.start_as_current_span(
        "process-users",
        input=nested_data
    ) as span:
        pass  # Input already set via context manager

    print("✓ Nested structure masked while preserving structure")
    print()

    # Flush telemetry
    client.flush()
    client_comprehensive.flush()

    print("=" * 50)
    print("All examples completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
