"""
Attribute validation utilities for Brokle integrations.

Provides utilities for validating and sanitizing OpenTelemetry attributes
to ensure they conform to the expected format and size constraints.
"""

import logging

logger = logging.getLogger(__name__)


class AttributeValidator:
    """
    Validates and sanitizes OpenTelemetry attributes.

    Ensures attributes conform to OpenTelemetry standards and
    Brokle-specific requirements for telemetry data.
    """

    @staticmethod
    def validate_attribute_value(value):
        """
        Validate and sanitize an attribute value for OpenTelemetry.

        Args:
            value: The attribute value to validate

        Returns:
            Sanitized attribute value
        """
        # Handle None values
        if value is None:
            return ""

        # Convert to string if not a basic type
        if not isinstance(value, (str, int, float, bool)):
            try:
                return str(value)
            except:
                return "<serialization_failed>"

        # Truncate long strings
        if isinstance(value, str) and len(value) > 1000:
            return value[:997] + "..."

        return value

    @staticmethod
    def sanitize_attributes(attributes: dict) -> dict:
        """
        Sanitize all attributes in a dictionary.

        Args:
            attributes: Dictionary of attributes

        Returns:
            Sanitized attributes dictionary
        """
        sanitized = {}

        for key, value in attributes.items():
            # Validate key
            if not isinstance(key, str):
                continue

            # Validate and sanitize value
            sanitized_value = AttributeValidator.validate_attribute_value(value)
            sanitized[key] = sanitized_value

        return sanitized
