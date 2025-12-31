"""
Built-in masking utilities for common PII patterns.

Provides pre-built masking functions and helpers for common privacy use cases:
- Email addresses
- Phone numbers
- SSN (Social Security Numbers)
- Credit card numbers
- API keys
- Custom field-based masking

All masking functions support recursive application to nested data structures
(dicts, lists) while preserving the original structure.
"""

import re
from typing import Any, Callable, List


class MaskingHelper:
    """
    Pre-built masking functions for common PII patterns.

    All methods are static and can be used directly or composed together
    for custom masking strategies.

    Example:
        from brokle import Brokle
        from brokle.utils.masking import MaskingHelper

        # Use pre-built PII masker
        client = Brokle(api_key="bk_secret", mask=MaskingHelper.mask_pii)

        # Use specific masker
        client = Brokle(api_key="bk_secret", mask=MaskingHelper.mask_emails)

        # Field-based masking
        client = Brokle(
            api_key="bk_secret",
            mask=MaskingHelper.field_mask(['password', 'ssn', 'api_key'])
        )
    """

    # ========== Regex Patterns ==========

    EMAIL_PATTERN = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    """Matches email addresses (RFC 5322 simplified)"""

    PHONE_PATTERN = r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"
    """Matches US phone numbers (xxx-xxx-xxxx, xxx.xxx.xxxx, xxxxxxxxxx)"""

    SSN_PATTERN = r"\b\d{3}-\d{2}-\d{4}\b"
    """Matches US Social Security Numbers (xxx-xx-xxxx)"""

    CREDIT_CARD_PATTERN = r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"
    """Matches credit card numbers (16 digits with optional separators)"""

    API_KEY_PATTERN = r"(sk|pk|bk|api)_[a-zA-Z0-9_]{20,}"
    """Matches common API key formats (sk_, pk_, bk_, api_ prefix + 20+ chars)"""

    # ========== Replacement Strings ==========

    EMAIL_REPLACEMENT = "[EMAIL]"
    PHONE_REPLACEMENT = "[PHONE]"
    SSN_REPLACEMENT = "[SSN]"
    CREDIT_CARD_REPLACEMENT = "[CREDIT_CARD]"
    API_KEY_REPLACEMENT = "[API_KEY]"

    # ========== Primary Masking Functions ==========

    @staticmethod
    def mask_pii(data: Any) -> Any:
        """
        Mask all common PII patterns (email, phone, SSN, credit cards, API keys).

        This is the recommended all-in-one masking function for general use.

        Args:
            data: The data to mask (supports strings, dicts, lists, primitives)

        Returns:
            Masked data with same structure as input

        Example:
            >>> mask_pii("Contact john@example.com or call 555-123-4567")
            'Contact [EMAIL] or call [PHONE]'

            >>> mask_pii({"email": "admin@company.com", "count": 42})
            {'email': '[EMAIL]', 'count': 42}
        """
        return MaskingHelper._recursive_mask(
            data,
            lambda s: MaskingHelper._apply_all_patterns(s),
        )

    @staticmethod
    def mask_emails(data: Any) -> Any:
        """
        Mask email addresses only.

        Args:
            data: The data to mask

        Returns:
            Data with emails replaced by [EMAIL]

        Example:
            >>> mask_emails("Send to john@example.com and admin@company.org")
            'Send to [EMAIL] and [EMAIL]'
        """
        return MaskingHelper._recursive_mask(
            data,
            lambda s: re.sub(
                MaskingHelper.EMAIL_PATTERN, MaskingHelper.EMAIL_REPLACEMENT, s
            ),
        )

    @staticmethod
    def mask_phones(data: Any) -> Any:
        """
        Mask phone numbers only.

        Args:
            data: The data to mask

        Returns:
            Data with phone numbers replaced by [PHONE]

        Example:
            >>> mask_phones("Call 555-123-4567 or 555.987.6543")
            'Call [PHONE] or [PHONE]'
        """
        return MaskingHelper._recursive_mask(
            data,
            lambda s: re.sub(
                MaskingHelper.PHONE_PATTERN, MaskingHelper.PHONE_REPLACEMENT, s
            ),
        )

    @staticmethod
    def mask_ssn(data: Any) -> Any:
        """
        Mask Social Security Numbers only.

        Args:
            data: The data to mask

        Returns:
            Data with SSNs replaced by [SSN]

        Example:
            >>> mask_ssn("SSN: 123-45-6789")
            'SSN: [SSN]'
        """
        return MaskingHelper._recursive_mask(
            data,
            lambda s: re.sub(
                MaskingHelper.SSN_PATTERN, MaskingHelper.SSN_REPLACEMENT, s
            ),
        )

    @staticmethod
    def mask_credit_cards(data: Any) -> Any:
        """
        Mask credit card numbers only.

        Args:
            data: The data to mask

        Returns:
            Data with credit card numbers replaced by [CREDIT_CARD]

        Example:
            >>> mask_credit_cards("Card: 1234-5678-9012-3456")
            'Card: [CREDIT_CARD]'
        """
        return MaskingHelper._recursive_mask(
            data,
            lambda s: re.sub(
                MaskingHelper.CREDIT_CARD_PATTERN,
                MaskingHelper.CREDIT_CARD_REPLACEMENT,
                s,
            ),
        )

    @staticmethod
    def mask_api_keys(data: Any) -> Any:
        """
        Mask API keys only.

        Matches common patterns: sk_, pk_, bk_, api_ followed by 20+ characters.

        Args:
            data: The data to mask

        Returns:
            Data with API keys replaced by [API_KEY]

        Example:
            >>> mask_api_keys("Key: sk_test_1234567890abcdefghij")
            'Key: [API_KEY]'
        """
        return MaskingHelper._recursive_mask(
            data,
            lambda s: re.sub(
                MaskingHelper.API_KEY_PATTERN, MaskingHelper.API_KEY_REPLACEMENT, s
            ),
        )

    # ========== Field-Based Masking ==========

    @staticmethod
    def field_mask(
        field_names: List[str],
        replacement: str = "***MASKED***",
        case_sensitive: bool = False,
    ) -> Callable[[Any], Any]:
        """
        Create a masking function that masks specific field names in dictionaries.

        Useful for masking known sensitive fields by name (e.g., 'password', 'ssn').

        Args:
            field_names: List of field names to mask
            replacement: Replacement value for masked fields
            case_sensitive: Whether field name matching is case-sensitive

        Returns:
            A masking function ready to use with Brokle client

        Example:
            >>> masker = field_mask(['password', 'ssn', 'api_key'])
            >>> masker({'user': 'john', 'password': 'secret123'})
            {'user': 'john', 'password': '***MASKED***'}

            >>> # Use with Brokle
            >>> client = Brokle(api_key="bk_secret", mask=field_mask(['password']))
        """
        field_set = set(
            field_names if case_sensitive else [f.lower() for f in field_names]
        )

        def mask_fields(data: Any) -> Any:
            if isinstance(data, dict):
                result = {}
                for key, value in data.items():
                    check_key = key if case_sensitive else key.lower()
                    if check_key in field_set:
                        result[key] = replacement
                    elif isinstance(value, (dict, list)):
                        result[key] = mask_fields(value)
                    else:
                        result[key] = value
                return result
            elif isinstance(data, list):
                return [mask_fields(item) for item in data]
            else:
                return data

        return mask_fields

    # ========== Advanced Combinators ==========

    @staticmethod
    def combine_masks(*mask_functions: Callable[[Any], Any]) -> Callable[[Any], Any]:
        """
        Combine multiple masking functions into a single function.

        The functions are applied in order (left to right).

        Args:
            *mask_functions: Variable number of masking functions to combine

        Returns:
            A single masking function that applies all provided functions

        Example:
            >>> combined = combine_masks(
            ...     MaskingHelper.mask_emails,
            ...     MaskingHelper.mask_phones,
            ...     field_mask(['password'])
            ... )
            >>> client = Brokle(api_key="bk_secret", mask=combined)
        """

        def combined_mask(data: Any) -> Any:
            result = data
            for mask_fn in mask_functions:
                result = mask_fn(result)
            return result

        return combined_mask

    @staticmethod
    def custom_pattern_mask(
        pattern: str,
        replacement: str,
        flags: int = 0,
    ) -> Callable[[Any], Any]:
        r"""
        Create a custom regex-based masking function.

        Args:
            pattern: Regular expression pattern to match
            replacement: Replacement string for matches
            flags: Optional regex flags (e.g., re.IGNORECASE)

        Returns:
            A masking function for the custom pattern

        Example:
            >>> # Mask IPv4 addresses
            >>> mask_ip = custom_pattern_mask(
            ...     r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            ...     '[IP_ADDRESS]'
            ... )
            >>> mask_ip("Server: 192.168.1.1")
            'Server: [IP_ADDRESS]'
        """
        compiled_pattern = re.compile(pattern, flags)

        def mask_pattern(data: Any) -> Any:
            return MaskingHelper._recursive_mask(
                data, lambda s: compiled_pattern.sub(replacement, s)
            )

        return mask_pattern

    # ========== Internal Helpers ==========

    @staticmethod
    def _apply_all_patterns(text: str) -> str:
        """Apply all PII patterns to a string."""
        # Apply patterns in order of specificity (most to least specific)
        text = re.sub(MaskingHelper.SSN_PATTERN, MaskingHelper.SSN_REPLACEMENT, text)
        text = re.sub(
            MaskingHelper.CREDIT_CARD_PATTERN,
            MaskingHelper.CREDIT_CARD_REPLACEMENT,
            text,
        )
        text = re.sub(
            MaskingHelper.API_KEY_PATTERN, MaskingHelper.API_KEY_REPLACEMENT, text
        )
        text = re.sub(
            MaskingHelper.EMAIL_PATTERN, MaskingHelper.EMAIL_REPLACEMENT, text
        )
        text = re.sub(
            MaskingHelper.PHONE_PATTERN, MaskingHelper.PHONE_REPLACEMENT, text
        )
        return text

    @staticmethod
    def _recursive_mask(data: Any, mask_fn: Callable[[str], str]) -> Any:
        """
        Recursively apply a string masking function to nested data structures.

        Preserves structure and handles dicts, lists, strings, and primitives.

        Args:
            data: The data to mask
            mask_fn: Function that masks strings

        Returns:
            Masked data with same structure as input
        """
        if isinstance(data, dict):
            return {
                key: MaskingHelper._recursive_mask(value, mask_fn)
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [MaskingHelper._recursive_mask(item, mask_fn) for item in data]
        elif isinstance(data, str):
            return mask_fn(data)
        else:
            # Return primitives (int, float, bool, None) unchanged
            return data
