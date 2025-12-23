"""
Exception classes for the datasets module.

This module provides exception classes for dataset-related errors.
"""


class DatasetError(Exception):
    """
    Error with dataset operations.

    Raised when dataset API requests fail due to:
    - Network errors
    - Authentication/authorization issues
    - Invalid dataset data
    - Server errors
    """

    pass
