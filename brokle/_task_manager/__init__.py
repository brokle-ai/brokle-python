"""
Background task processing for the Brokle SDK.

This module provides non-blocking background processing capabilities
inspired by Optik's architecture.
"""

from .processor import get_background_processor, BackgroundProcessor

__all__ = [
    "get_background_processor",
    "BackgroundProcessor",
]