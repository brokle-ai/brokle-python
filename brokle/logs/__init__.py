"""
Brokle Logs Module.

Provides LoggerProvider setup with OTLP export for structured logging
and event emission from AI applications.

Example:
    >>> from brokle.logs import BrokleLoggerProvider
    >>> provider = BrokleLoggerProvider(config)
    >>> logger = provider.get_logger()
"""

from .provider import BrokleLoggerProvider, create_logger_provider

__all__ = [
    "BrokleLoggerProvider",
    "create_logger_provider",
]
