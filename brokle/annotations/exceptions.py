"""
Annotation Queue Exceptions

Custom exceptions for annotation queue operations.
"""


class AnnotationError(Exception):
    """Base exception for annotation queue operations."""

    pass


class QueueNotFoundError(AnnotationError):
    """Raised when an annotation queue is not found."""

    pass


class ItemNotFoundError(AnnotationError):
    """Raised when an annotation queue item is not found."""

    pass


class ItemLockedError(AnnotationError):
    """Raised when an item is locked by another user."""

    pass


class NoItemsAvailableError(AnnotationError):
    """Raised when no items are available for annotation."""

    pass


class AssignmentError(AnnotationError):
    """Raised when assignment operations fail."""

    pass
