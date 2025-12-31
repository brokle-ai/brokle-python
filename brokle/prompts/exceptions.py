"""
Prompt-specific exceptions.

These exceptions provide structured error handling for prompt operations
with consistent error codes and status codes for API compatibility.
"""

from typing import List, Optional


class PromptError(Exception):
    """
    Base exception for prompt operations.

    All prompt-related exceptions inherit from this class, providing
    consistent error handling with error codes and optional HTTP status codes.

    Attributes:
        message: Human-readable error message
        code: Machine-readable error code (e.g., "PROMPT_NOT_FOUND")
        status_code: Optional HTTP status code for API responses
    """

    def __init__(self, message: str, code: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code

    def __str__(self) -> str:
        if self.status_code:
            return f"[{self.code}] {self.message} (HTTP {self.status_code})"
        return f"[{self.code}] {self.message}"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"code={self.code!r}, "
            f"status_code={self.status_code!r})"
        )


class PromptNotFoundError(PromptError):
    """
    Raised when a prompt is not found.

    This exception is raised when attempting to fetch a prompt that
    doesn't exist or when the specified version/label is not available.

    Attributes:
        name: The prompt name that was requested
        version: The specific version requested (if any)
        label: The label requested (if any)
    """

    def __init__(
        self, name: str, version: Optional[int] = None, label: Optional[str] = None
    ):
        if version is not None:
            target = f"version {version}"
        elif label:
            target = f'label "{label}"'
        else:
            target = "latest"

        message = f'Prompt "{name}" not found ({target})'
        super().__init__(message, "PROMPT_NOT_FOUND", 404)

        self.name = name
        self.version = version
        self.label = label


class PromptCompileError(PromptError):
    """
    Raised when prompt compilation fails.

    This exception is raised when template compilation fails,
    typically due to missing required variables.

    Attributes:
        missing_variables: List of variable names that were required but not provided
    """

    def __init__(self, message: str, missing_variables: Optional[List[str]] = None):
        super().__init__(message, "PROMPT_COMPILE_ERROR")
        self.missing_variables = missing_variables or []


class PromptFetchError(PromptError):
    """
    Raised when fetching a prompt fails.

    This exception is raised when an HTTP request to fetch a prompt
    fails due to network issues, server errors, or other problems.

    Attributes:
        status_code: HTTP status code from the failed request (if available)
    """

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message, "PROMPT_FETCH_ERROR", status_code)
