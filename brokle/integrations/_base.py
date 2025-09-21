"""
Base Instrumentation Framework for Brokle Platform.

This module provides the foundational interfaces and data structures for
implementing auto-instrumentation across multiple AI providers. The abstraction
layer ensures consistent behavior while allowing provider-specific customization.

Usage:
    class MyProviderInstrumentation(BaseInstrumentation):
        def get_method_configs(self) -> List[InstrumentationConfig]:
            return [
                InstrumentationConfig(
                    module="my_provider.resources.chat",
                    object="Chat.create",
                    name="my_provider_chat_create",
                    operation_type="llm"
                )
            ]

        def extract_request_metadata(self, args, kwargs) -> Dict[str, Any]:
            # Provider-specific request parsing
            pass
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


@dataclass
class InstrumentationConfig:
    """Configuration for instrumenting a specific provider method."""

    module: str                         # "openai.resources.chat.completions"
    object: str                        # "Completions.create"
    name: str                          # "chat_completions_create"
    operation_type: str                # "llm", "embedding", "image", "audio"
    is_async: bool = False             # Whether method returns awaitable
    supports_streaming: bool = False   # Whether method supports streaming responses
    error_types: List[Type[Exception]] = field(default_factory=list)  # Provider-specific errors

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.module or not self.object or not self.name:
            raise ValueError("module, object, and name are required")

        if self.operation_type not in ["llm", "embedding", "image", "audio", "other"]:
            raise ValueError(f"Invalid operation_type: {self.operation_type}")


@dataclass
class ManualInstrumentationOptions:
    """Typed options for manual instrumentation."""
    enable_caching: Optional[bool] = None
    enable_evaluation: Optional[bool] = None
    tags: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)  # Future extensibility


def _normalize_token_usage(usage: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize token usage fields to consistent naming.

    Handles both v1 field names (prompt_tokens/completion_tokens) and
    v2 field names (input_tokens/output_tokens) for cross-provider compatibility.

    Args:
        usage: Raw usage dictionary from provider response

    Returns:
        Dictionary with normalized field names (input_tokens, output_tokens, total_tokens)
    """
    if not usage:
        return {}

    normalized = {}

    # Input tokens: prefer input_tokens, fallback to prompt_tokens
    if "input_tokens" in usage:
        input_tokens = usage["input_tokens"]
    elif "prompt_tokens" in usage:
        input_tokens = usage["prompt_tokens"]
    else:
        input_tokens = None

    if input_tokens is not None:
        normalized["input_tokens"] = input_tokens

    # Output tokens: prefer output_tokens, fallback to completion_tokens
    if "output_tokens" in usage:
        output_tokens = usage["output_tokens"]
    elif "completion_tokens" in usage:
        output_tokens = usage["completion_tokens"]
    else:
        output_tokens = None

    if output_tokens is not None:
        normalized["output_tokens"] = output_tokens

    # Total tokens: prefer total_tokens, fallback to sum of input+output
    total_tokens = usage.get("total_tokens")
    if total_tokens is not None:
        normalized["total_tokens"] = total_tokens
    elif input_tokens is not None and output_tokens is not None:
        normalized["total_tokens"] = input_tokens + output_tokens

    return normalized


@dataclass
class StandardMetadata:
    """Standardized metadata structure across all providers."""

    provider: str                      # "openai", "anthropic", "gemini"
    model: str                         # "gpt-4", "claude-3", etc.
    operation_type: str                # "llm", "embedding", etc.
    request: Dict[str, Any]            # Sanitized request data
    response: Dict[str, Any]           # Sanitized response data
    usage: Dict[str, Any]              # Token/character usage stats
    cost_usd: Optional[float] = None   # Calculated cost in USD
    latency_ms: Optional[int] = None   # Response latency
    auto_instrumented: bool = True     # Whether auto or manually instrumented
    manual_options: Optional[ManualInstrumentationOptions] = None  # Manual wrapper options
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_otel_attributes(self) -> Dict[str, Any]:
        """Convert to OpenTelemetry span attributes."""
        attrs = {
            "brokle.provider": self.provider,
            "brokle.model": self.model,
            "brokle.operation.type": self.operation_type,
            "brokle.auto_instrumented": self.auto_instrumented,
        }

        # Add usage metrics if available
        if self.usage:
            normalized_usage = _normalize_token_usage(self.usage)
            if normalized_usage.get("input_tokens") is not None:
                attrs["brokle.usage.input_tokens"] = normalized_usage["input_tokens"]
            if normalized_usage.get("output_tokens") is not None:
                attrs["brokle.usage.output_tokens"] = normalized_usage["output_tokens"]
            if normalized_usage.get("total_tokens") is not None:
                attrs["brokle.usage.total_tokens"] = normalized_usage["total_tokens"]

        # Add performance metrics
        if self.cost_usd is not None:
            attrs["brokle.cost.usd"] = self.cost_usd
        if self.latency_ms is not None:
            attrs["brokle.latency.ms"] = self.latency_ms

        # Add manual configuration flags if available
        if self.manual_options:
            if self.manual_options.enable_caching is not None:
                attrs["brokle.manual.caching_enabled"] = self.manual_options.enable_caching
            if self.manual_options.enable_evaluation is not None:
                attrs["brokle.manual.evaluation_enabled"] = self.manual_options.enable_evaluation
            # Add tags if available
            if self.manual_options.tags:
                for key, value in self.manual_options.tags.items():
                    attrs[f"brokle.manual.tag.{key}"] = value
            # Add extra options
            if self.manual_options.extra:
                for key, value in self.manual_options.extra.items():
                    attrs[f"brokle.manual.extra.{key}"] = value

        return attrs


class BaseInstrumentation(ABC):
    """
    Abstract base class for all provider instrumentations.

    This interface defines the contract that all provider integrations must
    implement. It ensures consistent behavior across providers while allowing
    for provider-specific customization.
    """

    def __init__(self, provider_name: str):
        """Initialize with provider name for identification."""
        self.provider_name = provider_name
        self._method_configs = None  # Lazy-loaded cache

    @abstractmethod
    def get_method_configs(self) -> List[InstrumentationConfig]:
        """
        Return list of methods to instrument for this provider.

        Returns:
            List of InstrumentationConfig objects defining which methods
            to wrap and how to handle them.
        """
        pass

    @abstractmethod
    def extract_request_metadata(self, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """
        Extract safe request metadata from method arguments.

        Args:
            args: Positional arguments passed to the instrumented method
            kwargs: Keyword arguments passed to the instrumented method

        Returns:
            Dictionary containing sanitized request data (no sensitive info)
        """
        pass

    @abstractmethod
    def extract_response_metadata(self, result: Any) -> Dict[str, Any]:
        """
        Extract safe response metadata from method result.

        Args:
            result: Return value from the instrumented method

        Returns:
            Dictionary containing sanitized response data and usage metrics
        """
        pass

    @abstractmethod
    def calculate_cost(self, model: str, usage: Dict[str, Any]) -> Optional[float]:
        """
        Calculate approximate cost for the API call.

        Args:
            model: Model name used for the request
            usage: Usage statistics (tokens, characters, etc.)

        Returns:
            Estimated cost in USD, or None if calculation not possible
        """
        pass

    def classify_error(self, exc: Exception) -> str:
        """
        Classify provider-specific errors for better error handling.

        Args:
            exc: Exception that occurred during method execution

        Returns:
            Error classification string (e.g., "rate_limit", "auth_failure")
        """
        exc_name = type(exc).__name__.lower()

        # Common error patterns across providers
        if "rate" in exc_name or "quota" in exc_name:
            return "rate_limit"
        elif "auth" in exc_name or "permission" in exc_name or "forbidden" in exc_name:
            return "auth_failure"
        elif "not_found" in exc_name or "model" in exc_name:
            return "model_unavailable"
        elif "timeout" in exc_name or "connection" in exc_name:
            return "network_error"
        else:
            return "unknown_error"

    def should_suppress_error(self, exc: Exception) -> bool:
        """
        Determine if an error should be suppressed in instrumentation.

        Some provider-specific errors might be expected and shouldn't
        prevent instrumentation from working.

        Args:
            exc: Exception that occurred during instrumentation

        Returns:
            True if error should be suppressed, False otherwise
        """
        error_type = self.classify_error(exc)

        # Generally suppress network and temporary errors
        return error_type in ["network_error", "rate_limit"]

    def get_supported_versions(self) -> List[str]:
        """
        Return list of supported provider SDK versions.

        Returns:
            List of version strings this instrumentation supports
        """
        return ["*"]  # Default to supporting all versions

    def validate_environment(self) -> Dict[str, Any]:
        """
        Validate the environment for this provider.

        Returns:
            Dictionary with validation results and any issues found
        """
        return {
            "provider": self.provider_name,
            "supported": True,
            "issues": [],
            "version": "unknown"
        }

    def create_standard_metadata(
        self,
        config: InstrumentationConfig,
        args: tuple,
        kwargs: dict,
        result: Any,
        latency_ms: Optional[int] = None
    ) -> StandardMetadata:
        """
        Create standardized metadata from instrumentation data.

        This is a helper method that combines provider-specific extraction
        with the standard metadata structure.

        Args:
            config: Configuration for the instrumented method
            args: Method arguments
            kwargs: Method keyword arguments
            result: Method return value
            latency_ms: Optional latency measurement

        Returns:
            StandardMetadata object with all relevant information
        """
        request_data = self.extract_request_metadata(args, kwargs)
        response_data = self.extract_response_metadata(result)

        # Extract model and usage from request/response
        model = request_data.get("model", "unknown")
        usage = response_data.get("usage") or {}
        cost = self.calculate_cost(model, usage)

        return StandardMetadata(
            provider=self.provider_name,
            model=model,
            operation_type=config.operation_type,
            request=request_data,
            response=response_data,
            usage=usage,
            cost_usd=cost,
            latency_ms=latency_ms,
            auto_instrumented=True
        )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"{self.__class__.__name__}(provider='{self.provider_name}')"


class InstrumentationError(Exception):
    """Base exception for instrumentation-related errors."""

    def __init__(self, message: str, provider: str, method: str = None):
        self.message = message
        self.provider = provider
        self.method = method
        super().__init__(message)


class UnsupportedProviderError(InstrumentationError):
    """Raised when trying to use an unsupported provider."""
    pass


class ConfigurationError(InstrumentationError):
    """Raised when provider configuration is invalid."""
    pass