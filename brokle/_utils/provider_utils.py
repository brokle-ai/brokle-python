"""
Provider utilities for Brokle SDK.

Simple provider utilities extracted and cleaned from the old integration framework.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def normalize_token_usage(usage: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize token usage across different providers.

    Args:
        usage: Raw usage dictionary from provider

    Returns:
        Normalized usage with standard field names
    """
    if not isinstance(usage, dict):
        return {}

    normalized = {}

    # Map common field variations to standard names
    input_fields = ['input_tokens', 'prompt_tokens', 'tokens_input']
    output_fields = ['output_tokens', 'completion_tokens', 'tokens_output']
    total_fields = ['total_tokens', 'tokens_total']

    # Find input tokens
    for field in input_fields:
        if field in usage and usage[field] is not None:
            normalized['input_tokens'] = int(usage[field])
            break
    else:
        normalized['input_tokens'] = 0

    # Find output tokens
    for field in output_fields:
        if field in usage and usage[field] is not None:
            normalized['output_tokens'] = int(usage[field])
            break
    else:
        normalized['output_tokens'] = 0

    # Find total tokens or calculate it
    for field in total_fields:
        if field in usage and usage[field] is not None:
            normalized['total_tokens'] = int(usage[field])
            break
    else:
        normalized['total_tokens'] = normalized['input_tokens'] + normalized['output_tokens']

    return normalized


def estimate_token_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    provider: str = "openai"
) -> Optional[float]:
    """
    Estimate cost for token usage based on model pricing.

    Args:
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        provider: Provider name

    Returns:
        Estimated cost in USD or None if pricing unknown
    """
    # Simplified pricing table (per 1M tokens)
    pricing = {
        "openai": {
            "gpt-4": {"input": 30.0, "output": 60.0},
            "gpt-4-turbo": {"input": 10.0, "output": 30.0},
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
            "text-embedding-ada-002": {"input": 0.1, "output": 0.0}
        },
        "anthropic": {
            "claude-3-opus": {"input": 15.0, "output": 75.0},
            "claude-3-sonnet": {"input": 3.0, "output": 15.0},
            "claude-3-haiku": {"input": 0.25, "output": 1.25}
        }
    }

    try:
        provider_pricing = pricing.get(provider.lower(), {})

        # Find closest model match
        model_key = None
        model_lower = model.lower()
        for key in provider_pricing:
            if key in model_lower:
                model_key = key
                break

        if not model_key:
            return None

        model_pricing = provider_pricing[model_key]
        input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output"]

        return round(input_cost + output_cost, 6)

    except Exception as e:
        logger.debug(f"Failed to estimate cost for {model}: {e}")
        return None


def detect_provider_from_model(model: str) -> str:
    """
    Detect provider from model name.

    Args:
        model: Model name

    Returns:
        Provider name
    """
    model_lower = model.lower()

    if any(prefix in model_lower for prefix in ['gpt-', 'text-', 'davinci', 'curie']):
        return "openai"
    elif any(prefix in model_lower for prefix in ['claude-', 'anthropic']):
        return "anthropic"
    elif any(prefix in model_lower for prefix in ['gemini', 'palm', 'bison']):
        return "google"
    elif any(prefix in model_lower for prefix in ['command', 'embed']):
        return "cohere"
    else:
        return "unknown"


def standardize_error_response(
    error: Exception,
    provider: str,
    operation: str
) -> Dict[str, Any]:
    """
    Standardize error response across providers.

    Args:
        error: Original exception
        provider: Provider name
        operation: Operation that failed

    Returns:
        Standardized error response
    """
    error_type = type(error).__name__
    error_message = str(error)

    # Classify error
    if "rate" in error_message.lower() or "429" in error_message:
        category = "rate_limit"
    elif "auth" in error_message.lower() or "401" in error_message:
        category = "authentication"
    elif "permission" in error_message.lower() or "403" in error_message:
        category = "permission"
    elif "not found" in error_message.lower() or "404" in error_message:
        category = "not_found"
    elif "timeout" in error_message.lower():
        category = "timeout"
    elif "connection" in error_message.lower():
        category = "network"
    else:
        category = "unknown"

    return {
        "provider": provider,
        "operation": operation,
        "error_type": error_type,
        "error_message": error_message,
        "error_category": category,
        "retryable": category in ["rate_limit", "timeout", "network"]
    }


def extract_model_info(response: Any) -> Dict[str, Any]:
    """
    Extract model information from provider response.

    Args:
        response: Provider response object

    Returns:
        Model information dictionary
    """
    info = {}

    try:
        # Common attributes to check
        if hasattr(response, 'model'):
            info['model'] = response.model
        elif hasattr(response, 'model_name'):
            info['model'] = response.model_name

        if hasattr(response, 'id'):
            info['id'] = response.id

        if hasattr(response, 'created'):
            info['created'] = response.created

        if hasattr(response, 'object'):
            info['object'] = response.object

        # Provider-specific handling
        if hasattr(response, 'usage'):
            usage = response.usage
            if hasattr(usage, '__dict__'):
                info['usage'] = normalize_token_usage(usage.__dict__)
            elif isinstance(usage, dict):
                info['usage'] = normalize_token_usage(usage)

    except Exception as e:
        logger.debug(f"Failed to extract model info: {e}")

    return info


def build_provider_headers(
    provider: str,
    api_key: str,
    additional_headers: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """
    Build standard headers for provider requests.

    Args:
        provider: Provider name
        api_key: API key for the provider
        additional_headers: Additional headers to include

    Returns:
        Headers dictionary
    """
    headers = {
        "User-Agent": "Brokle-SDK/1.0.0",
        "Content-Type": "application/json"
    }

    # Provider-specific auth headers
    if provider.lower() == "openai":
        headers["Authorization"] = f"Bearer {api_key}"
    elif provider.lower() == "anthropic":
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
    elif provider.lower() == "google":
        headers["Authorization"] = f"Bearer {api_key}"
    elif provider.lower() == "cohere":
        headers["Authorization"] = f"Bearer {api_key}"

    # Add additional headers
    if additional_headers:
        headers.update(additional_headers)

    return headers