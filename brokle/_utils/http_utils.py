"""
HTTP utilities for Brokle SDK.

Simple HTTP utilities extracted and cleaned from the old integration framework.
"""

import logging
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


def safe_serialize_request(
    data: Dict[str, Any], max_length: int = 1000
) -> Dict[str, Any]:
    """
    Safely serialize request data, removing sensitive information.

    Args:
        data: Request data to serialize
        max_length: Maximum length for text fields

    Returns:
        Sanitized request data
    """
    try:
        # Filter to safe fields
        safe_fields = {
            "model",
            "messages",
            "prompt",
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "stream",
            "n",
            "logprobs",
            "echo",
            "suffix",
            "user",
            "input",
            "encoding_format",
            "dimensions",
            "top_k",
            "stop_sequences",
        }

        filtered_data = {k: v for k, v in data.items() if k in safe_fields}

        # Sanitize messages if present
        if "messages" in filtered_data and isinstance(filtered_data["messages"], list):
            sanitized_messages = []
            for msg in filtered_data["messages"][:10]:  # Limit to first 10
                if isinstance(msg, dict) and "content" in msg:
                    sanitized_msg = {
                        "role": msg.get("role", "unknown"),
                        "content": (
                            str(msg["content"])[:max_length] if msg["content"] else None
                        ),
                    }
                    sanitized_messages.append(sanitized_msg)
            filtered_data["messages"] = sanitized_messages

        # Sanitize prompt if present
        if "prompt" in filtered_data and filtered_data["prompt"]:
            filtered_data["prompt"] = str(filtered_data["prompt"])[:max_length]

        return filtered_data

    except Exception as e:
        logger.debug(f"Failed to serialize request data: {e}")
        return {"extraction_error": "Failed to extract request data"}


def safe_serialize_response(result: Any, max_length: int = 2000) -> Dict[str, Any]:
    """
    Safely serialize response data, limiting content length.

    Args:
        result: Response object to serialize
        max_length: Maximum length for content fields

    Returns:
        Sanitized response data
    """
    try:
        if hasattr(result, "model_dump"):
            # Pydantic model
            data = result.model_dump()
        elif hasattr(result, "__dict__"):
            # Generic object with attributes
            data = result.__dict__.copy()
        else:
            return {"type": str(type(result).__name__)}

        # Filter to safe fields
        safe_fields = {
            "id",
            "object",
            "created",
            "model",
            "choices",
            "usage",
            "data",
            "system_fingerprint",
            "type",
            "role",
            "content",
            "stop_reason",
            "stop_sequence",
        }

        filtered_data = {k: v for k, v in data.items() if k in safe_fields}

        # Sanitize choices if present
        if "choices" in filtered_data and isinstance(filtered_data["choices"], list):
            sanitized_choices = []
            for choice in filtered_data["choices"][:5]:  # Limit to first 5
                # Create a safe representation without mutating original objects
                choice_dict = {}

                if hasattr(choice, "__dict__"):
                    # Build new dict from choice attributes
                    for key, value in choice.__dict__.items():
                        if key == "message" and hasattr(value, "content"):
                            # Create new message dict with truncated content
                            original_content = value.content
                            truncated_content = (
                                str(original_content)[:max_length] + "..."
                                if original_content
                                and len(str(original_content)) > max_length
                                else original_content
                            )
                            choice_dict[key] = {
                                "role": getattr(value, "role", None),
                                "content": truncated_content,
                            }
                        else:
                            choice_dict[key] = value
                elif isinstance(choice, dict):
                    # Build new dict from choice dict
                    for key, value in choice.items():
                        if (
                            key == "message"
                            and isinstance(value, dict)
                            and "content" in value
                        ):
                            # Create new message dict with truncated content
                            original_content = value["content"]
                            truncated_content = (
                                str(original_content)[:max_length] + "..."
                                if original_content
                                and len(str(original_content)) > max_length
                                else original_content
                            )
                            choice_dict[key] = {
                                "role": value.get("role"),
                                "content": truncated_content,
                            }
                        else:
                            choice_dict[key] = value
                else:
                    continue

                sanitized_choices.append(choice_dict)
            filtered_data["choices"] = sanitized_choices

        return filtered_data

    except Exception as e:
        logger.debug(f"Failed to serialize response data: {e}")
        return {"extraction_error": "Failed to extract response data"}


def classify_provider_error(exc: Exception, provider: str) -> str:
    """
    Classify provider-specific errors into standard categories.

    Args:
        exc: Exception to classify
        provider: Provider name (openai, anthropic, etc.)

    Returns:
        Error classification string
    """
    exc_name = type(exc).__name__.lower()
    exc_message = str(exc).lower()

    # Common error patterns
    if "ratelimiterror" in exc_name or "rate limit" in exc_message:
        return "rate_limit"
    elif "authenticationerror" in exc_name or "authentication" in exc_message:
        return "auth_failure"
    elif "permissionerror" in exc_name or "permission" in exc_message:
        return "permission_denied"
    elif "notfounderror" in exc_name or "not found" in exc_message:
        return "model_unavailable"
    elif "timeouterror" in exc_name or "timeout" in exc_message:
        return "timeout"
    elif "connectionerror" in exc_name or "connection" in exc_message:
        return "network_error"
    elif "badrequest" in exc_name or "invalid" in exc_message:
        return "invalid_request"
    else:
        return "unknown_error"
