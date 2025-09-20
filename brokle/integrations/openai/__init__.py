"""
OpenAI Auto-Instrumentation for Brokle Platform.

This module provides automatic instrumentation for OpenAI API calls.
Simply import this module to enable comprehensive observability for all OpenAI usage.

Usage:
    import brokle.integrations.openai  # Enables auto-instrumentation

    from openai import OpenAI
    client = OpenAI()
    # All OpenAI calls now automatically tracked by Brokle
"""

import logging
import time
import functools
import inspect
from typing import Any, Dict, Optional, Union
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Check for required dependencies
try:
    import wrapt
    HAS_WRAPT = True
except ImportError:
    HAS_WRAPT = False
    logger.warning("wrapt library not available - auto-instrumentation disabled")

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    openai = None

# Global instrumentation state
_instrumented = False
_instrumentation_errors = []


def _get_brokle_client():
    """Get Brokle client with error handling."""
    try:
        from ...client import get_client
        return get_client()
    except Exception as e:
        logger.debug(f"Brokle client not available: {e}")
        return None


def _extract_request_data(args, kwargs) -> Dict[str, Any]:
    """Extract safe request data from method arguments."""
    try:
        # Skip 'self' parameter, extract key fields
        data = kwargs.copy()

        # Add positional args if present (skip 'self')
        if len(args) > 1:
            if 'model' not in data and len(args) > 1:
                data['model'] = args[1]
            if 'messages' not in data and len(args) > 2:
                data['messages'] = args[2]
            if 'prompt' not in data and len(args) > 2:
                data['prompt'] = args[2]

        # Filter to safe, relevant fields only
        safe_fields = {
            'model', 'messages', 'prompt', 'temperature', 'max_tokens',
            'top_p', 'frequency_penalty', 'presence_penalty', 'stop',
            'stream', 'n', 'logprobs', 'echo', 'suffix', 'user',
            'input', 'encoding_format', 'dimensions'
        }

        filtered_data = {k: v for k, v in data.items() if k in safe_fields}

        # Sanitize messages if present (remove any sensitive content)
        if 'messages' in filtered_data and isinstance(filtered_data['messages'], list):
            sanitized_messages = []
            for msg in filtered_data['messages'][:10]:  # Limit to first 10 messages
                if isinstance(msg, dict) and 'content' in msg:
                    sanitized_msg = {
                        'role': msg.get('role', 'unknown'),
                        'content': str(msg['content'])[:1000] if msg['content'] else None  # Limit content length
                    }
                    sanitized_messages.append(sanitized_msg)
            filtered_data['messages'] = sanitized_messages

        return filtered_data

    except Exception as e:
        logger.debug(f"Failed to extract request data: {e}")
        return {"extraction_error": "Failed to extract request data"}


def _extract_response_data(result) -> Dict[str, Any]:
    """Extract safe response data from method result."""
    try:
        if hasattr(result, 'model_dump'):
            # Pydantic model
            data = result.model_dump()
        elif hasattr(result, '__dict__'):
            # Generic object with attributes
            data = result.__dict__.copy()
        else:
            return {"type": str(type(result).__name__)}

        # Filter to safe fields
        safe_fields = {
            'id', 'object', 'created', 'model', 'choices', 'usage',
            'data', 'system_fingerprint'
        }

        filtered_data = {k: v for k, v in data.items() if k in safe_fields}

        # Sanitize choices if present (limit content length)
        if 'choices' in filtered_data and isinstance(filtered_data['choices'], list):
            sanitized_choices = []
            for choice in filtered_data['choices'][:5]:  # Limit to first 5 choices
                if hasattr(choice, '__dict__'):
                    choice_dict = choice.__dict__.copy()
                elif isinstance(choice, dict):
                    choice_dict = choice.copy()
                else:
                    continue

                # Limit message content length
                if 'message' in choice_dict and hasattr(choice_dict['message'], 'content'):
                    content = choice_dict['message'].content
                    if content and len(str(content)) > 2000:
                        choice_dict['message'].content = str(content)[:2000] + "..."

                sanitized_choices.append(choice_dict)
            filtered_data['choices'] = sanitized_choices

        return filtered_data

    except Exception as e:
        logger.debug(f"Failed to extract response data: {e}")
        return {"extraction_error": "Failed to extract response data"}


def _calculate_cost(model: str, usage: Dict[str, Any]) -> Optional[float]:
    """Calculate approximate OpenAI cost based on model and usage."""
    if not model or not usage:
        return None

    try:
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # Simplified pricing (per 1K tokens) - approximate rates
        pricing_map = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-32k": {"input": 0.06, "output": 0.12},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
            "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
            "text-embedding-ada-002": {"input": 0.0001, "output": 0.0001},
            "text-embedding-3-small": {"input": 0.00002, "output": 0.00002},
            "text-embedding-3-large": {"input": 0.00013, "output": 0.00013},
        }

        # Find matching pricing
        pricing = None
        model_lower = model.lower()
        for model_key, model_pricing in pricing_map.items():
            if model_key in model_lower:
                pricing = model_pricing
                break

        if not pricing:
            # Default fallback pricing
            pricing = {"input": 0.01, "output": 0.02}

        input_cost = (prompt_tokens / 1000) * pricing["input"]
        output_cost = (completion_tokens / 1000) * pricing["output"]

        return round(input_cost + output_cost, 6)

    except Exception as e:
        logger.debug(f"Failed to calculate cost: {e}")
        return None


def _create_openai_wrapper(method_name: str, operation_type: str = "llm"):
    """Create a wrapper function for OpenAI methods."""

    def wrapper(wrapped, instance, args, kwargs):
        """Universal wrapper for OpenAI methods with comprehensive observability."""
        start_time = time.perf_counter()
        client = _get_brokle_client()

        # Initialize tracking variables
        observation_id = None
        observation = None
        request_data = {}

        # Setup observability (non-blocking - never fail user's code)
        if client:
            try:
                # Extract request data safely
                request_data = _extract_request_data(args, kwargs)

                # Create generation span for LLM operations
                if operation_type == "llm":
                    observation = client.generation(
                        name=f"OpenAI {method_name}",
                        model=request_data.get("model", "unknown"),
                        provider="openai",
                        metadata={
                            "method": method_name,
                            "operation_type": operation_type,
                            "library": "openai",
                            "auto_instrumented": True
                        }
                    ).start()
                    observation_id = id(observation)  # Use object id since spans don't have .id
                else:
                    # For embeddings and other operations, use span
                    observation = client.span(
                        name=f"OpenAI {method_name}",
                        metadata={
                            "provider": "openai",
                            "method": method_name,
                            "operation_type": operation_type,
                            "library": "openai",
                            "auto_instrumented": True,
                            "model": request_data.get("model", "unknown")
                        }
                    ).start()
                    observation_id = id(observation)  # Use object id since spans don't have .id

                logger.debug(f"Created {operation_type} observation {observation_id} for OpenAI {method_name}")

            except Exception as e:
                logger.debug(f"Failed to setup observability for {method_name}: {e}")
                # Continue without observability - never break user code
                observation = None
                observation_id = None

        # Execute original method (this should NEVER fail due to observability)
        try:
            result = wrapped(*args, **kwargs)

            # Check if result is a coroutine (async method)
            is_coroutine = inspect.iscoroutine(result)

            # Only calculate latency for sync methods (P1 fix: avoid bogus latency for async)
            latency_ms = None
            if not is_coroutine:
                end_time = time.perf_counter()
                latency_ms = int((end_time - start_time) * 1000)

            # Complete observation with success data
            if observation:
                try:
                    # Initialize variables for both sync and async cases
                    total_cost = None

                    # For coroutines, we can't extract response data until it's awaited
                    # Only process response data for sync methods
                    if not is_coroutine:
                        # Extract response data safely
                        response_data = _extract_response_data(result)
                        usage = response_data.get("usage", {})

                        # Ensure usage is a dict (guard against None from streaming/certain endpoints)
                        if usage is None:
                            usage = {}

                        # Calculate cost
                        total_cost = _calculate_cost(
                            request_data.get("model", ""),
                            usage
                        ) if request_data else None

                        # Update observation with response data
                        observation.update(
                            metadata={
                                **observation.metadata,
                                "response": response_data,
                                "usage": usage,
                                "total_cost": total_cost
                            }
                        )

                        # Update metrics for OTEL attributes (P1 fix: restore dashboard/alert metrics)
                        if hasattr(observation, 'update_metrics'):
                            observation.update_metrics(
                                input_tokens=usage.get("prompt_tokens"),
                                output_tokens=usage.get("completion_tokens"),
                                total_tokens=usage.get("total_tokens"),
                                cost_usd=total_cost,
                                latency_ms=latency_ms
                            )
                    else:
                        # For async methods, just update basic metadata
                        observation.update(
                            metadata={
                                **observation.metadata,
                                "async_method": True,
                                "note": "Metrics will be incomplete for async methods - response not yet awaited"
                            }
                        )

                    # End the observation successfully
                    observation.end(status_message="success")

                    # Log completion (handle both sync and async cases)
                    if not is_coroutine:
                        logger.debug(f"Completed observation {observation_id} for {method_name} (cost: ${total_cost})")
                    else:
                        logger.debug(f"Completed observation {observation_id} for {method_name} (async - metrics incomplete)")

                except Exception as e:
                    logger.debug(f"Failed to complete observation for {method_name}: {e}")
                    # Don't fail user code even if observability completion fails

            return result

        except Exception as e:
            # Record error in observation, but always re-raise original exception
            if observation:
                try:
                    from opentelemetry.trace import StatusCode

                    # End the observation with error status
                    observation.end(
                        status=StatusCode.ERROR,
                        status_message=f"error: {str(e)[:200]}"  # Limit error message length
                    )

                    logger.debug(f"Recorded error in observation {observation_id} for {method_name}")

                except Exception as complete_error:
                    logger.debug(f"Failed to record error in observation: {complete_error}")
                    # Even error recording failures shouldn't break user code

            # Always re-raise the original exception
            raise

    return wrapper


def _instrument_openai_methods():
    """Instrument OpenAI methods with Brokle observability."""
    global _instrumented, _instrumentation_errors

    if _instrumented:
        logger.debug("OpenAI already instrumented")
        return True

    if not HAS_OPENAI:
        error_msg = "OpenAI library not available for instrumentation"
        _instrumentation_errors.append(error_msg)
        logger.warning(error_msg)
        return False

    if not HAS_WRAPT:
        error_msg = "wrapt library not available for instrumentation"
        _instrumentation_errors.append(error_msg)
        logger.warning(error_msg)
        return False

    try:
        # Define methods to instrument with their configurations
        methods_to_instrument = [
            # Chat completions
            {
                "module": "openai.resources.chat.completions",
                "object": "Completions.create",
                "name": "chat_completions_create",
                "type": "llm"
            },
            {
                "module": "openai.resources.chat.completions",
                "object": "AsyncCompletions.create",
                "name": "async_chat_completions_create",
                "type": "llm"
            },
            # Text completions
            {
                "module": "openai.resources.completions",
                "object": "Completions.create",
                "name": "completions_create",
                "type": "llm"
            },
            {
                "module": "openai.resources.completions",
                "object": "AsyncCompletions.create",
                "name": "async_completions_create",
                "type": "llm"
            },
            # Embeddings
            {
                "module": "openai.resources.embeddings",
                "object": "Embeddings.create",
                "name": "embeddings_create",
                "type": "embedding"
            },
            {
                "module": "openai.resources.embeddings",
                "object": "AsyncEmbeddings.create",
                "name": "async_embeddings_create",
                "type": "embedding"
            }
        ]

        instrumented_count = 0

        for method_config in methods_to_instrument:
            try:
                wrapt.wrap_function_wrapper(
                    method_config["module"],
                    method_config["object"],
                    _create_openai_wrapper(
                        method_config["name"],
                        method_config["type"]
                    )
                )
                instrumented_count += 1
                logger.debug(f"Instrumented {method_config['module']}.{method_config['object']}")

            except Exception as e:
                error_msg = f"Failed to instrument {method_config['object']}: {e}"
                _instrumentation_errors.append(error_msg)
                logger.debug(error_msg)

        if instrumented_count > 0:
            _instrumented = True
            logger.info(f"Brokle OpenAI auto-instrumentation enabled ({instrumented_count} methods)")
            return True
        else:
            error_msg = "No OpenAI methods could be instrumented"
            _instrumentation_errors.append(error_msg)
            logger.warning(error_msg)
            return False

    except Exception as e:
        error_msg = f"Failed to instrument OpenAI library: {e}"
        _instrumentation_errors.append(error_msg)
        logger.error(error_msg)
        return False


def is_instrumented() -> bool:
    """Check if OpenAI is currently instrumented."""
    return _instrumented


def get_instrumentation_errors() -> list:
    """Get list of instrumentation errors."""
    return _instrumentation_errors.copy()


def get_instrumentation_status() -> Dict[str, Any]:
    """Get detailed instrumentation status."""
    return {
        "instrumented": _instrumented,
        "openai_available": HAS_OPENAI,
        "wrapt_available": HAS_WRAPT,
        "errors": _instrumentation_errors.copy(),
        "client_available": _get_brokle_client() is not None
    }


# Auto-instrument on import
if HAS_OPENAI and HAS_WRAPT:
    try:
        _instrument_openai_methods()
    except Exception as e:
        logger.warning(f"Auto-instrumentation failed on import: {e}")
        _instrumentation_errors.append(f"Auto-instrumentation failed: {e}")
else:
    missing = []
    if not HAS_OPENAI:
        missing.append("openai")
    if not HAS_WRAPT:
        missing.append("wrapt")

    logger.debug(f"Auto-instrumentation skipped - missing dependencies: {', '.join(missing)}")


# Export public interface
__all__ = [
    "is_instrumented",
    "get_instrumentation_errors",
    "get_instrumentation_status",
    "HAS_OPENAI",
    "HAS_WRAPT"
]