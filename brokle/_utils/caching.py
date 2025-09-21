"""
Caching utilities for Brokle SDK.

Simple caching utilities extracted and cleaned from the old integration framework.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def generate_cache_key(
    data: Dict[str, Any],
    prefix: str = "brokle",
    max_length: int = 250
) -> str:
    """
    Generate a cache key from request data.

    Args:
        data: Request data to hash
        prefix: Key prefix
        max_length: Maximum key length

    Returns:
        Cache key string
    """
    try:
        # Sort data for consistent hashing
        sorted_data = json.dumps(data, sort_keys=True, separators=(',', ':'))

        # Create hash
        hash_obj = hashlib.sha256(sorted_data.encode('utf-8'))
        hash_hex = hash_obj.hexdigest()[:16]  # Use first 16 chars

        # Create key with prefix
        cache_key = f"{prefix}:{hash_hex}"

        # Truncate if too long
        if len(cache_key) > max_length:
            cache_key = cache_key[:max_length]

        return cache_key

    except Exception as e:
        logger.debug(f"Failed to generate cache key: {e}")
        # Fallback to timestamp-based key
        return f"{prefix}:{int(time.time())}"


def normalize_cache_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize data for consistent caching.

    Args:
        data: Data to normalize

    Returns:
        Normalized data dictionary
    """
    if not isinstance(data, dict):
        return {}

    normalized = {}

    # Fields that affect response content
    cache_relevant_fields = {
        'model', 'messages', 'prompt', 'temperature', 'max_tokens',
        'top_p', 'frequency_penalty', 'presence_penalty', 'stop',
        'system', 'tools', 'tool_choice'
    }

    for key, value in data.items():
        if key in cache_relevant_fields:
            # Normalize the value
            if isinstance(value, (str, int, float, bool)) or value is None:
                normalized[key] = value
            elif isinstance(value, list):
                # Normalize lists (e.g., messages)
                normalized[key] = [
                    normalize_cache_data(item) if isinstance(item, dict) else item
                    for item in value
                ]
            elif isinstance(value, dict):
                normalized[key] = normalize_cache_data(value)
            else:
                normalized[key] = str(value)

    return normalized


def is_cacheable_request(data: Dict[str, Any]) -> bool:
    """
    Determine if a request is suitable for caching.

    Args:
        data: Request data to check

    Returns:
        True if request is cacheable
    """
    try:
        # Don't cache streaming requests
        if data.get('stream', False):
            return False

        # Don't cache requests with high randomness
        temperature = data.get('temperature', 0.0)
        if isinstance(temperature, (int, float)) and temperature > 0.8:
            return False

        # Don't cache requests with tools/function calling
        if data.get('tools') or data.get('functions'):
            return False

        # Must have model and content
        if not data.get('model'):
            return False

        # Check for content in messages or prompt
        messages = data.get('messages', [])
        prompt = data.get('prompt')

        if not prompt and not messages:
            return False

        return True

    except Exception as e:
        logger.debug(f"Error checking cacheability: {e}")
        return False


def extract_cache_metadata(response: Any) -> Dict[str, Any]:
    """
    Extract metadata from response for cache storage.

    Args:
        response: Response object

    Returns:
        Cache metadata dictionary
    """
    metadata = {
        'cached_at': int(time.time()),
        'cache_version': '1.0'
    }

    try:
        # Extract basic response info
        if hasattr(response, 'id'):
            metadata['response_id'] = response.id

        if hasattr(response, 'model'):
            metadata['model'] = response.model

        if hasattr(response, 'created'):
            metadata['created'] = response.created

        # Extract usage info if available
        if hasattr(response, 'usage'):
            usage = response.usage
            if hasattr(usage, '__dict__'):
                metadata['usage'] = usage.__dict__
            elif isinstance(usage, dict):
                metadata['usage'] = usage

    except Exception as e:
        logger.debug(f"Failed to extract cache metadata: {e}")

    return metadata


def calculate_cache_similarity(
    cached_data: Dict[str, Any],
    request_data: Dict[str, Any],
    threshold: float = 0.9
) -> float:
    """
    Calculate similarity between cached and request data.

    Args:
        cached_data: Previously cached request data
        request_data: Current request data
        threshold: Similarity threshold

    Returns:
        Similarity score (0.0 to 1.0)
    """
    try:
        # For now, use exact match similarity
        # In a full implementation, this would use embedding similarity

        # Normalize both datasets
        cached_norm = normalize_cache_data(cached_data)
        request_norm = normalize_cache_data(request_data)

        # Check exact match for key fields
        key_fields = ['model', 'temperature', 'max_tokens']

        matches = 0
        total = len(key_fields)

        for field in key_fields:
            if cached_norm.get(field) == request_norm.get(field):
                matches += 1

        # Check content similarity (simplified)
        cached_content = str(cached_norm.get('messages', cached_norm.get('prompt', '')))
        request_content = str(request_norm.get('messages', request_norm.get('prompt', '')))

        if cached_content == request_content:
            matches += 2  # Weight content more heavily
            total += 2

        similarity = matches / total if total > 0 else 0.0
        return similarity

    except Exception as e:
        logger.debug(f"Failed to calculate cache similarity: {e}")
        return 0.0


def should_use_cache(
    request_data: Dict[str, Any],
    cache_enabled: bool = True,
    max_age_seconds: int = 3600
) -> bool:
    """
    Determine if cache should be used for this request.

    Args:
        request_data: Request data
        cache_enabled: Whether caching is enabled
        max_age_seconds: Maximum cache age

    Returns:
        True if cache should be used
    """
    if not cache_enabled:
        return False

    # Check if request is cacheable
    if not is_cacheable_request(request_data):
        return False

    # Add any additional business logic here
    return True