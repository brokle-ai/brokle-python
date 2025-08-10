"""
Response parsing utilities for Brokle SDK.
"""

import logging
from typing import Any, Dict, Optional, Type, TypeVar, Union

from pydantic import BaseModel, ValidationError

from .error_handler import handle_api_error, BrokleError

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


def parse_api_response(
    response_data: Dict[str, Any],
    response_model: Optional[Type[T]] = None,
    status_code: int = 200
) -> Union[T, Dict[str, Any]]:
    """Parse API response and convert to appropriate format."""
    # Check if response indicates an error
    if not response_data.get('success', True):
        raise handle_api_error(response_data, status_code)
    
    # Extract data from response
    data = response_data.get('data', response_data)
    
    # Parse with response model if provided
    if response_model:
        try:
            return response_model(**data)
        except ValidationError as e:
            logger.error(f"Failed to parse response: {e}")
            raise BrokleError(
                message="Invalid response format",
                error_code="response_parsing_error",
                details={"validation_errors": e.errors()}
            )
    
    return data


def parse_error_response(response_data: Dict[str, Any], status_code: int) -> BrokleError:
    """Parse error response and create appropriate exception."""
    return handle_api_error(response_data, status_code)


def extract_response_metadata(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata from API response."""
    meta = response_data.get('meta', {})
    
    # Extract common metadata fields
    metadata = {
        'request_id': meta.get('request_id'),
        'timestamp': meta.get('timestamp'),
        'processing_time_ms': meta.get('processing_time_ms'),
        'version': meta.get('version'),
    }
    
    # Extract Brokle specific metadata
    brokle_meta = {
        'provider': meta.get('provider'),
        'provider_model_id': meta.get('provider_model_id'),
        'routing_strategy': meta.get('routing_strategy'),
        'routing_reason': meta.get('routing_reason'),
        'cost_usd': meta.get('cost_usd'),
        'cost_input_usd': meta.get('cost_input_usd'),
        'cost_output_usd': meta.get('cost_output_usd'),
        'tokens_input': meta.get('tokens_input'),
        'tokens_output': meta.get('tokens_output'),
        'tokens_total': meta.get('tokens_total'),
        'latency_ms': meta.get('latency_ms'),
        'provider_latency_ms': meta.get('provider_latency_ms'),
        'gateway_latency_ms': meta.get('gateway_latency_ms'),
        'cached': meta.get('cached'),
        'cache_hit': meta.get('cache_hit'),
        'cache_strategy': meta.get('cache_strategy'),
        'cache_similarity_score': meta.get('cache_similarity_score'),
        'evaluation_scores': meta.get('evaluation_scores'),
        'quality_score': meta.get('quality_score'),
        'ab_test_variant': meta.get('ab_test_variant'),
        'ab_test_experiment': meta.get('ab_test_experiment'),
        'custom_tags': meta.get('custom_tags'),
    }
    
    # Merge metadata
    metadata.update(brokle_meta)
    
    # Remove None values
    return {k: v for k, v in metadata.items() if v is not None}


def validate_response_format(response_data: Any) -> Dict[str, Any]:
    """Validate response format and ensure it's a dictionary."""
    if not isinstance(response_data, dict):
        raise BrokleError(
            message="Invalid response format: expected JSON object",
            error_code="invalid_response_format",
            details={"response_type": type(response_data).__name__}
        )
    
    return response_data


def extract_streaming_metadata(chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract metadata from streaming response chunk."""
    if not isinstance(chunk, dict):
        return None
    
    # Look for metadata in common locations
    meta_fields = ['meta', 'metadata', 'x-metadata']
    
    for field in meta_fields:
        if field in chunk:
            return extract_response_metadata({field: chunk[field]})
    
    return None