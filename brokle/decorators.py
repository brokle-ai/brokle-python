"""
Decorators for Brokle SDK observability.

This module provides the @observe decorator following the LangFuse pattern
but adapted for Brokle's specific features.
"""

import asyncio
import functools
import inspect
import logging
from typing import Any, Callable, Dict, Optional, TypeVar, Union, List
from datetime import datetime

from .config import get_config
from .core.telemetry import get_telemetry_manager, start_span, start_generation
from .core.background_processor import get_background_processor
from .core.serialization import serialize

logger = logging.getLogger(__name__)

# Type variables
F = TypeVar('F', bound=Callable[..., Any])


class BrokleDecorator:
    """Implementation of the @observe decorator for Brokle Platform."""
    
    def __init__(self):
        self.telemetry_manager = get_telemetry_manager()
        self.background_processor = get_background_processor()
        self.config = get_config()
    
    def observe(
        self,
        func: Optional[F] = None,
        *,
        name: Optional[str] = None,
        as_type: Optional[str] = None,
        capture_input: Optional[bool] = None,
        capture_output: Optional[bool] = None,
        transform_to_string: Optional[Callable[[Any], str]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[F, Callable[[F], F]]:
        """
        Decorator for observing function calls with Brokle Platform.
        
        Args:
            func: The function to decorate
            name: Custom name for the observation
            as_type: Type of observation ('generation' for LLM calls, 'span' for general)
            capture_input: Whether to capture function input
            capture_output: Whether to capture function output
            transform_to_string: Function to transform output to string
            user_id: User ID for tracking
            session_id: Session ID for tracking
            tags: Tags for the observation
            metadata: Additional metadata
            **kwargs: Additional attributes
        
        Returns:
            Decorated function with observability
        """
        # Default values
        should_capture_input = capture_input if capture_input is not None else True
        should_capture_output = capture_output if capture_output is not None else True
        
        def decorator(func: F) -> F:
            if asyncio.iscoroutinefunction(func):
                return self._async_observe(
                    func,
                    name=name,
                    as_type=as_type,
                    capture_input=should_capture_input,
                    capture_output=should_capture_output,
                    transform_to_string=transform_to_string,
                    user_id=user_id,
                    session_id=session_id,
                    tags=tags,
                    metadata=metadata,
                    **kwargs
                )
            else:
                return self._sync_observe(
                    func,
                    name=name,
                    as_type=as_type,
                    capture_input=should_capture_input,
                    capture_output=should_capture_output,
                    transform_to_string=transform_to_string,
                    user_id=user_id,
                    session_id=session_id,
                    tags=tags,
                    metadata=metadata,
                    **kwargs
                )
        
        # Handle decorator with or without parentheses
        if func is None:
            return decorator
        else:
            return decorator(func)
    
    def _sync_observe(
        self,
        func: F,
        name: Optional[str],
        as_type: Optional[str],
        capture_input: bool,
        capture_output: bool,
        transform_to_string: Optional[Callable[[Any], str]],
        user_id: Optional[str],
        session_id: Optional[str],
        tags: Optional[List[str]],
        metadata: Optional[Dict[str, Any]],
        **kwargs
    ) -> F:
        """Synchronous wrapper for observed functions."""
        
        @functools.wraps(func)
        def wrapper(*args, **func_kwargs) -> Any:
            # Extract Brokle specific kwargs
            trace_id = func_kwargs.pop("brokle_trace_id", None)
            parent_span_id = func_kwargs.pop("brokle_parent_span_id", None)
            api_key = func_kwargs.pop("brokle_api_key", None)
            
            # Determine observation name
            obs_name = name or func.__name__
            
            # Capture input if enabled
            input_data = None
            if capture_input:
                input_data = self._get_input_from_func_args(
                    is_method=self._is_method(func),
                    func_args=args,
                    func_kwargs=func_kwargs
                )
            
            # Determine span type
            span_type = as_type or "span"
            
            # Create span based on type
            if span_type == "generation":
                span_context = start_generation(
                    obs_name,
                    input_data=input_data,
                    user_id=user_id,
                    session_id=session_id,
                    metadata=metadata,
                    tags=tags,
                    **kwargs
                )
            else:
                span_context = start_span(
                    obs_name,
                    span_type=span_type,
                    attributes={
                        "input_data": serialize(input_data) if input_data else None,
                        "user_id": user_id,
                        "session_id": session_id,
                        "tags": tags,
                        "metadata": serialize(metadata) if metadata else None,
                        **kwargs
                    }
                )
            
            with span_context as span:
                start_time = datetime.now()
                
                try:
                    # Execute function
                    result = func(*args, **func_kwargs)
                    
                    # Handle output capture
                    if capture_output:
                        output_data = result
                        
                        # Handle generators
                        if inspect.isgenerator(result):
                            output_data = self._wrap_sync_generator(
                                result, 
                                span, 
                                transform_to_string
                            )
                            return output_data
                        
                        # Apply transform if provided
                        if transform_to_string:
                            output_data = transform_to_string(result)
                        
                        # Update span with output
                        if span:
                            self.telemetry_manager.update_span_attributes(
                                span,
                                output_data=serialize(output_data)
                            )
                    
                    # Submit telemetry data
                    self._submit_telemetry(
                        span_name=obs_name,
                        span_type=span_type,
                        input_data=input_data,
                        output_data=result if capture_output else None,
                        start_time=start_time,
                        end_time=datetime.now(),
                        user_id=user_id,
                        session_id=session_id,
                        tags=tags,
                        metadata=metadata,
                        status="success"
                    )
                    
                    return result
                    
                except Exception as e:
                    # Record error in span
                    if span:
                        self.telemetry_manager.record_error(
                            span,
                            e,
                            error_type=type(e).__name__
                        )
                    
                    # Submit error telemetry
                    self._submit_telemetry(
                        span_name=obs_name,
                        span_type=span_type,
                        input_data=input_data,
                        output_data=None,
                        start_time=start_time,
                        end_time=datetime.now(),
                        user_id=user_id,
                        session_id=session_id,
                        tags=tags,
                        metadata=metadata,
                        status="error",
                        error=str(e)
                    )
                    
                    raise
        
        return wrapper
    
    def _async_observe(
        self,
        func: F,
        name: Optional[str],
        as_type: Optional[str],
        capture_input: bool,
        capture_output: bool,
        transform_to_string: Optional[Callable[[Any], str]],
        user_id: Optional[str],
        session_id: Optional[str],
        tags: Optional[List[str]],
        metadata: Optional[Dict[str, Any]],
        **kwargs
    ) -> F:
        """Asynchronous wrapper for observed functions."""
        
        @functools.wraps(func)
        async def async_wrapper(*args, **func_kwargs) -> Any:
            # Extract Brokle specific kwargs
            trace_id = func_kwargs.pop("brokle_trace_id", None)
            parent_span_id = func_kwargs.pop("brokle_parent_span_id", None)
            api_key = func_kwargs.pop("brokle_api_key", None)
            
            # Determine observation name
            obs_name = name or func.__name__
            
            # Capture input if enabled
            input_data = None
            if capture_input:
                input_data = self._get_input_from_func_args(
                    is_method=self._is_method(func),
                    func_args=args,
                    func_kwargs=func_kwargs
                )
            
            # Determine span type
            span_type = as_type or "span"
            
            # Create span based on type
            if span_type == "generation":
                span_context = start_generation(
                    obs_name,
                    input_data=input_data,
                    user_id=user_id,
                    session_id=session_id,
                    metadata=metadata,
                    tags=tags,
                    **kwargs
                )
            else:
                span_context = start_span(
                    obs_name,
                    span_type=span_type,
                    attributes={
                        "input_data": serialize(input_data) if input_data else None,
                        "user_id": user_id,
                        "session_id": session_id,
                        "tags": tags,
                        "metadata": serialize(metadata) if metadata else None,
                        **kwargs
                    }
                )
            
            with span_context as span:
                start_time = datetime.now()
                
                try:
                    # Execute async function
                    result = await func(*args, **func_kwargs)
                    
                    # Handle output capture
                    if capture_output:
                        output_data = result
                        
                        # Handle async generators
                        if inspect.isasyncgen(result):
                            output_data = self._wrap_async_generator(
                                result, 
                                span, 
                                transform_to_string
                            )
                            return output_data
                        
                        # Apply transform if provided
                        if transform_to_string:
                            output_data = transform_to_string(result)
                        
                        # Update span with output
                        if span:
                            self.telemetry_manager.update_span_attributes(
                                span,
                                output_data=serialize(output_data)
                            )
                    
                    # Submit telemetry data
                    self._submit_telemetry(
                        span_name=obs_name,
                        span_type=span_type,
                        input_data=input_data,
                        output_data=result if capture_output else None,
                        start_time=start_time,
                        end_time=datetime.now(),
                        user_id=user_id,
                        session_id=session_id,
                        tags=tags,
                        metadata=metadata,
                        status="success"
                    )
                    
                    return result
                    
                except Exception as e:
                    # Record error in span
                    if span:
                        self.telemetry_manager.record_error(
                            span,
                            e,
                            error_type=type(e).__name__
                        )
                    
                    # Submit error telemetry
                    self._submit_telemetry(
                        span_name=obs_name,
                        span_type=span_type,
                        input_data=input_data,
                        output_data=None,
                        start_time=start_time,
                        end_time=datetime.now(),
                        user_id=user_id,
                        session_id=session_id,
                        tags=tags,
                        metadata=metadata,
                        status="error",
                        error=str(e)
                    )
                    
                    raise
        
        return async_wrapper
    
    def _submit_telemetry(
        self,
        span_name: str,
        span_type: str,
        input_data: Any,
        output_data: Any,
        start_time: datetime,
        end_time: datetime,
        user_id: Optional[str],
        session_id: Optional[str],
        tags: Optional[List[str]],
        metadata: Optional[Dict[str, Any]],
        status: str,
        error: Optional[str] = None
    ) -> None:
        """Submit telemetry data to background processor."""
        if not self.config.telemetry_enabled:
            return
        
        try:
            telemetry_data = {
                "span_name": span_name,
                "span_type": span_type,
                "input_data": serialize(input_data),
                "output_data": serialize(output_data),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_ms": (end_time - start_time).total_seconds() * 1000,
                "user_id": user_id,
                "session_id": session_id,
                "tags": tags,
                "metadata": metadata,
                "status": status,
                "error": error,
                "sdk_version": "0.1.0",
                "project_id": self.config.project_id,
                "environment": self.config.environment,
            }
            
            self.background_processor.submit_telemetry(telemetry_data)
            
        except Exception as e:
            logger.error(f"Failed to submit telemetry: {e}")
    
    def _wrap_sync_generator(
        self,
        generator,
        span,
        transform_to_string: Optional[Callable[[Any], str]]
    ):
        """Wrap synchronous generator to capture output."""
        items = []
        
        try:
            for item in generator:
                items.append(item)
                yield item
        finally:
            # Update span with collected items
            if span:
                output = items
                if transform_to_string:
                    output = transform_to_string(items)
                elif all(isinstance(item, str) for item in items):
                    output = "".join(items)
                
                self.telemetry_manager.update_span_attributes(
                    span,
                    output_data=serialize(output)
                )
    
    async def _wrap_async_generator(
        self,
        generator,
        span,
        transform_to_string: Optional[Callable[[Any], str]]
    ):
        """Wrap asynchronous generator to capture output."""
        items = []
        
        try:
            async for item in generator:
                items.append(item)
                yield item
        finally:
            # Update span with collected items
            if span:
                output = items
                if transform_to_string:
                    output = transform_to_string(items)
                elif all(isinstance(item, str) for item in items):
                    output = "".join(items)
                
                self.telemetry_manager.update_span_attributes(
                    span,
                    output_data=serialize(output)
                )
    
    @staticmethod
    def _is_method(func: Callable) -> bool:
        """Check if function is a method."""
        return (
            "self" in inspect.signature(func).parameters
            or "cls" in inspect.signature(func).parameters
        )
    
    def _get_input_from_func_args(
        self,
        *,
        is_method: bool = False,
        func_args: tuple = (),
        func_kwargs: Dict = {},
    ) -> Dict:
        """Get input data from function arguments."""
        # Remove implicitly passed "self" or "cls" argument
        logged_args = func_args[1:] if is_method else func_args
        
        return {
            "args": logged_args,
            "kwargs": func_kwargs,
        }


# Global decorator instance
_decorator = BrokleDecorator()

# Export the observe decorator
observe = _decorator.observe