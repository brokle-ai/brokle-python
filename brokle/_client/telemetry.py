"""
OpenTelemetry integration for Brokle SDK.

This module provides comprehensive OpenTelemetry integration.
but adapted for Brokle's specific features.
"""

import os
import logging
import threading
from contextlib import contextmanager
from typing import Any, Dict, Optional, List, Union, ContextManager
from datetime import datetime

from opentelemetry import trace, context
from opentelemetry.trace import Tracer, Span, Status, StatusCode
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.propagators.b3 import B3MultiFormat
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

# Optional baggage propagator - may not be available in all OpenTelemetry versions
try:
    from opentelemetry.propagators.baggage import BaggagePropagator
    BAGGAGE_PROPAGATOR_AVAILABLE = True
except ImportError:
    BaggagePropagator = None
    BAGGAGE_PROPAGATOR_AVAILABLE = False

from ..config import Config
from ..types.attributes import (
    BrokleOtelSpanAttributes,
    create_trace_attributes,
    create_span_attributes,
    create_generation_attributes,
    create_routing_attributes,
    create_cache_attributes,
    create_evaluation_attributes,
    create_error_attributes,
)
from .serialization import serialize

logger = logging.getLogger(__name__)


class TelemetryManager:
    """Manages OpenTelemetry telemetry for Brokle SDK."""
    
    def __init__(self, config: Config, telemetry_client=None):
        self.config = config
        self._tracer: Optional[Tracer] = None
        self._tracer_provider: Optional[TracerProvider] = None
        self._initialized = False
        self._lock = threading.Lock()
        self._telemetry_client = telemetry_client
        self._pending_spans = []
        self._active_traces = {}
    
    def initialize(self) -> None:
        """Initialize OpenTelemetry components."""
        if self._initialized or not self.config.otel_enabled:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            try:
                # Create resource
                resource = Resource.create({
                    ResourceAttributes.SERVICE_NAME: self.config.otel_service_name,
                    ResourceAttributes.SERVICE_VERSION: "0.1.0",
                    "brokle.sdk.version": "0.1.0",
                    "brokle.sdk.language": "python",
                })
                
                # Create tracer provider
                self._tracer_provider = TracerProvider(resource=resource)
                
                # Set up OTLP exporter if endpoint is configured
                if self.config.otel_endpoint:
                    headers = self.config.otel_headers or {}
                    otlp_exporter = OTLPSpanExporter(
                        endpoint=self.config.otel_endpoint,
                        headers=headers,
                        timeout=30,
                    )
                    
                    # Add batch span processor
                    span_processor = BatchSpanProcessor(
                        otlp_exporter,
                        max_queue_size=2048,
                        max_export_batch_size=512,
                        export_timeout_millis=30000,
                    )
                    self._tracer_provider.add_span_processor(span_processor)
                
                # Set global tracer provider
                trace.set_tracer_provider(self._tracer_provider)
                
                # Set up propagators
                propagators = [
                    TraceContextTextMapPropagator(),
                    B3MultiFormat(),
                ]

                # Add BaggagePropagator if available
                if BAGGAGE_PROPAGATOR_AVAILABLE:
                    propagators.append(BaggagePropagator())

                set_global_textmap(CompositePropagator(propagators))
                
                # Create tracer
                self._tracer = trace.get_tracer("brokle")
                
                self._initialized = True
                logger.info("OpenTelemetry telemetry initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize OpenTelemetry: {e}")
                self._initialized = False
    
    def get_tracer(self) -> Optional[Tracer]:
        """Get OpenTelemetry tracer."""
        if not self._initialized:
            self.initialize()
        return self._tracer
    
    def is_enabled(self) -> bool:
        """Check if telemetry is enabled."""
        return self.config.otel_enabled and self._initialized
    
    @contextmanager
    def start_span(
        self,
        name: str,
        *,
        span_type: str = "span",
        attributes: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ContextManager[Optional[Span]]:
        """Start a new span with Brokle attributes."""
        if not self.is_enabled():
            yield None
            return
        
        tracer = self.get_tracer()
        if not tracer:
            yield None
            return
        
        # Create span attributes
        span_attributes = {}
        if attributes:
            span_attributes.update(attributes)
        
        # Add SDK attributes
        span_attributes.update({
            BrokleOtelSpanAttributes.SDK_VERSION: "0.1.0",
            BrokleOtelSpanAttributes.SDK_LANGUAGE: "python",
            BrokleOtelSpanAttributes.SDK_INTEGRATION_TYPE: span_type,
        })
        
        # Add additional kwargs as attributes
        for key, value in kwargs.items():
            if value is not None:
                attr_key = f"brokle.{key}"
                span_attributes[attr_key] = serialize(value)
        
        with tracer.start_as_current_span(name, attributes=span_attributes) as span:
            try:
                # Send span data to telemetry service if client is available
                if self._telemetry_client and span:
                    self._send_span_to_service(span, span_type, span_attributes)
                
                yield span
            except Exception as e:
                if span:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    
                    # Send error information to telemetry service
                    if self._telemetry_client:
                        self._send_error_to_service(span, e)
                raise
    
    @contextmanager
    def start_trace(
        self,
        name: str,
        *,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> ContextManager[Optional[Span]]:
        """Start a new trace with Brokle trace attributes."""
        trace_attrs = create_trace_attributes(
            name=name,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
            tags=tags,
            **kwargs
        )
        
        with self.start_span(name, span_type="trace", attributes=trace_attrs) as span:
            yield span
    
    @contextmanager
    def start_generation(
        self,
        name: str,
        *,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        input_data: Optional[Any] = None,
        **kwargs
    ) -> ContextManager[Optional[Span]]:
        """Start a new generation span for LLM calls."""
        generation_attrs = create_generation_attributes(
            name=name,
            model=model,
            provider=provider,
            input=input_data,
            **kwargs
        )
        
        with self.start_span(name, span_type="generation", attributes=generation_attrs) as span:
            yield span
    
    def update_span_attributes(self, span: Span, **attributes: Any) -> None:
        """Update span attributes."""
        if not span or not self.is_enabled():
            return
        
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, serialize(value))
    
    def update_generation_attributes(
        self,
        span: Span,
        *,
        output_data: Optional[Any] = None,
        usage_details: Optional[Dict[str, Any]] = None,
        cost_details: Optional[Dict[str, Any]] = None,
        routing_info: Optional[Dict[str, Any]] = None,
        cache_info: Optional[Dict[str, Any]] = None,
        evaluation_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """Update generation span with Brokle specific attributes."""
        if not span or not self.is_enabled():
            return
        
        # Update with generation attributes
        gen_attrs = create_generation_attributes(
            output=output_data,
            usage_details=usage_details,
            cost_details=cost_details,
            **kwargs
        )
        
        for key, value in gen_attrs.items():
            if value is not None:
                span.set_attribute(key, value)
        
        # Update with routing attributes
        if routing_info:
            routing_attrs = create_routing_attributes(**routing_info)
            for key, value in routing_attrs.items():
                if value is not None:
                    span.set_attribute(key, value)
        
        # Update with cache attributes
        if cache_info:
            cache_attrs = create_cache_attributes(**cache_info)
            for key, value in cache_attrs.items():
                if value is not None:
                    span.set_attribute(key, value)
        
        # Update with evaluation attributes
        if evaluation_info:
            eval_attrs = create_evaluation_attributes(**evaluation_info)
            for key, value in eval_attrs.items():
                if value is not None:
                    span.set_attribute(key, value)
    
    def record_error(
        self,
        span: Span,
        error: Exception,
        *,
        error_type: Optional[str] = None,
        error_code: Optional[str] = None,
        provider: Optional[str] = None,
        retryable: Optional[bool] = None,
    ) -> None:
        """Record error information in span."""
        if not span or not self.is_enabled():
            return
        
        # Set span status
        span.set_status(Status(StatusCode.ERROR, str(error)))
        span.record_exception(error)
        
        # Add error attributes
        error_attrs = create_error_attributes(
            error_type=error_type or type(error).__name__,
            error_code=error_code,
            error_message=str(error),
            provider=provider,
            retryable=retryable,
        )
        
        for key, value in error_attrs.items():
            if value is not None:
                span.set_attribute(key, value)
    
    def flush(self) -> None:
        """Flush pending telemetry data."""
        if not self._tracer_provider:
            return
        
        try:
            self._tracer_provider.force_flush(timeout_millis=30000)
        except Exception as e:
            logger.error(f"Failed to flush telemetry: {e}")
    
    def shutdown(self) -> None:
        """Shutdown telemetry system."""
        if not self._tracer_provider:
            return
        
        try:
            # Flush any pending spans to telemetry service
            if self._telemetry_client and self._pending_spans:
                self._flush_pending_spans()
            
            self._tracer_provider.shutdown()
            self._initialized = False
        except Exception as e:
            logger.error(f"Failed to shutdown telemetry: {e}")
    
    def _send_span_to_service(self, span: Span, span_type: str, attributes: Dict[str, Any]) -> None:
        """Send span data to telemetry service in background."""
        if not span or not self._telemetry_client:
            return
        
        try:
            span_context = span.get_span_context()
            trace_id = format(span_context.trace_id, '032x')
            span_id = format(span_context.span_id, '016x')
            
            # Get parent span ID if available
            parent_span_id = None
            if hasattr(span, 'parent') and span.parent:
                parent_span_id = format(span.parent.span_id, '016x')
            
            span_data = {
                "trace_id": trace_id,
                "span_id": span_id,
                "parent_span_id": parent_span_id,
                "name": span.name,
                "span_type": span_type,
                "start_time": datetime.now().isoformat(),
                "attributes": attributes,
                "status": {"code": "OK"}
            }
            
            # Add to pending spans for batch processing
            self._pending_spans.append(span_data)
            
            # Flush if batch size reached
            if len(self._pending_spans) >= 50:
                self._flush_pending_spans()
                
        except Exception as e:
            logger.error(f"Failed to send span to service: {e}")
    
    def _send_error_to_service(self, span: Span, error: Exception) -> None:
        """Send error information to telemetry service."""
        if not span or not self._telemetry_client:
            return
        
        try:
            span_context = span.get_span_context()
            span_id = format(span_context.span_id, '016x')
            
            error_event = {
                "span_id": span_id,
                "event_type": "error",
                "timestamp": datetime.now().isoformat(),
                "attributes": {
                    "error.type": type(error).__name__,
                    "error.message": str(error),
                    "error.stack": getattr(error, '__traceback__', None)
                }
            }
            
            # Add error event to telemetry client
            self._telemetry_client.add_event(error_event)
            
        except Exception as e:
            logger.error(f"Failed to send error to service: {e}")
    
    def _flush_pending_spans(self) -> None:
        """Flush pending spans to telemetry service."""
        if not self._pending_spans or not self._telemetry_client:
            return
        
        try:
            spans_to_send = self._pending_spans.copy()
            self._pending_spans.clear()
            
            # Send spans in background task
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._telemetry_client.create_spans_batch(spans_to_send))
                else:
                    loop.run_until_complete(self._telemetry_client.create_spans_batch(spans_to_send))
            except RuntimeError:
                # No event loop, create new one
                asyncio.run(self._telemetry_client.create_spans_batch(spans_to_send))
                
        except Exception as e:
            logger.error(f"Failed to flush spans to service: {e}")
    
    def set_telemetry_client(self, client) -> None:
        """Set the telemetry service client."""
        self._telemetry_client = client


# Global telemetry manager instance
_telemetry_manager: Optional[TelemetryManager] = None
_lock = threading.Lock()


def get_telemetry_manager(config: Optional[Config] = None, telemetry_client=None) -> TelemetryManager:
    """Get or create telemetry manager."""
    global _telemetry_manager
    
    with _lock:
        if _telemetry_manager is None:
            if config is None:
                from ..config import get_config
                config = get_config()
            _telemetry_manager = TelemetryManager(config, telemetry_client)
            _telemetry_manager.initialize()
        elif telemetry_client and not _telemetry_manager._telemetry_client:
            # Set telemetry client if not already set
            _telemetry_manager.set_telemetry_client(telemetry_client)
        
        return _telemetry_manager


def reset_telemetry_manager() -> None:
    """Reset telemetry manager (for testing)."""
    global _telemetry_manager
    
    with _lock:
        if _telemetry_manager:
            _telemetry_manager.shutdown()
        _telemetry_manager = None


# Convenience functions
def get_tracer() -> Optional[Tracer]:
    """Get OpenTelemetry tracer."""
    return get_telemetry_manager().get_tracer()


def is_telemetry_enabled() -> bool:
    """Check if telemetry is enabled."""
    return get_telemetry_manager().is_enabled()


def start_span(name: str, **kwargs) -> ContextManager[Optional[Span]]:
    """Start a new span."""
    return get_telemetry_manager().start_span(name, **kwargs)


def start_trace(name: str, **kwargs) -> ContextManager[Optional[Span]]:
    """Start a new trace."""
    return get_telemetry_manager().start_trace(name, **kwargs)


def start_generation(name: str, **kwargs) -> ContextManager[Optional[Span]]:
    """Start a new generation span."""
    return get_telemetry_manager().start_generation(name, **kwargs)


def flush_telemetry() -> None:
    """Flush pending telemetry data."""
    get_telemetry_manager().flush()


def shutdown_telemetry() -> None:
    """Shutdown telemetry system."""
    get_telemetry_manager().shutdown()