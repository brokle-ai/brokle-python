"""
Telemetry client for comprehensive observability via telemetry-service.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from ..types.requests import (
    TelemetryTraceRequest,
    TelemetrySpanRequest, 
    TelemetryEventBatchRequest,
)
from ..types.responses import (
    TelemetryTraceResponse,
    TelemetrySpanResponse,
    TelemetryEventBatchResponse,
)

logger = logging.getLogger(__name__)


class TelemetryClient:
    """Client for comprehensive telemetry and observability via telemetry-service."""
    
    def __init__(self, brokle_client: 'Brokle'):
        self.brokle_client = brokle_client
        self._pending_events = []
        self._batch_size = 100
        self._batch_timeout = 5.0  # seconds
    
    async def create_trace(
        self,
        name: str,
        *,
        trace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> TelemetryTraceResponse:
        """Create a new trace in telemetry service."""
        request = TelemetryTraceRequest(
            trace_id=trace_id,
            name=name,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
            tags=tags
        )
        
        return await self.brokle_client._make_request(
            "POST",
            "/api/v1/telemetry/traces",
            request.model_dump(exclude_none=True),
            response_model=TelemetryTraceResponse
        )
    
    def create_trace_sync(self, name: str, **kwargs) -> TelemetryTraceResponse:
        """Create trace synchronously."""
        return asyncio.run(self.create_trace(name, **kwargs))
    
    async def get_trace(self, trace_id: str) -> TelemetryTraceResponse:
        """Get trace by ID."""
        return await self.brokle_client._make_request(
            "GET",
            f"/api/v1/telemetry/traces/{trace_id}",
            response_model=TelemetryTraceResponse
        )
    
    def get_trace_sync(self, trace_id: str) -> TelemetryTraceResponse:
        """Get trace synchronously."""
        return asyncio.run(self.get_trace(trace_id))
    
    async def end_trace(self, trace_id: str, **metadata) -> TelemetryTraceResponse:
        """End a trace."""
        return await self.brokle_client._make_request(
            "PUT",
            f"/api/v1/telemetry/traces/{trace_id}/end",
            metadata,
            response_model=TelemetryTraceResponse
        )
    
    def end_trace_sync(self, trace_id: str, **metadata) -> TelemetryTraceResponse:
        """End trace synchronously."""
        return asyncio.run(self.end_trace(trace_id, **metadata))
    
    async def list_traces(
        self,
        *,
        limit: Optional[int] = None,
        offset: Optional[int] = None, 
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        **kwargs
    ) -> List[TelemetryTraceResponse]:
        """List traces with filters."""
        params = {
            "limit": limit,
            "offset": offset,
            "user_id": user_id,
            "session_id": session_id,
            "start_time": start_time,
            "end_time": end_time,
            **kwargs
        }
        
        response = await self.brokle_client._make_request(
            "GET",
            "/api/v1/telemetry/traces",
            {k: v for k, v in params.items() if v is not None}
        )
        
        if isinstance(response, dict) and "traces" in response:
            return [TelemetryTraceResponse(**trace) for trace in response["traces"]]
        return []
    
    def list_traces_sync(self, **kwargs) -> List[TelemetryTraceResponse]:
        """List traces synchronously."""
        return asyncio.run(self.list_traces(**kwargs))
    
    async def create_span(
        self,
        trace_id: str,
        name: str,
        *,
        span_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        span_type: str = "span",
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        events: Optional[List[Dict[str, Any]]] = None,
        status: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> TelemetrySpanResponse:
        """Create a new span."""
        request = TelemetrySpanRequest(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            name=name,
            span_type=span_type,
            start_time=start_time,
            end_time=end_time,
            attributes=attributes,
            events=events,
            status=status
        )
        
        return await self.brokle_client._make_request(
            "POST",
            "/api/v1/telemetry/spans",
            request.model_dump(exclude_none=True),
            response_model=TelemetrySpanResponse
        )
    
    def create_span_sync(self, trace_id: str, name: str, **kwargs) -> TelemetrySpanResponse:
        """Create span synchronously."""
        return asyncio.run(self.create_span(trace_id, name, **kwargs))
    
    async def get_span(self, span_id: str) -> TelemetrySpanResponse:
        """Get span by ID."""
        return await self.brokle_client._make_request(
            "GET",
            f"/api/v1/telemetry/spans/{span_id}",
            response_model=TelemetrySpanResponse
        )
    
    def get_span_sync(self, span_id: str) -> TelemetrySpanResponse:
        """Get span synchronously."""
        return asyncio.run(self.get_span(span_id))
    
    async def end_span(self, span_id: str, **attributes) -> TelemetrySpanResponse:
        """End a span."""
        return await self.brokle_client._make_request(
            "PUT",
            f"/api/v1/telemetry/spans/{span_id}/end",
            attributes,
            response_model=TelemetrySpanResponse
        )
    
    def end_span_sync(self, span_id: str, **attributes) -> TelemetrySpanResponse:
        """End span synchronously."""
        return asyncio.run(self.end_span(span_id, **attributes))
    
    async def list_spans(
        self,
        *,
        trace_id: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        **kwargs
    ) -> List[TelemetrySpanResponse]:
        """List spans with filters."""
        params = {
            "trace_id": trace_id,
            "limit": limit,
            "offset": offset,
            **kwargs
        }
        
        response = await self.brokle_client._make_request(
            "GET",
            "/api/v1/telemetry/spans",
            {k: v for k, v in params.items() if v is not None}
        )
        
        if isinstance(response, dict) and "spans" in response:
            return [TelemetrySpanResponse(**span) for span in response["spans"]]
        return []
    
    def list_spans_sync(self, **kwargs) -> List[TelemetrySpanResponse]:
        """List spans synchronously."""
        return asyncio.run(self.list_spans(**kwargs))
    
    async def create_spans_batch(
        self,
        spans: List[Dict[str, Any]]
    ) -> List[TelemetrySpanResponse]:
        """Create multiple spans in batch."""
        response = await self.brokle_client._make_request(
            "POST",
            "/api/v1/telemetry/spans/batch",
            {"spans": spans}
        )
        
        if isinstance(response, dict) and "spans" in response:
            return [TelemetrySpanResponse(**span) for span in response["spans"]]
        return []
    
    def create_spans_batch_sync(self, spans: List[Dict[str, Any]]) -> List[TelemetrySpanResponse]:
        """Create spans batch synchronously."""
        return asyncio.run(self.create_spans_batch(spans))
    
    async def submit_events_batch(
        self,
        events: List[Dict[str, Any]],
        *,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TelemetryEventBatchResponse:
        """Submit telemetry events in batch."""
        request = TelemetryEventBatchRequest(
            events=events,
            metadata=metadata
        )
        
        return await self.brokle_client._make_request(
            "POST",
            "/api/v1/telemetry/events/batch",
            request.model_dump(exclude_none=True),
            response_model=TelemetryEventBatchResponse
        )
    
    def submit_events_batch_sync(
        self,
        events: List[Dict[str, Any]],
        **kwargs
    ) -> TelemetryEventBatchResponse:
        """Submit events batch synchronously."""
        return asyncio.run(self.submit_events_batch(events, **kwargs))
    
    async def ingest_otel_traces(self, traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ingest OpenTelemetry traces."""
        return await self.brokle_client._make_request(
            "POST",
            "/api/v1/otel/traces",
            {"traces": traces}
        )
    
    def ingest_otel_traces_sync(self, traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ingest OTel traces synchronously."""
        return asyncio.run(self.ingest_otel_traces(traces))
    
    async def ingest_otel_spans(self, spans: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ingest OpenTelemetry spans."""
        return await self.brokle_client._make_request(
            "POST",
            "/api/v1/otel/spans",
            {"spans": spans}
        )
    
    def ingest_otel_spans_sync(self, spans: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ingest OTel spans synchronously."""
        return asyncio.run(self.ingest_otel_spans(spans))
    
    async def get_otel_config(self) -> Dict[str, Any]:
        """Get OpenTelemetry configuration."""
        return await self.brokle_client._make_request(
            "GET",
            "/api/v1/otel/config"
        )
    
    def get_otel_config_sync(self) -> Dict[str, Any]:
        """Get OTel config synchronously."""
        return asyncio.run(self.get_otel_config())
    
    # Background processing methods
    def add_event(self, event: Dict[str, Any]) -> None:
        """Add event to pending batch."""
        self._pending_events.append(event)
        
        # Auto-submit if batch is full
        if len(self._pending_events) >= self._batch_size:
            asyncio.create_task(self._submit_pending_events())
    
    async def _submit_pending_events(self) -> None:
        """Submit pending events in background."""
        if not self._pending_events:
            return
        
        events_to_submit = self._pending_events.copy()
        self._pending_events.clear()
        
        try:
            await self.submit_events_batch(events_to_submit)
        except Exception as e:
            logger.error(f"Failed to submit telemetry events: {e}")
            # Re-add events for retry (simple strategy)
            self._pending_events.extend(events_to_submit[:50])  # Limit retry size
    
    async def flush_pending_events(self) -> None:
        """Flush all pending events."""
        await self._submit_pending_events()
    
    def flush_pending_events_sync(self) -> None:
        """Flush pending events synchronously."""
        asyncio.run(self.flush_pending_events())