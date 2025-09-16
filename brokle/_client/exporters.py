"""
Custom OpenTelemetry exporters for Brokle Platform.

This module provides custom OTEL exporters that convert OTEL spans
into Brokle API calls, bridging OTEL standard with Brokle backend.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Sequence
from datetime import datetime

import httpx
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import Span as OTELSpan

from .attributes import BrokleOtelSpanAttributes

logger = logging.getLogger(__name__)


class BrokleSpanExporter(SpanExporter):
    """
    Custom OTEL span exporter that sends spans to Brokle API.

    This exporter converts OTEL spans to Brokle API format and sends them
    to the appropriate Brokle observability endpoints.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        public_key: str,
        timeout: float = 30.0,
        session: Optional[httpx.AsyncClient] = None
    ):
        self.endpoint = endpoint.rstrip('/')
        self.api_key = api_key
        self.public_key = public_key
        self.timeout = timeout
        self._session = session or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )

    def export(self, spans: Sequence[OTELSpan]) -> SpanExportResult:
        """Export spans to Brokle API."""
        try:
            # Convert spans to Brokle format
            traces, observations = self._convert_spans(spans)

            # Send to Brokle API (run async in sync context)
            asyncio.run(self._send_to_brokle(traces, observations))

            return SpanExportResult.SUCCESS

        except Exception as e:
            logger.error(f"Failed to export spans to Brokle: {e}")
            return SpanExportResult.FAILURE

    def _convert_spans(self, spans: Sequence[OTELSpan]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Convert OTEL spans to Brokle traces and observations."""
        traces = []
        observations = []

        # Group spans by trace
        trace_groups = {}
        for span in spans:
            trace_id = span.get_span_context().trace_id
            if trace_id not in trace_groups:
                trace_groups[trace_id] = []
            trace_groups[trace_id].append(span)

        # Convert each trace group
        for trace_id, trace_spans in trace_groups.items():
            # Create trace from first span or root span
            root_span = self._find_root_span(trace_spans)
            if root_span:
                trace = self._span_to_trace(root_span, trace_id)
                traces.append(trace)

            # Convert all spans to observations
            for span in trace_spans:
                observation = self._span_to_observation(span, trace_id)
                observations.append(observation)

        return traces, observations

    def _find_root_span(self, spans: List[OTELSpan]) -> Optional[OTELSpan]:
        """Find the root span (span without parent) in a trace."""
        for span in spans:
            if span.parent is None:
                return span
        return spans[0] if spans else None

    def _span_to_trace(self, span: OTELSpan, trace_id: int) -> Dict[str, Any]:
        """Convert OTEL span to Brokle trace format."""
        attributes = span.attributes or {}

        trace = {
            "external_trace_id": str(trace_id),
            "name": attributes.get(BrokleOtelSpanAttributes.TRACE_NAME, span.name),
            "metadata": self._parse_metadata(attributes.get(BrokleOtelSpanAttributes.TRACE_METADATA)),
            "tags": self._parse_tags(attributes.get("brokle.span.tags")),
        }

        # Add optional fields
        if BrokleOtelSpanAttributes.TRACE_USER_ID in attributes:
            trace["user_id"] = attributes[BrokleOtelSpanAttributes.TRACE_USER_ID]

        if "session.id" in attributes:
            trace["session_id"] = attributes["session.id"]

        if BrokleOtelSpanAttributes.TRACE_PROJECT_ID in attributes:
            trace["project_id"] = attributes[BrokleOtelSpanAttributes.TRACE_PROJECT_ID]

        return trace

    def _span_to_observation(self, span: OTELSpan, trace_id: int) -> Dict[str, Any]:
        """Convert OTEL span to Brokle observation format."""
        attributes = span.attributes or {}
        span_type = attributes.get(BrokleOtelSpanAttributes.SPAN_TYPE, "span")

        observation = {
            "external_trace_id": str(trace_id),
            "external_observation_id": str(span.get_span_context().span_id),
            "name": span.name,
            "type": span_type,
            "start_time": self._ns_to_datetime(span.start_time),
            "end_time": self._ns_to_datetime(span.end_time) if span.end_time else None,
            "metadata": self._parse_metadata(attributes.get(BrokleOtelSpanAttributes.SPAN_METADATA)),
            "tags": self._parse_tags(attributes.get("brokle.span.tags")),
        }

        # Add LLM-specific attributes for generation spans
        if span_type == "generation":
            self._add_llm_attributes(observation, attributes)

        # Add Brokle-specific attributes
        self._add_brokle_attributes(observation, attributes)

        # Add input/output if available
        if BrokleOtelSpanAttributes.SPAN_INPUT in attributes:
            observation["input_data"] = self._parse_json(attributes[BrokleOtelSpanAttributes.SPAN_INPUT])

        if BrokleOtelSpanAttributes.SPAN_OUTPUT in attributes:
            observation["output_data"] = self._parse_json(attributes[BrokleOtelSpanAttributes.SPAN_OUTPUT])

        # Add status
        if span.status:
            observation["status_message"] = span.status.description or "completed"

        return observation

    def _add_llm_attributes(self, observation: Dict[str, Any], attributes: Dict[str, Any]) -> None:
        """Add LLM-specific attributes to observation."""
        # Model information
        if BrokleOtelSpanAttributes.GENERATION_MODEL in attributes:
            observation["model"] = attributes[BrokleOtelSpanAttributes.GENERATION_MODEL]

        if BrokleOtelSpanAttributes.GENERATION_PROVIDER in attributes:
            observation["provider"] = attributes[BrokleOtelSpanAttributes.GENERATION_PROVIDER]

        # Token usage
        usage = {}
        if BrokleOtelSpanAttributes.TOKENS_INPUT in attributes:
            usage["input_tokens"] = int(attributes[BrokleOtelSpanAttributes.TOKENS_INPUT])

        if BrokleOtelSpanAttributes.TOKENS_OUTPUT in attributes:
            usage["output_tokens"] = int(attributes[BrokleOtelSpanAttributes.TOKENS_OUTPUT])

        if BrokleOtelSpanAttributes.TOKENS_TOTAL in attributes:
            usage["total_tokens"] = int(attributes[BrokleOtelSpanAttributes.TOKENS_TOTAL])

        if usage:
            observation["usage"] = usage

        # Cost information
        if BrokleOtelSpanAttributes.COST_USD in attributes:
            observation["cost_usd"] = float(attributes[BrokleOtelSpanAttributes.COST_USD])

        # Latency
        if BrokleOtelSpanAttributes.LATENCY_MS in attributes:
            observation["latency_ms"] = int(attributes[BrokleOtelSpanAttributes.LATENCY_MS])

    def _add_brokle_attributes(self, observation: Dict[str, Any], attributes: Dict[str, Any]) -> None:
        """Add Brokle AI platform specific attributes."""
        brokle_metadata = observation.get("metadata", {})

        # Routing information
        if BrokleOtelSpanAttributes.ROUTING_STRATEGY in attributes:
            brokle_metadata["routing_strategy"] = attributes[BrokleOtelSpanAttributes.ROUTING_STRATEGY]

        if BrokleOtelSpanAttributes.ROUTING_DECISION in attributes:
            brokle_metadata["routing_decision"] = attributes[BrokleOtelSpanAttributes.ROUTING_DECISION]

        # Caching information
        if BrokleOtelSpanAttributes.CACHE_HIT in attributes:
            brokle_metadata["cache_hit"] = self._parse_bool(attributes[BrokleOtelSpanAttributes.CACHE_HIT])

        if BrokleOtelSpanAttributes.CACHE_SIMILARITY_SCORE in attributes:
            brokle_metadata["cache_similarity_score"] = float(attributes[BrokleOtelSpanAttributes.CACHE_SIMILARITY_SCORE])

        # Quality scoring
        if BrokleOtelSpanAttributes.EVALUATION_QUALITY_SCORE in attributes:
            brokle_metadata["quality_score"] = float(attributes[BrokleOtelSpanAttributes.EVALUATION_QUALITY_SCORE])

        observation["metadata"] = brokle_metadata

    async def _send_to_brokle(self, traces: List[Dict[str, Any]], observations: List[Dict[str, Any]]) -> None:
        """Send traces and observations to Brokle API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-Public-Key": self.public_key,
            "Content-Type": "application/json",
            "User-Agent": "brokle-python-sdk/0.1.0",
        }

        try:
            # Send traces first
            if traces:
                await self._send_traces(traces, headers)

            # Then send observations
            if observations:
                await self._send_observations(observations, headers)

        except Exception as e:
            logger.error(f"Failed to send data to Brokle API: {e}")
            raise

    async def _send_traces(self, traces: List[Dict[str, Any]], headers: Dict[str, str]) -> None:
        """Send traces to Brokle API."""
        url = f"{self.endpoint}/api/v1/observability/traces/batch"
        data = {"traces": traces}

        response = await self._session.post(url, json=data, headers=headers)
        response.raise_for_status()

    async def _send_observations(self, observations: List[Dict[str, Any]], headers: Dict[str, str]) -> None:
        """Send observations to Brokle API."""
        # Group observations by trace for batch processing
        trace_groups = {}
        for obs in observations:
            trace_id = obs["external_trace_id"]
            if trace_id not in trace_groups:
                trace_groups[trace_id] = []
            trace_groups[trace_id].append(obs)

        # Send each group
        for trace_id, trace_observations in trace_groups.items():
            url = f"{self.endpoint}/api/v1/observability/observations/batch"
            data = {"observations": trace_observations}

            response = await self._session.post(url, json=data, headers=headers)
            response.raise_for_status()

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        if self._session:
            asyncio.run(self._session.aclose())

    # Utility methods

    def _ns_to_datetime(self, nanoseconds: int) -> str:
        """Convert nanoseconds to ISO datetime string."""
        seconds = nanoseconds / 1_000_000_000
        dt = datetime.fromtimestamp(seconds)
        return dt.isoformat()

    def _parse_metadata(self, metadata_str: Optional[str]) -> Dict[str, Any]:
        """Parse metadata string to dict."""
        if not metadata_str:
            return {}
        try:
            return json.loads(metadata_str)
        except (json.JSONDecodeError, TypeError):
            return {}

    def _parse_tags(self, tags_str: Optional[str]) -> List[str]:
        """Parse tags string to list."""
        if not tags_str:
            return []
        try:
            return json.loads(tags_str)
        except (json.JSONDecodeError, TypeError):
            return []

    def _parse_json(self, json_str: Optional[str]) -> Any:
        """Parse JSON string."""
        if not json_str:
            return None
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return str(json_str)

    def _parse_bool(self, bool_str: Any) -> bool:
        """Parse boolean value."""
        if isinstance(bool_str, bool):
            return bool_str
        if isinstance(bool_str, str):
            return bool_str.lower() in ('true', '1', 'yes', 'on')
        return bool(bool_str)