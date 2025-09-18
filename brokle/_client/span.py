"""
Brokle span classes.

This module provides span hierarchy for different types of operations:
- BrokleSpan: General purpose spans
- BrokleGeneration: LLM generation spans with enhanced attributes
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from datetime import datetime

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from opentelemetry.util._decorator import _AgnosticContextManager, _agnosticcontextmanager

from .attributes import BrokleOtelSpanAttributes

if TYPE_CHECKING:
    from .client import Brokle

logger = logging.getLogger(__name__)


class BrokleSpan:
    """
    Brokle span implementation.

    This class wraps OTEL spans with Brokle-specific functionality
    and provides a clean API.
    """

    def __init__(
        self,
        client: 'Brokle',
        name: str,
        *,
        trace_id: Optional[str] = None,
        parent_observation_id: Optional[str] = None,
        level: str = "DEFAULT",
        status_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        **kwargs
    ):
        self.client = client
        self.name = name
        self.trace_id = trace_id or str(uuid.uuid4())
        self.parent_observation_id = parent_observation_id
        self.level = level
        self.status_message = status_message
        self.metadata = metadata or {}
        self.tags = tags or []

        # OTEL span
        self._otel_span: Optional[trace.Span] = None
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None

    def __enter__(self) -> 'BrokleSpan':
        """Context manager entry."""
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if exc_type is not None:
            self.end(status=StatusCode.ERROR, status_message=str(exc_val))
        else:
            self.end()

    def start(self) -> 'BrokleSpan':
        """Start the span and create OTEL span."""
        try:
            self._start_time = datetime.utcnow()

            # Create OTEL span
            self._otel_span = self.client.tracer.start_span(self.name)

            # Set Brokle-specific attributes
            if self._otel_span:
                self._otel_span.set_attribute(BrokleOtelSpanAttributes.SPAN_TYPE, "span")
                self._otel_span.set_attribute(BrokleOtelSpanAttributes.SPAN_NAME, self.name)
                self._otel_span.set_attribute(BrokleOtelSpanAttributes.SPAN_LEVEL, self.level)

                if self.trace_id:
                    self._otel_span.set_attribute(BrokleOtelSpanAttributes.TRACE_NAME, self.trace_id)

                if self.parent_observation_id:
                    self._otel_span.set_attribute("brokle.span.parent_observation_id", self.parent_observation_id)

                if self.metadata:
                    self._otel_span.set_attribute(BrokleOtelSpanAttributes.SPAN_METADATA, str(self.metadata))

                if self.tags:
                    self._otel_span.set_attribute("brokle.span.tags", str(self.tags))

                if self.status_message:
                    self._otel_span.set_attribute(BrokleOtelSpanAttributes.SPAN_STATUS_MESSAGE, self.status_message)

        except Exception as e:
            logger.error(f"Failed to start Brokle span: {e}")

        return self

    def end(
        self,
        *,
        status: StatusCode = StatusCode.OK,
        status_message: Optional[str] = None,
        **kwargs
    ) -> None:
        """End the span."""
        try:
            self._end_time = datetime.utcnow()

            if self._otel_span:
                # Set status
                self._otel_span.set_status(Status(status, status_message))

                # Calculate duration
                if self._start_time and self._end_time:
                    duration_ms = int((self._end_time - self._start_time).total_seconds() * 1000)
                    self._otel_span.set_attribute("brokle.span.duration_ms", duration_ms)

                # End the OTEL span
                self._otel_span.end()

        except Exception as e:
            logger.error(f"Failed to end Brokle span: {e}")

    def update(
        self,
        *,
        level: Optional[str] = None,
        status_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> 'BrokleSpan':
        """Update span attributes."""
        try:
            if level is not None:
                self.level = level
                if self._otel_span:
                    self._otel_span.set_attribute(BrokleOtelSpanAttributes.SPAN_LEVEL, level)

            if status_message is not None:
                self.status_message = status_message
                if self._otel_span:
                    self._otel_span.set_attribute(BrokleOtelSpanAttributes.SPAN_STATUS_MESSAGE, status_message)

            if metadata is not None:
                self.metadata.update(metadata)
                if self._otel_span:
                    self._otel_span.set_attribute(BrokleOtelSpanAttributes.SPAN_METADATA, str(self.metadata))

            if tags is not None:
                self.tags.extend(tags)
                if self._otel_span:
                    self._otel_span.set_attribute("brokle.span.tags", str(self.tags))

        except Exception as e:
            logger.error(f"Failed to update Brokle span: {e}")

        return self

    def set_attribute(self, key: str, value: Any) -> 'BrokleSpan':
        """Set a custom attribute on the span."""
        try:
            if self._otel_span:
                self._otel_span.set_attribute(key, str(value))
        except Exception as e:
            logger.error(f"Failed to set span attribute {key}: {e}")

        return self


class BrokleGeneration(BrokleSpan):
    """
    Brokle generation span for LLM operations.

    This class extends BrokleSpan with LLM-specific functionality
    and attributes for comprehensive LLM observability.
    """

    def __init__(
        self,
        client: 'Brokle',
        name: str,
        *,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        model_parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(client, name, **kwargs)
        self.model = model
        self.provider = provider
        self.model_parameters = model_parameters or {}

        # LLM-specific attributes
        self.input_tokens: Optional[int] = None
        self.output_tokens: Optional[int] = None
        self.total_tokens: Optional[int] = None
        self.cost_usd: Optional[float] = None
        self.latency_ms: Optional[int] = None

        # Brokle AI platform attributes
        self.routing_strategy: Optional[str] = None
        self.routing_decision: Optional[str] = None
        self.cache_hit: Optional[bool] = None
        self.cache_similarity_score: Optional[float] = None
        self.quality_score: Optional[float] = None

    def start(self) -> 'BrokleGeneration':
        """Start the generation span with LLM-specific attributes."""
        super().start()

        try:
            if self._otel_span:
                # Set generation type
                self._otel_span.set_attribute(BrokleOtelSpanAttributes.SPAN_TYPE, "generation")
                self._otel_span.set_attribute(BrokleOtelSpanAttributes.GENERATION_TYPE, "llm")

                # Set LLM attributes
                if self.model:
                    self._otel_span.set_attribute(BrokleOtelSpanAttributes.GENERATION_MODEL, self.model)

                if self.provider:
                    self._otel_span.set_attribute(BrokleOtelSpanAttributes.GENERATION_PROVIDER, self.provider)

                if self.model_parameters:
                    self._otel_span.set_attribute(BrokleOtelSpanAttributes.GENERATION_MODEL_PARAMETERS, str(self.model_parameters))

        except Exception as e:
            logger.error(f"Failed to start Brokle generation: {e}")

        return self

    def update_metrics(
        self,
        *,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        cost_usd: Optional[float] = None,
        latency_ms: Optional[int] = None,
        **kwargs
    ) -> 'BrokleGeneration':
        """Update LLM metrics on the generation span."""
        try:
            if input_tokens is not None:
                self.input_tokens = input_tokens
                if self._otel_span:
                    self._otel_span.set_attribute(BrokleOtelSpanAttributes.TOKENS_INPUT, input_tokens)

            if output_tokens is not None:
                self.output_tokens = output_tokens
                if self._otel_span:
                    self._otel_span.set_attribute(BrokleOtelSpanAttributes.TOKENS_OUTPUT, output_tokens)

            if total_tokens is not None:
                self.total_tokens = total_tokens
                if self._otel_span:
                    self._otel_span.set_attribute(BrokleOtelSpanAttributes.TOKENS_TOTAL, total_tokens)

            if cost_usd is not None:
                self.cost_usd = cost_usd
                if self._otel_span:
                    self._otel_span.set_attribute(BrokleOtelSpanAttributes.COST_USD, cost_usd)

            if latency_ms is not None:
                self.latency_ms = latency_ms
                if self._otel_span:
                    self._otel_span.set_attribute(BrokleOtelSpanAttributes.LATENCY_MS, latency_ms)

        except Exception as e:
            logger.error(f"Failed to update generation metrics: {e}")

        return self

    def update_brokle_metrics(
        self,
        *,
        routing_strategy: Optional[str] = None,
        routing_decision: Optional[str] = None,
        cache_hit: Optional[bool] = None,
        cache_similarity_score: Optional[float] = None,
        quality_score: Optional[float] = None,
        **kwargs
    ) -> 'BrokleGeneration':
        """Update Brokle AI platform specific metrics."""
        try:
            if routing_strategy is not None:
                self.routing_strategy = routing_strategy
                if self._otel_span:
                    self._otel_span.set_attribute(BrokleOtelSpanAttributes.ROUTING_STRATEGY, routing_strategy)

            if routing_decision is not None:
                self.routing_decision = routing_decision
                if self._otel_span:
                    self._otel_span.set_attribute(BrokleOtelSpanAttributes.ROUTING_DECISION, routing_decision)

            if cache_hit is not None:
                self.cache_hit = cache_hit
                if self._otel_span:
                    self._otel_span.set_attribute(BrokleOtelSpanAttributes.CACHE_HIT, cache_hit)

            if cache_similarity_score is not None:
                self.cache_similarity_score = cache_similarity_score
                if self._otel_span:
                    self._otel_span.set_attribute(BrokleOtelSpanAttributes.CACHE_SIMILARITY_SCORE, cache_similarity_score)

            if quality_score is not None:
                self.quality_score = quality_score
                if self._otel_span:
                    self._otel_span.set_attribute(BrokleOtelSpanAttributes.EVALUATION_QUALITY_SCORE, quality_score)

        except Exception as e:
            logger.error(f"Failed to update Brokle metrics: {e}")

        return self