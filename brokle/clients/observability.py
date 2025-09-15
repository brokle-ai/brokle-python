"""
Observability client for comprehensive LLM observability via the new observability-service.

This client provides direct integration with the Brokle observability backend,
supporting traces, observations, and quality scores with real-time analytics.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import uuid

from ..types.requests import (
    ObservabilityTraceRequest,
    ObservabilityObservationRequest,
    ObservabilityQualityScoreRequest,
    ObservabilityBatchRequest,
)
from ..types.responses import (
    ObservabilityTraceResponse,
    ObservabilityObservationResponse,
    ObservabilityQualityScoreResponse,
    ObservabilityStatsResponse,
    ObservabilityListResponse,
)

logger = logging.getLogger(__name__)


class ObservabilityClient:
    """Client for comprehensive LLM observability via observability-service."""

    def __init__(self, brokle_client: 'Brokle'):
        self.brokle_client = brokle_client
        self._pending_observations = []
        self._pending_scores = []
        self._batch_size = 50
        self._batch_timeout = 2.0  # seconds

    # Trace Management

    async def create_trace(
        self,
        name: str,
        *,
        external_trace_id: Optional[str] = None,
        project_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        parent_trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ObservabilityTraceResponse:
        """Create a new trace in observability service."""
        if not external_trace_id:
            external_trace_id = str(uuid.uuid4())

        request = ObservabilityTraceRequest(
            project_id=project_id or self.brokle_client.config.project_id,
            external_trace_id=external_trace_id,
            name=name,
            user_id=user_id,
            session_id=session_id,
            parent_trace_id=parent_trace_id,
            metadata=metadata or {},
            tags=tags or {}
        )

        return await self.brokle_client._make_request(
            "POST",
            "/api/v1/observability/traces",
            request.model_dump(exclude_none=True),
            response_model=ObservabilityTraceResponse
        )

    def create_trace_sync(self, name: str, **kwargs) -> ObservabilityTraceResponse:
        """Create trace synchronously."""
        return asyncio.run(self.create_trace(name, **kwargs))

    async def get_trace(self, trace_id: str) -> ObservabilityTraceResponse:
        """Get trace by ID."""
        return await self.brokle_client._make_request(
            "GET",
            f"/api/v1/observability/traces/{trace_id}",
            response_model=ObservabilityTraceResponse
        )

    def get_trace_sync(self, trace_id: str) -> ObservabilityTraceResponse:
        """Get trace synchronously."""
        return asyncio.run(self.get_trace(trace_id))

    async def get_trace_with_observations(self, trace_id: str) -> ObservabilityTraceResponse:
        """Get trace with all its observations."""
        return await self.brokle_client._make_request(
            "GET",
            f"/api/v1/observability/traces/{trace_id}/observations",
            response_model=ObservabilityTraceResponse
        )

    def get_trace_with_observations_sync(self, trace_id: str) -> ObservabilityTraceResponse:
        """Get trace with observations synchronously."""
        return asyncio.run(self.get_trace_with_observations(trace_id))

    async def update_trace(
        self,
        trace_id: str,
        *,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ObservabilityTraceResponse:
        """Update an existing trace."""
        update_data = {}
        if name is not None:
            update_data["name"] = name
        if user_id is not None:
            update_data["user_id"] = user_id
        if metadata is not None:
            update_data["metadata"] = metadata
        if tags is not None:
            update_data["tags"] = tags

        return await self.brokle_client._make_request(
            "PUT",
            f"/api/v1/observability/traces/{trace_id}",
            update_data,
            response_model=ObservabilityTraceResponse
        )

    def update_trace_sync(self, trace_id: str, **kwargs) -> ObservabilityTraceResponse:
        """Update trace synchronously."""
        return asyncio.run(self.update_trace(trace_id, **kwargs))

    async def delete_trace(self, trace_id: str) -> bool:
        """Delete a trace."""
        response = await self.brokle_client._make_request(
            "DELETE",
            f"/api/v1/observability/traces/{trace_id}"
        )
        return response.get("message") == "Trace deleted successfully"

    def delete_trace_sync(self, trace_id: str) -> bool:
        """Delete trace synchronously."""
        return asyncio.run(self.delete_trace(trace_id))

    async def list_traces(
        self,
        *,
        project_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        name: Optional[str] = None,
        external_trace_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: Optional[int] = 50,
        offset: Optional[int] = 0,
        sort_by: Optional[str] = "created_at",
        sort_order: Optional[str] = "desc",
        **kwargs
    ) -> ObservabilityListResponse:
        """List traces with filters."""
        params = {
            "project_id": project_id or self.brokle_client.config.project_id,
            "user_id": user_id,
            "session_id": session_id,
            "name": name,
            "external_trace_id": external_trace_id,
            "start_time": start_time,
            "end_time": end_time,
            "limit": limit,
            "offset": offset,
            "sort_by": sort_by,
            "sort_order": sort_order,
            **kwargs
        }

        return await self.brokle_client._make_request(
            "GET",
            "/api/v1/observability/traces",
            {k: v for k, v in params.items() if v is not None},
            response_model=ObservabilityListResponse
        )

    def list_traces_sync(self, **kwargs) -> ObservabilityListResponse:
        """List traces synchronously."""
        return asyncio.run(self.list_traces(**kwargs))

    async def get_trace_stats(self, trace_id: str) -> ObservabilityStatsResponse:
        """Get comprehensive statistics for a trace."""
        return await self.brokle_client._make_request(
            "GET",
            f"/api/v1/observability/traces/{trace_id}/stats",
            response_model=ObservabilityStatsResponse
        )

    def get_trace_stats_sync(self, trace_id: str) -> ObservabilityStatsResponse:
        """Get trace stats synchronously."""
        return asyncio.run(self.get_trace_stats(trace_id))

    # Observation Management

    async def create_observation(
        self,
        trace_id: str,
        name: str,
        observation_type: str,
        *,
        external_observation_id: Optional[str] = None,
        parent_observation_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        level: Optional[str] = "DEFAULT",
        status_message: Optional[str] = None,
        version: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        model_parameters: Optional[Dict[str, Any]] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        input_cost: Optional[float] = None,
        output_cost: Optional[float] = None,
        total_cost: Optional[float] = None,
        **kwargs
    ) -> ObservabilityObservationResponse:
        """Create a new observation."""
        if not external_observation_id:
            external_observation_id = str(uuid.uuid4())
        if not start_time:
            start_time = datetime.utcnow()

        request = ObservabilityObservationRequest(
            trace_id=trace_id,
            external_observation_id=external_observation_id,
            parent_observation_id=parent_observation_id,
            type=observation_type,
            name=name,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat() if end_time else None,
            level=level,
            status_message=status_message,
            version=version,
            model=model,
            provider=provider,
            input=input_data or {},
            output=output_data or {},
            model_parameters=model_parameters or {},
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=completion_tokens or 0,
            total_tokens=total_tokens or 0,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost
        )

        return await self.brokle_client._make_request(
            "POST",
            "/api/v1/observability/observations",
            request.model_dump(exclude_none=True),
            response_model=ObservabilityObservationResponse
        )

    def create_observation_sync(self, trace_id: str, name: str, observation_type: str, **kwargs) -> ObservabilityObservationResponse:
        """Create observation synchronously."""
        return asyncio.run(self.create_observation(trace_id, name, observation_type, **kwargs))

    async def get_observation(self, observation_id: str) -> ObservabilityObservationResponse:
        """Get observation by ID."""
        return await self.brokle_client._make_request(
            "GET",
            f"/api/v1/observability/observations/{observation_id}",
            response_model=ObservabilityObservationResponse
        )

    def get_observation_sync(self, observation_id: str) -> ObservabilityObservationResponse:
        """Get observation synchronously."""
        return asyncio.run(self.get_observation(observation_id))

    async def complete_observation(
        self,
        observation_id: str,
        *,
        end_time: Optional[datetime] = None,
        output_data: Optional[Dict[str, Any]] = None,
        status_message: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        input_cost: Optional[float] = None,
        output_cost: Optional[float] = None,
        total_cost: Optional[float] = None,
        quality_score: Optional[float] = None,
        **kwargs
    ) -> ObservabilityObservationResponse:
        """Complete an observation with final data."""
        if not end_time:
            end_time = datetime.utcnow()

        completion_data = {
            "end_time": end_time.isoformat(),
            "output": output_data or {},
            "status_message": status_message,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "quality_score": quality_score
        }

        return await self.brokle_client._make_request(
            "POST",
            f"/api/v1/observability/observations/{observation_id}/complete",
            {k: v for k, v in completion_data.items() if v is not None},
            response_model=ObservabilityObservationResponse
        )

    def complete_observation_sync(self, observation_id: str, **kwargs) -> ObservabilityObservationResponse:
        """Complete observation synchronously."""
        return asyncio.run(self.complete_observation(observation_id, **kwargs))

    async def update_observation(
        self,
        observation_id: str,
        *,
        name: Optional[str] = None,
        end_time: Optional[datetime] = None,
        level: Optional[str] = None,
        status_message: Optional[str] = None,
        output_data: Optional[Dict[str, Any]] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        output_cost: Optional[float] = None,
        total_cost: Optional[float] = None,
        **kwargs
    ) -> ObservabilityObservationResponse:
        """Update an existing observation."""
        update_data = {}
        if name is not None:
            update_data["name"] = name
        if end_time is not None:
            update_data["end_time"] = end_time.isoformat()
        if level is not None:
            update_data["level"] = level
        if status_message is not None:
            update_data["status_message"] = status_message
        if output_data is not None:
            update_data["output"] = output_data
        if completion_tokens is not None:
            update_data["completion_tokens"] = completion_tokens
        if total_tokens is not None:
            update_data["total_tokens"] = total_tokens
        if output_cost is not None:
            update_data["output_cost"] = output_cost
        if total_cost is not None:
            update_data["total_cost"] = total_cost

        return await self.brokle_client._make_request(
            "PUT",
            f"/api/v1/observability/observations/{observation_id}",
            update_data,
            response_model=ObservabilityObservationResponse
        )

    def update_observation_sync(self, observation_id: str, **kwargs) -> ObservabilityObservationResponse:
        """Update observation synchronously."""
        return asyncio.run(self.update_observation(observation_id, **kwargs))

    async def delete_observation(self, observation_id: str) -> bool:
        """Delete an observation."""
        response = await self.brokle_client._make_request(
            "DELETE",
            f"/api/v1/observability/observations/{observation_id}"
        )
        return response.get("message") == "Observation deleted successfully"

    def delete_observation_sync(self, observation_id: str) -> bool:
        """Delete observation synchronously."""
        return asyncio.run(self.delete_observation(observation_id))

    async def list_observations(
        self,
        *,
        trace_id: Optional[str] = None,
        observation_type: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: Optional[int] = 50,
        offset: Optional[int] = 0,
        sort_by: Optional[str] = "start_time",
        sort_order: Optional[str] = "desc",
        **kwargs
    ) -> ObservabilityListResponse:
        """List observations with filters."""
        params = {
            "trace_id": trace_id,
            "type": observation_type,
            "provider": provider,
            "model": model,
            "start_time": start_time,
            "end_time": end_time,
            "limit": limit,
            "offset": offset,
            "sort_by": sort_by,
            "sort_order": sort_order,
            **kwargs
        }

        return await self.brokle_client._make_request(
            "GET",
            "/api/v1/observability/observations",
            {k: v for k, v in params.items() if v is not None},
            response_model=ObservabilityListResponse
        )

    def list_observations_sync(self, **kwargs) -> ObservabilityListResponse:
        """List observations synchronously."""
        return asyncio.run(self.list_observations(**kwargs))

    async def get_observations_by_trace(self, trace_id: str) -> ObservabilityListResponse:
        """Get all observations for a specific trace."""
        return await self.brokle_client._make_request(
            "GET",
            f"/api/v1/observability/traces/{trace_id}/observations",
            response_model=ObservabilityListResponse
        )

    def get_observations_by_trace_sync(self, trace_id: str) -> ObservabilityListResponse:
        """Get observations by trace synchronously."""
        return asyncio.run(self.get_observations_by_trace(trace_id))

    # Quality Score Management

    async def create_quality_score(
        self,
        trace_id: str,
        score_name: str,
        data_type: str,
        source: str,
        *,
        observation_id: Optional[str] = None,
        score_value: Optional[float] = None,
        string_value: Optional[str] = None,
        evaluator_name: Optional[str] = None,
        evaluator_version: Optional[str] = None,
        comment: Optional[str] = None,
        author_user_id: Optional[str] = None,
        **kwargs
    ) -> ObservabilityQualityScoreResponse:
        """Create a new quality score."""
        request = ObservabilityQualityScoreRequest(
            trace_id=trace_id,
            observation_id=observation_id,
            score_name=score_name,
            score_value=score_value,
            string_value=string_value,
            data_type=data_type,
            source=source,
            evaluator_name=evaluator_name,
            evaluator_version=evaluator_version,
            comment=comment,
            author_user_id=author_user_id
        )

        return await self.brokle_client._make_request(
            "POST",
            "/api/v1/observability/quality-scores",
            request.model_dump(exclude_none=True),
            response_model=ObservabilityQualityScoreResponse
        )

    def create_quality_score_sync(self, trace_id: str, score_name: str, data_type: str, source: str, **kwargs) -> ObservabilityQualityScoreResponse:
        """Create quality score synchronously."""
        return asyncio.run(self.create_quality_score(trace_id, score_name, data_type, source, **kwargs))

    async def get_quality_score(self, score_id: str) -> ObservabilityQualityScoreResponse:
        """Get quality score by ID."""
        return await self.brokle_client._make_request(
            "GET",
            f"/api/v1/observability/quality-scores/{score_id}",
            response_model=ObservabilityQualityScoreResponse
        )

    def get_quality_score_sync(self, score_id: str) -> ObservabilityQualityScoreResponse:
        """Get quality score synchronously."""
        return asyncio.run(self.get_quality_score(score_id))

    async def update_quality_score(
        self,
        score_id: str,
        *,
        score_value: Optional[float] = None,
        string_value: Optional[str] = None,
        comment: Optional[str] = None,
        evaluator_version: Optional[str] = None,
        **kwargs
    ) -> ObservabilityQualityScoreResponse:
        """Update an existing quality score."""
        update_data = {}
        if score_value is not None:
            update_data["score_value"] = score_value
        if string_value is not None:
            update_data["string_value"] = string_value
        if comment is not None:
            update_data["comment"] = comment
        if evaluator_version is not None:
            update_data["evaluator_version"] = evaluator_version

        return await self.brokle_client._make_request(
            "PUT",
            f"/api/v1/observability/quality-scores/{score_id}",
            update_data,
            response_model=ObservabilityQualityScoreResponse
        )

    def update_quality_score_sync(self, score_id: str, **kwargs) -> ObservabilityQualityScoreResponse:
        """Update quality score synchronously."""
        return asyncio.run(self.update_quality_score(score_id, **kwargs))

    async def delete_quality_score(self, score_id: str) -> bool:
        """Delete a quality score."""
        response = await self.brokle_client._make_request(
            "DELETE",
            f"/api/v1/observability/quality-scores/{score_id}"
        )
        return response.get("message") == "Quality score deleted successfully"

    def delete_quality_score_sync(self, score_id: str) -> bool:
        """Delete quality score synchronously."""
        return asyncio.run(self.delete_quality_score(score_id))

    async def list_quality_scores(
        self,
        *,
        trace_id: Optional[str] = None,
        observation_id: Optional[str] = None,
        score_name: Optional[str] = None,
        source: Optional[str] = None,
        data_type: Optional[str] = None,
        evaluator_name: Optional[str] = None,
        limit: Optional[int] = 50,
        offset: Optional[int] = 0,
        sort_by: Optional[str] = "created_at",
        sort_order: Optional[str] = "desc",
        **kwargs
    ) -> ObservabilityListResponse:
        """List quality scores with filters."""
        params = {
            "trace_id": trace_id,
            "observation_id": observation_id,
            "score_name": score_name,
            "source": source,
            "data_type": data_type,
            "evaluator_name": evaluator_name,
            "limit": limit,
            "offset": offset,
            "sort_by": sort_by,
            "sort_order": sort_order,
            **kwargs
        }

        return await self.brokle_client._make_request(
            "GET",
            "/api/v1/observability/quality-scores",
            {k: v for k, v in params.items() if v is not None},
            response_model=ObservabilityListResponse
        )

    def list_quality_scores_sync(self, **kwargs) -> ObservabilityListResponse:
        """List quality scores synchronously."""
        return asyncio.run(self.list_quality_scores(**kwargs))

    async def get_quality_scores_by_trace(self, trace_id: str) -> ObservabilityListResponse:
        """Get all quality scores for a specific trace."""
        return await self.brokle_client._make_request(
            "GET",
            f"/api/v1/observability/traces/{trace_id}/quality-scores",
            response_model=ObservabilityListResponse
        )

    def get_quality_scores_by_trace_sync(self, trace_id: str) -> ObservabilityListResponse:
        """Get quality scores by trace synchronously."""
        return asyncio.run(self.get_quality_scores_by_trace(trace_id))

    async def get_quality_scores_by_observation(self, observation_id: str) -> ObservabilityListResponse:
        """Get all quality scores for a specific observation."""
        return await self.brokle_client._make_request(
            "GET",
            f"/api/v1/observability/observations/{observation_id}/quality-scores",
            response_model=ObservabilityListResponse
        )

    def get_quality_scores_by_observation_sync(self, observation_id: str) -> ObservabilityListResponse:
        """Get quality scores by observation synchronously."""
        return asyncio.run(self.get_quality_scores_by_observation(observation_id))

    # Batch Operations

    async def create_traces_batch(self, traces: List[Dict[str, Any]]) -> List[ObservabilityTraceResponse]:
        """Create multiple traces in batch."""
        request = ObservabilityBatchRequest(traces=traces)

        response = await self.brokle_client._make_request(
            "POST",
            "/api/v1/observability/traces/batch",
            request.model_dump(exclude_none=True)
        )

        if isinstance(response, dict) and "traces" in response:
            return [ObservabilityTraceResponse(**trace) for trace in response["traces"]]
        return []

    def create_traces_batch_sync(self, traces: List[Dict[str, Any]]) -> List[ObservabilityTraceResponse]:
        """Create traces batch synchronously."""
        return asyncio.run(self.create_traces_batch(traces))

    async def create_observations_batch(self, observations: List[Dict[str, Any]]) -> List[ObservabilityObservationResponse]:
        """Create multiple observations in batch."""
        request = ObservabilityBatchRequest(observations=observations)

        response = await self.brokle_client._make_request(
            "POST",
            "/api/v1/observability/observations/batch",
            request.model_dump(exclude_none=True)
        )

        if isinstance(response, dict) and "observations" in response:
            return [ObservabilityObservationResponse(**obs) for obs in response["observations"]]
        return []

    def create_observations_batch_sync(self, observations: List[Dict[str, Any]]) -> List[ObservabilityObservationResponse]:
        """Create observations batch synchronously."""
        return asyncio.run(self.create_observations_batch(observations))

    # Background processing methods
    def add_observation(self, observation: Dict[str, Any]) -> None:
        """Add observation to pending batch."""
        self._pending_observations.append(observation)

        # Auto-submit if batch is full
        if len(self._pending_observations) >= self._batch_size:
            asyncio.create_task(self._submit_pending_observations())

    def add_quality_score(self, score: Dict[str, Any]) -> None:
        """Add quality score to pending batch."""
        self._pending_scores.append(score)

        # Auto-submit if batch is full
        if len(self._pending_scores) >= self._batch_size:
            asyncio.create_task(self._submit_pending_scores())

    async def _submit_pending_observations(self) -> None:
        """Submit pending observations in background."""
        if not self._pending_observations:
            return

        observations_to_submit = self._pending_observations.copy()
        self._pending_observations.clear()

        try:
            await self.create_observations_batch(observations_to_submit)
        except Exception as e:
            logger.error(f"Failed to submit observations batch: {e}")
            # Re-add observations for retry (simple strategy)
            self._pending_observations.extend(observations_to_submit[:25])  # Limit retry size

    async def _submit_pending_scores(self) -> None:
        """Submit pending quality scores in background."""
        if not self._pending_scores:
            return

        scores_to_submit = self._pending_scores.copy()
        self._pending_scores.clear()

        try:
            # Note: Batch quality score creation would need to be implemented in backend
            for score in scores_to_submit:
                await self.create_quality_score(**score)
        except Exception as e:
            logger.error(f"Failed to submit quality scores: {e}")
            # Re-add scores for retry (simple strategy)
            self._pending_scores.extend(scores_to_submit[:25])  # Limit retry size

    async def flush_pending_data(self) -> None:
        """Flush all pending observations and scores."""
        await asyncio.gather(
            self._submit_pending_observations(),
            self._submit_pending_scores(),
            return_exceptions=True
        )

    def flush_pending_data_sync(self) -> None:
        """Flush pending data synchronously."""
        asyncio.run(self.flush_pending_data())