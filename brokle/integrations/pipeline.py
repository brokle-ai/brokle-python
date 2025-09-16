"""
Async processing pipeline for Brokle SDK.

Provides enterprise-grade async processing capabilities with
stages, middleware, and comprehensive monitoring.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union, AsyncIterator, Type
from concurrent.futures import ThreadPoolExecutor

from .events import Event, EventBus, EventPriority, publish_event
from .._task_manager.queue import Task, TaskQueue, TaskPriority, get_queue_manager
from .._task_manager.workers import WorkerPool, ThreadWorker, AsyncWorker

logger = logging.getLogger(__name__)


class PipelineStageType(Enum):
    """Types of pipeline stages."""
    PREPROCESSOR = "preprocessor"
    PROCESSOR = "processor"
    POSTPROCESSOR = "postprocessor"
    VALIDATOR = "validator"
    TRANSFORMER = "transformer"
    AGGREGATOR = "aggregator"
    FILTER = "filter"


class PipelineStatus(Enum):
    """Pipeline execution status."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class PipelineContext:
    """Context passed through pipeline stages."""
    id: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Processing state
    current_stage: Optional[str] = None
    stage_results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def __post_init__(self):
        """Initialize context."""
        if not self.id:
            self.id = str(uuid.uuid4())

    @property
    def duration_ms(self) -> Optional[float]:
        """Get processing duration in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None

    def add_error(self, error: str) -> None:
        """Add error to context."""
        self.errors.append(error)
        logger.error(f"Pipeline context {self.id} error: {error}")

    def set_stage_result(self, stage_name: str, result: Any) -> None:
        """Set result for a specific stage."""
        self.stage_results[stage_name] = result

    def get_stage_result(self, stage_name: str) -> Any:
        """Get result from a specific stage."""
        return self.stage_results.get(stage_name)


class PipelineStage(ABC):
    """
    Abstract base class for pipeline stages.

    Stages are the building blocks of processing pipelines,
    each responsible for a specific transformation or operation.
    """

    def __init__(
        self,
        name: str,
        stage_type: PipelineStageType = PipelineStageType.PROCESSOR,
        enabled: bool = True,
        timeout_seconds: float = 300.0
    ):
        self.name = name
        self.stage_type = stage_type
        self.enabled = enabled
        self.timeout_seconds = timeout_seconds

        # Statistics
        self.executions = 0
        self.failures = 0
        self.total_execution_time_ms = 0.0

    @abstractmethod
    async def process(self, context: PipelineContext) -> PipelineContext:
        """
        Process pipeline context.

        Args:
            context: Pipeline context to process

        Returns:
            Updated pipeline context
        """
        pass

    async def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute stage with error handling and metrics.

        Args:
            context: Pipeline context

        Returns:
            Updated context
        """
        if not self.enabled:
            logger.debug(f"Stage {self.name} is disabled, skipping")
            return context

        context.current_stage = self.name
        start_time = time.time()

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self.process(context),
                timeout=self.timeout_seconds
            )

            # Update statistics
            execution_time_ms = (time.time() - start_time) * 1000
            self.executions += 1
            self.total_execution_time_ms += execution_time_ms

            logger.debug(f"Stage {self.name} completed in {execution_time_ms:.1f}ms")
            return result

        except asyncio.TimeoutError:
            error_msg = f"Stage {self.name} timeout after {self.timeout_seconds}s"
            context.add_error(error_msg)
            self.failures += 1
            return context

        except Exception as e:
            error_msg = f"Stage {self.name} failed: {str(e)}"
            context.add_error(error_msg)
            self.failures += 1
            logger.error(error_msg)
            return context

    def get_statistics(self) -> Dict[str, Any]:
        """Get stage execution statistics."""
        avg_execution_time = (
            self.total_execution_time_ms / self.executions
            if self.executions > 0 else 0.0
        )

        success_rate = (
            (self.executions - self.failures) / self.executions
            if self.executions > 0 else 1.0
        )

        return {
            "name": self.name,
            "stage_type": self.stage_type.value,
            "enabled": self.enabled,
            "executions": self.executions,
            "failures": self.failures,
            "success_rate": success_rate,
            "avg_execution_time_ms": avg_execution_time,
            "total_execution_time_ms": self.total_execution_time_ms
        }


class PreprocessorStage(PipelineStage):
    """
    Data preprocessing stage.

    Handles data validation, normalization, and preparation.
    """

    def __init__(self, name: str = "preprocessor", **kwargs):
        super().__init__(name, PipelineStageType.PREPROCESSOR, **kwargs)

    async def process(self, context: PipelineContext) -> PipelineContext:
        """Default preprocessing implementation."""
        # Validate required fields
        if "input" not in context.data:
            context.add_error("Missing required 'input' field")
            return context

        # Add preprocessing metadata
        context.metadata["preprocessed_at"] = datetime.utcnow().isoformat()
        context.metadata["original_data_size"] = len(str(context.data))

        return context


class ValidatorStage(PipelineStage):
    """
    Data validation stage.

    Validates data against schemas or business rules.
    """

    def __init__(
        self,
        name: str = "validator",
        validation_rules: Optional[List[Callable]] = None,
        **kwargs
    ):
        super().__init__(name, PipelineStageType.VALIDATOR, **kwargs)
        self.validation_rules = validation_rules or []

    async def process(self, context: PipelineContext) -> PipelineContext:
        """Validate data using configured rules."""
        for rule in self.validation_rules:
            try:
                if asyncio.iscoroutinefunction(rule):
                    is_valid = await rule(context.data)
                else:
                    is_valid = rule(context.data)

                if not is_valid:
                    context.add_error(f"Validation rule {rule.__name__} failed")

            except Exception as e:
                context.add_error(f"Validation rule {rule.__name__} error: {str(e)}")

        return context


class TransformerStage(PipelineStage):
    """
    Data transformation stage.

    Transforms data from one format to another.
    """

    def __init__(
        self,
        name: str = "transformer",
        transform_function: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(name, PipelineStageType.TRANSFORMER, **kwargs)
        self.transform_function = transform_function

    async def process(self, context: PipelineContext) -> PipelineContext:
        """Transform data using configured function."""
        if self.transform_function:
            try:
                if asyncio.iscoroutinefunction(self.transform_function):
                    context.data = await self.transform_function(context.data)
                else:
                    context.data = self.transform_function(context.data)

                context.metadata["transformed_at"] = datetime.utcnow().isoformat()

            except Exception as e:
                context.add_error(f"Transform function error: {str(e)}")

        return context


class FilterStage(PipelineStage):
    """
    Data filtering stage.

    Filters data based on criteria.
    """

    def __init__(
        self,
        name: str = "filter",
        filter_function: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(name, PipelineStageType.FILTER, **kwargs)
        self.filter_function = filter_function

    async def process(self, context: PipelineContext) -> PipelineContext:
        """Filter data using configured function."""
        if self.filter_function:
            try:
                if asyncio.iscoroutinefunction(self.filter_function):
                    should_include = await self.filter_function(context.data)
                else:
                    should_include = self.filter_function(context.data)

                if not should_include:
                    context.metadata["filtered_out"] = True
                    context.data = {}  # Clear data if filtered out

            except Exception as e:
                context.add_error(f"Filter function error: {str(e)}")

        return context


class AggregatorStage(PipelineStage):
    """
    Data aggregation stage.

    Combines multiple data points into aggregated results.
    """

    def __init__(
        self,
        name: str = "aggregator",
        aggregation_function: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(name, PipelineStageType.AGGREGATOR, **kwargs)
        self.aggregation_function = aggregation_function
        self.accumulated_data: List[Any] = []

    async def process(self, context: PipelineContext) -> PipelineContext:
        """Aggregate data using configured function."""
        # Add current data to accumulator
        self.accumulated_data.append(context.data)

        if self.aggregation_function:
            try:
                if asyncio.iscoroutinefunction(self.aggregation_function):
                    aggregated = await self.aggregation_function(self.accumulated_data)
                else:
                    aggregated = self.aggregation_function(self.accumulated_data)

                context.data = aggregated
                context.metadata["aggregated_at"] = datetime.utcnow().isoformat()
                context.metadata["aggregated_count"] = len(self.accumulated_data)

            except Exception as e:
                context.add_error(f"Aggregation function error: {str(e)}")

        return context


class AsyncProcessingPipeline:
    """
    Enterprise async processing pipeline.

    Provides configurable stages, middleware, monitoring,
    and reliable execution with error handling.
    """

    def __init__(
        self,
        name: str,
        stages: Optional[List[PipelineStage]] = None,
        max_concurrent_executions: int = 10,
        enable_events: bool = True,
        enable_metrics: bool = True
    ):
        self.name = name
        self.stages = stages or []
        self.max_concurrent_executions = max_concurrent_executions
        self.enable_events = enable_events
        self.enable_metrics = enable_metrics

        # Execution control
        self.status = PipelineStatus.IDLE
        self._execution_semaphore = asyncio.Semaphore(max_concurrent_executions)
        self._shutdown = False

        # Statistics
        self.executions = 0
        self.successes = 0
        self.failures = 0
        self.total_execution_time_ms = 0.0

        # Event bus integration
        if self.enable_events:
            self.event_bus = EventBus()

    def add_stage(self, stage: PipelineStage, position: Optional[int] = None) -> None:
        """Add stage to pipeline."""
        if position is None:
            self.stages.append(stage)
        else:
            self.stages.insert(position, stage)

        logger.info(f"Added stage {stage.name} to pipeline {self.name}")

    def remove_stage(self, stage_name: str) -> bool:
        """Remove stage from pipeline."""
        for i, stage in enumerate(self.stages):
            if stage.name == stage_name:
                del self.stages[i]
                logger.info(f"Removed stage {stage_name} from pipeline {self.name}")
                return True
        return False

    def get_stage(self, stage_name: str) -> Optional[PipelineStage]:
        """Get stage by name."""
        for stage in self.stages:
            if stage.name == stage_name:
                return stage
        return None

    async def execute(
        self,
        data: Dict[str, Any],
        context_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PipelineContext:
        """
        Execute pipeline on data.

        Args:
            data: Input data
            context_id: Optional context ID
            metadata: Optional metadata

        Returns:
            Pipeline context with results
        """
        async with self._execution_semaphore:
            context = PipelineContext(
                id=context_id or str(uuid.uuid4()),
                data=data,
                metadata=metadata or {}
            )

            context.start_time = datetime.utcnow()
            start_time = time.time()

            try:
                # Publish start event
                if self.enable_events:
                    publish_event(
                        "pipeline.execution.started",
                        {
                            "pipeline_name": self.name,
                            "context_id": context.id,
                            "stages_count": len(self.stages)
                        }
                    )

                # Execute stages
                for stage in self.stages:
                    if self._shutdown:
                        context.add_error("Pipeline shutdown during execution")
                        break

                    context = await stage.execute(context)

                    # Stop if errors occurred and stage is critical
                    if context.errors and getattr(stage, 'stop_on_error', False):
                        break

                # Mark completion
                context.end_time = datetime.utcnow()
                execution_time_ms = (time.time() - start_time) * 1000

                # Update statistics
                self.executions += 1
                self.total_execution_time_ms += execution_time_ms

                if not context.errors:
                    self.successes += 1
                else:
                    self.failures += 1

                # Publish completion event
                if self.enable_events:
                    publish_event(
                        "pipeline.execution.completed",
                        {
                            "pipeline_name": self.name,
                            "context_id": context.id,
                            "success": len(context.errors) == 0,
                            "execution_time_ms": execution_time_ms,
                            "errors": context.errors
                        }
                    )

                logger.info(
                    f"Pipeline {self.name} executed context {context.id} in {execution_time_ms:.1f}ms "
                    f"with {len(context.errors)} errors"
                )

                return context

            except Exception as e:
                context.add_error(f"Pipeline execution error: {str(e)}")
                context.end_time = datetime.utcnow()
                self.failures += 1

                # Publish error event
                if self.enable_events:
                    publish_event(
                        "pipeline.execution.failed",
                        {
                            "pipeline_name": self.name,
                            "context_id": context.id,
                            "error": str(e)
                        }
                    )

                logger.error(f"Pipeline {self.name} execution failed: {e}")
                return context

    async def execute_batch(
        self,
        data_batch: List[Dict[str, Any]],
        batch_size: int = 10
    ) -> List[PipelineContext]:
        """
        Execute pipeline on batch of data.

        Args:
            data_batch: List of input data
            batch_size: Size of concurrent batches

        Returns:
            List of pipeline contexts
        """
        results = []

        # Process in batches
        for i in range(0, len(data_batch), batch_size):
            batch = data_batch[i:i + batch_size]

            # Execute batch concurrently
            tasks = [
                self.execute(data)
                for data in batch
            ]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions
            for result in batch_results:
                if isinstance(result, Exception):
                    # Create error context
                    error_context = PipelineContext(
                        id=str(uuid.uuid4()),
                        data={}
                    )
                    error_context.add_error(f"Batch execution error: {str(result)}")
                    results.append(error_context)
                else:
                    results.append(result)

        return results

    def start(self) -> None:
        """Start pipeline."""
        if self.status == PipelineStatus.RUNNING:
            logger.warning(f"Pipeline {self.name} already running")
            return

        self.status = PipelineStatus.RUNNING
        self._shutdown = False

        logger.info(f"Pipeline {self.name} started")

    def pause(self) -> None:
        """Pause pipeline."""
        if self.status == PipelineStatus.RUNNING:
            self.status = PipelineStatus.PAUSED
            logger.info(f"Pipeline {self.name} paused")

    def resume(self) -> None:
        """Resume pipeline."""
        if self.status == PipelineStatus.PAUSED:
            self.status = PipelineStatus.RUNNING
            logger.info(f"Pipeline {self.name} resumed")

    def stop(self) -> None:
        """Stop pipeline."""
        self.status = PipelineStatus.STOPPED
        self._shutdown = True

        logger.info(f"Pipeline {self.name} stopped")

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline execution statistics."""
        avg_execution_time = (
            self.total_execution_time_ms / self.executions
            if self.executions > 0 else 0.0
        )

        success_rate = (
            self.successes / self.executions
            if self.executions > 0 else 1.0
        )

        stage_stats = [stage.get_statistics() for stage in self.stages]

        return {
            "pipeline_name": self.name,
            "status": self.status.value,
            "executions": self.executions,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": success_rate,
            "avg_execution_time_ms": avg_execution_time,
            "total_execution_time_ms": self.total_execution_time_ms,
            "stages_count": len(self.stages),
            "stage_statistics": stage_stats
        }


class PipelineManager:
    """
    Manager for multiple processing pipelines.

    Provides centralized management, monitoring, and coordination.
    """

    def __init__(self):
        self.pipelines: Dict[str, AsyncProcessingPipeline] = {}
        self._lock = asyncio.Lock()

    async def register_pipeline(
        self,
        name: str,
        pipeline: AsyncProcessingPipeline
    ) -> None:
        """Register a pipeline."""
        async with self._lock:
            self.pipelines[name] = pipeline
            logger.info(f"Registered pipeline: {name}")

    async def unregister_pipeline(self, name: str) -> bool:
        """Unregister a pipeline."""
        async with self._lock:
            if name in self.pipelines:
                pipeline = self.pipelines[name]
                pipeline.stop()
                del self.pipelines[name]
                logger.info(f"Unregistered pipeline: {name}")
                return True
            return False

    def get_pipeline(self, name: str) -> Optional[AsyncProcessingPipeline]:
        """Get pipeline by name."""
        return self.pipelines.get(name)

    async def execute_pipeline(
        self,
        pipeline_name: str,
        data: Dict[str, Any],
        **kwargs
    ) -> Optional[PipelineContext]:
        """Execute specific pipeline."""
        pipeline = self.get_pipeline(pipeline_name)
        if not pipeline:
            logger.error(f"Pipeline {pipeline_name} not found")
            return None

        return await pipeline.execute(data, **kwargs)

    def get_all_statistics(self) -> Dict[str, Any]:
        """Get statistics for all pipelines."""
        return {
            name: pipeline.get_statistics()
            for name, pipeline in self.pipelines.items()
        }

    async def shutdown_all(self) -> None:
        """Shutdown all pipelines."""
        async with self._lock:
            for pipeline in self.pipelines.values():
                pipeline.stop()

            logger.info("All pipelines shut down")


# Global pipeline manager
_pipeline_manager: Optional[PipelineManager] = None


async def get_pipeline_manager() -> PipelineManager:
    """Get global pipeline manager."""
    global _pipeline_manager

    if _pipeline_manager is None:
        _pipeline_manager = PipelineManager()

    return _pipeline_manager