"""
Enterprise integrations and instrumentation for Brokle SDK.

This module provides comprehensive integration capabilities including:
- Automatic instrumentation for popular libraries
- Enterprise-grade webhook system
- Event bus for real-time communication
- Async processing pipelines
- Advanced background task management
"""

# Core instrumentation
from .openai_instrumentation import OpenAIInstrumentation
from .anthropic_instrumentation import AnthropicInstrumentation
from .langchain_instrumentation import LangChainInstrumentation
from .registry import (
    InstrumentationRegistry,
    auto_instrument,
    print_status,
    print_health_report,
    get_status,
    get_health_report,
    reset_all_errors,
    get_registry
)

# Advanced auto-instrumentation
from .auto_instrumentation import (
    AutoInstrumentationEngine,
    LibraryInfo,
    LibraryStatus,
    get_auto_engine,
    discover_libraries,
    get_instrumentation_status,
    instrument_library,
    uninstrument_library,
    enable_auto_instrumentation,
    disable_auto_instrumentation,
    health_check
)

# Webhook system
from .webhooks import (
    WebhookManager,
    WebhookEvent,
    WebhookEndpoint,
    WebhookEventType,
    WebhookStatus,
    get_webhook_manager,
    register_webhook,
    send_webhook_event,
    get_webhook_stats
)

# Event system
from .events import (
    EventBus,
    Event,
    EventFilter,
    EventSubscription,
    EventPriority,
    EventStatus,
    get_event_bus,
    publish_event,
    subscribe_to_events,
    get_event_statistics
)

# Processing pipeline
from .pipeline import (
    AsyncProcessingPipeline,
    PipelineStage,
    PipelineContext,
    PipelineStageType,
    PipelineStatus,
    PreprocessorStage,
    ValidatorStage,
    TransformerStage,
    FilterStage,
    AggregatorStage,
    PipelineManager,
    get_pipeline_manager
)

# Error handling utilities
from .error_handlers import (
    InstrumentationError,
    LibraryNotAvailableError,
    ObservabilityError,
    ConfigurationError,
    ErrorSeverity,
    get_error_handler
)

__all__ = [
    # Core instrumentation
    "OpenAIInstrumentation",
    "AnthropicInstrumentation",
    "LangChainInstrumentation",
    "InstrumentationRegistry",
    "auto_instrument",

    # Status and health monitoring
    "print_status",
    "print_health_report",
    "get_status",
    "get_health_report",
    "reset_all_errors",
    "get_registry",

    # Advanced auto-instrumentation
    "AutoInstrumentationEngine",
    "LibraryInfo",
    "LibraryStatus",
    "get_auto_engine",
    "discover_libraries",
    "get_instrumentation_status",
    "instrument_library",
    "uninstrument_library",
    "enable_auto_instrumentation",
    "disable_auto_instrumentation",
    "health_check",

    # Webhook system
    "WebhookManager",
    "WebhookEvent",
    "WebhookEndpoint",
    "WebhookEventType",
    "WebhookStatus",
    "get_webhook_manager",
    "register_webhook",
    "send_webhook_event",
    "get_webhook_stats",

    # Event system
    "EventBus",
    "Event",
    "EventFilter",
    "EventSubscription",
    "EventPriority",
    "EventStatus",
    "get_event_bus",
    "publish_event",
    "subscribe_to_events",
    "get_event_statistics",

    # Processing pipeline
    "AsyncProcessingPipeline",
    "PipelineStage",
    "PipelineContext",
    "PipelineStageType",
    "PipelineStatus",
    "PreprocessorStage",
    "ValidatorStage",
    "TransformerStage",
    "FilterStage",
    "AggregatorStage",
    "PipelineManager",
    "get_pipeline_manager",

    # Error handling
    "InstrumentationError",
    "LibraryNotAvailableError",
    "ObservabilityError",
    "ConfigurationError",
    "ErrorSeverity",
    "get_error_handler"
]