"""
Observation type definitions for Brokle SDK.

Provides semantic observation types matching Brokle's brokle.span.type attribute,
following the Langfuse pattern of 10+ semantic types for AI application tracing.
"""

from enum import Enum


class ObservationType(str, Enum):
    """
    Semantic observation types for AI application tracing.

    These types enable semantic differentiation in the Brokle backend,
    allowing for specialized analysis, visualization, and filtering.

    Types follow the Langfuse observation type hierarchy pattern,
    which has proven effective for AI application observability.

    Usage:
        >>> from brokle import observe
        >>> from brokle.observations import ObservationType
        >>>
        >>> @observe(as_type=ObservationType.AGENT)
        ... def my_agent(query: str):
        ...     return process_query(query)
    """

    # ========== Core Types ==========

    SPAN = "span"
    """
    Generic span for any operation.

    Use for general-purpose tracing of functions, methods, or code blocks
    that don't fit a more specific type.

    Example: Database queries, HTTP requests, business logic
    """

    GENERATION = "generation"
    """
    LLM generation (chat completion, text completion).

    Use for all LLM API calls that generate text or other content.
    Backend applies special handling for token usage, cost calculation,
    and model-specific analytics.

    Example: OpenAI chat.completions.create(), Anthropic messages.create()
    """

    EVENT = "event"
    """
    Point-in-time event (no duration).

    Use for logging significant occurrences that don't have meaningful duration.
    Backend may visualize these differently (markers vs spans).

    Example: User actions, system events, state transitions
    """

    # ========== AI Agent Types ==========

    AGENT = "agent"
    """
    Autonomous AI agent operation.

    Use for agent-based systems (LangChain agents, AutoGPT, custom agents)
    where the AI makes autonomous decisions about tool usage and next steps.

    Example: ReAct agent loops, planning agents, multi-step reasoning
    """

    TOOL = "tool"
    """
    Tool/function call within an agent or chain.

    Use for function calling, tool use, and external system invocations
    made by agents or LLM-driven workflows.

    Example: Calculator tools, API calls, database queries in agent context
    """

    CHAIN = "chain"
    """
    Chain of operations (sequential or parallel).

    Use for LangChain-style chains, pipelines, or any sequence of
    operations that form a logical unit.

    Example: LLMChain, SequentialChain, custom pipelines
    """

    # ========== RAG Types ==========

    RETRIEVAL = "retrieval"
    """
    RAG retrieval operation.

    Use for vector database queries, document retrieval, and any
    operation that fetches context for augmentation.

    Example: Vector similarity search, document fetching, context assembly
    """

    EMBEDDING = "embedding"
    """
    Embedding/vector generation.

    Use for text-to-vector operations, whether for indexing or querying.
    Backend tracks embedding dimensions and token usage.

    Example: OpenAI embeddings, sentence-transformers, custom models
    """

    # ========== Quality & Evaluation Types ==========

    EVALUATOR = "evaluator"
    """
    Quality evaluation operation.

    Use for LLM-as-judge evaluations, automated quality scoring,
    and any evaluation logic that produces quality metrics.

    Example: GPT-4 grading, custom evaluators, toxicity checks
    """

    GUARDRAIL = "guardrail"
    """
    Safety guardrail check.

    Use for content moderation, safety checks, and policy enforcement
    that validates or filters LLM inputs/outputs.

    Example: OpenAI moderation API, custom filters, PII detection
    """

    # ========== Utility Types ==========

    RERANK = "rerank"
    """
    Reranking operation.

    Use for reranking retrieved documents or results using
    cross-encoder models or LLM-based reranking.

    Example: Cohere rerank, cross-encoder models, LLM reranking
    """

    WORKFLOW = "workflow"
    """
    High-level workflow orchestration.

    Use for top-level workflow spans that orchestrate multiple
    agents, chains, or complex multi-step processes.

    Example: Multi-agent systems, complex pipelines, orchestration logic
    """


# Type aliases for convenience (matching SpanType)
SPAN = ObservationType.SPAN
GENERATION = ObservationType.GENERATION
EVENT = ObservationType.EVENT
AGENT = ObservationType.AGENT
TOOL = ObservationType.TOOL
CHAIN = ObservationType.CHAIN
RETRIEVAL = ObservationType.RETRIEVAL
EMBEDDING = ObservationType.EMBEDDING
EVALUATOR = ObservationType.EVALUATOR
GUARDRAIL = ObservationType.GUARDRAIL
RERANK = ObservationType.RERANK
WORKFLOW = ObservationType.WORKFLOW
