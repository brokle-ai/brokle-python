"""
Brokle Observations Module.

Provides semantic observation types for AI application tracing,
following patterns from Langfuse's observation type hierarchy.

Observation types enable semantic differentiation between:
- Generations (LLM calls)
- Spans (generic operations)
- Events (point-in-time occurrences)
- Agents (autonomous AI agents)
- Tools (function/tool calls)
- Chains (operation sequences)
- Retrievals (RAG operations)
- Embeddings (vector generations)

Example:
    >>> from brokle.observations import ObservationType
    >>> @observe(as_type=ObservationType.AGENT)
    ... def my_agent(query: str):
    ...     # Agent logic
    ...     pass
"""

from .types import ObservationType
from .wrapper import (
    BrokleAgent,
    BrokleEvent,
    BrokleGeneration,
    BrokleObservation,
    BrokleRetrieval,
    BrokleTool,
)

__all__ = [
    # Type enum
    "ObservationType",
    # Observation wrappers
    "BrokleObservation",
    "BrokleGeneration",
    "BrokleEvent",
    "BrokleAgent",
    "BrokleTool",
    "BrokleRetrieval",
]
