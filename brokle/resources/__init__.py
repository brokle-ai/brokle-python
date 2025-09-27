"""
OpenAI-style resource organization for Brokle SDK.

Provides chat, embeddings, and models resources with sync/async variants.
"""

from .base import AsyncBaseResource, BaseResource
from .chat import AsyncChatResource, ChatResource
from .embeddings import AsyncEmbeddingsResource, EmbeddingsResource
from .models import AsyncModelsResource, ModelsResource

__all__ = [
    "BaseResource",
    "AsyncBaseResource",
    "ChatResource",
    "AsyncChatResource",
    "EmbeddingsResource",
    "AsyncEmbeddingsResource",
    "ModelsResource",
    "AsyncModelsResource",
]
