"""
OpenAI-style resource organization for Brokle SDK.

Provides chat, embeddings, and models resources with sync/async variants.
"""

from .base import BaseResource, AsyncBaseResource
from .chat import ChatResource, AsyncChatResource
from .embeddings import EmbeddingsResource, AsyncEmbeddingsResource
from .models import ModelsResource, AsyncModelsResource

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