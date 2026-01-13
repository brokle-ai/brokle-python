"""SDK wrappers for automatic LLM observability."""

from .anthropic import wrap_anthropic, wrap_anthropic_async
from .azure_openai import wrap_azure_openai, wrap_azure_openai_async
from .bedrock import wrap_bedrock
from .cohere import wrap_cohere
from .google import wrap_google
from .mistral import wrap_mistral
from .openai import wrap_openai, wrap_openai_async

__all__ = [
    # OpenAI
    "wrap_openai",
    "wrap_openai_async",
    # Anthropic
    "wrap_anthropic",
    "wrap_anthropic_async",
    # Azure OpenAI
    "wrap_azure_openai",
    "wrap_azure_openai_async",
    # Google GenAI (google-genai)
    "wrap_google",
    # Mistral
    "wrap_mistral",
    # Cohere
    "wrap_cohere",
    # AWS Bedrock
    "wrap_bedrock",
]
