"""
OpenAI-compatible client implementation.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union, Iterator, AsyncIterator

try:
    import openai
    from openai import OpenAI as OpenAIClient, AsyncOpenAI as AsyncOpenAIClient
    from openai.types.chat import ChatCompletion, ChatCompletionChunk
    from openai.types import Completion, CompletionChoice, CreateEmbeddingResponse
    from openai import Stream, AsyncStream
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAIClient = None
    AsyncOpenAIClient = None

from ..client import Brokle
from ..config import get_config
from ..decorators import observe
# from ..core.telemetry import start_generation  # TODO: Fix broken import


class OpenAI:
    """OpenAI-compatible client that routes through Brokle."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        **kwargs
    ):
        """Initialize OpenAI-compatible client."""
        if not HAS_OPENAI:
            raise ImportError("OpenAI package is required for OpenAI compatibility. Install with: pip install openai")
        
        # Initialize Brokle client
        self.brokle = Brokle(
            public_key=api_key,
            host=base_url,
            secret_key=project,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs
        )
        
        # Initialize OpenAI client sections
        self.chat = self.ChatCompletions(self.brokle)
        self.completions = self.Completions(self.brokle)
        self.embeddings = self.Embeddings(self.brokle)
        self.models = self.Models(self.brokle)
    
    class ChatCompletions:
        """Chat completions with Brokle integration."""
        
        def __init__(self, brokle: Brokle):
            self.brokle = brokle
            self.completions = self
        
        @observe(as_type="generation")
        def create(
            self,
            *,
            model: str,
            messages: List[Dict[str, str]],
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            top_p: Optional[float] = None,
            frequency_penalty: Optional[float] = None,
            presence_penalty: Optional[float] = None,
            stop: Optional[Union[str, List[str]]] = None,
            stream: Optional[bool] = None,
            n: Optional[int] = None,
            logit_bias: Optional[Dict[str, float]] = None,
            user: Optional[str] = None,
            # Brokle specific parameters
            routing_strategy: Optional[str] = None,
            cache_strategy: Optional[str] = None,
            cache_similarity_threshold: Optional[float] = None,
            max_cost_usd: Optional[float] = None,
            evaluation_metrics: Optional[List[str]] = None,
            custom_tags: Optional[Dict[str, Any]] = None,
            **kwargs
        ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
            """Create chat completion with Brokle routing."""
            # Prepare request data
            request_data = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "stop": stop,
                "stream": stream,
                "n": n,
                "logit_bias": logit_bias,
                "user": user,
                # Brokle parameters
                "routing_strategy": routing_strategy,
                "cache_strategy": cache_strategy,
                "cache_similarity_threshold": cache_similarity_threshold,
                "max_cost_usd": max_cost_usd,
                "evaluation_metrics": evaluation_metrics,
                "custom_tags": custom_tags,
                **kwargs
            }
            
            # Remove None values
            request_data = {k: v for k, v in request_data.items() if v is not None}
            
            # Make request through Brokle
            response = self.brokle.chat.create_sync(**request_data)
            
            # Convert to OpenAI format
            return self._convert_to_openai_format(response, stream=stream)
        
        def _convert_to_openai_format(self, response: Any, stream: bool = False) -> Any:
            """Convert Brokle response to OpenAI format."""
            if stream:
                # Handle streaming response
                return self._create_stream_response(response)
            else:
                # Handle regular response
                return self._create_chat_completion(response)
        
        def _create_chat_completion(self, response: Any) -> ChatCompletion:
            """Create OpenAI ChatCompletion object."""
            # This is a simplified conversion - in production you'd want more robust handling
            return ChatCompletion(
                id=response.id,
                object="chat.completion",
                created=response.created,
                model=response.model,
                choices=response.choices,
                usage=response.usage,
                system_fingerprint=getattr(response, 'system_fingerprint', None)
            )
        
        def _create_stream_response(self, response: Any) -> Iterator[ChatCompletionChunk]:
            """Create streaming response."""
            # This would handle streaming conversion
            # For now, just return the response
            return response
    
    class Completions:
        """Text completions with Brokle integration."""
        
        def __init__(self, brokle: Brokle):
            self.brokle = brokle
        
        @observe(as_type="generation")
        def create(
            self,
            *,
            model: str,
            prompt: str,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            frequency_penalty: Optional[float] = None,
            presence_penalty: Optional[float] = None,
            stop: Optional[Union[str, List[str]]] = None,
            stream: Optional[bool] = None,
            n: Optional[int] = None,
            logprobs: Optional[int] = None,
            echo: Optional[bool] = None,
            suffix: Optional[str] = None,
            user: Optional[str] = None,
            # Brokle specific parameters
            routing_strategy: Optional[str] = None,
            cache_strategy: Optional[str] = None,
            cache_similarity_threshold: Optional[float] = None,
            max_cost_usd: Optional[float] = None,
            evaluation_metrics: Optional[List[str]] = None,
            custom_tags: Optional[Dict[str, Any]] = None,
            **kwargs
        ) -> Union[Completion, Iterator[Completion]]:
            """Create completion with Brokle routing."""
            # Prepare request data
            request_data = {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "stop": stop,
                "stream": stream,
                "n": n,
                "logprobs": logprobs,
                "echo": echo,
                "suffix": suffix,
                "user": user,
                # Brokle parameters
                "routing_strategy": routing_strategy,
                "cache_strategy": cache_strategy,
                "cache_similarity_threshold": cache_similarity_threshold,
                "max_cost_usd": max_cost_usd,
                "evaluation_metrics": evaluation_metrics,
                "custom_tags": custom_tags,
                **kwargs
            }
            
            # Remove None values
            request_data = {k: v for k, v in request_data.items() if v is not None}
            
            # Make request through Brokle
            response = self.brokle.completions.create_sync(**request_data)
            
            # Convert to OpenAI format
            return self._convert_to_openai_format(response, stream=stream)
        
        def _convert_to_openai_format(self, response: Any, stream: bool = False) -> Any:
            """Convert Brokle response to OpenAI format."""
            if stream:
                return self._create_stream_response(response)
            else:
                return self._create_completion(response)
        
        def _create_completion(self, response: Any) -> Completion:
            """Create OpenAI Completion object."""
            return Completion(
                id=response.id,
                object="text_completion",
                created=response.created,
                model=response.model,
                choices=response.choices,
                usage=response.usage
            )
        
        def _create_stream_response(self, response: Any) -> Iterator[Completion]:
            """Create streaming response."""
            return response
    
    class Embeddings:
        """Embeddings with Brokle integration."""
        
        def __init__(self, brokle: Brokle):
            self.brokle = brokle
        
        @observe(as_type="generation")
        def create(
            self,
            *,
            model: str,
            input: Union[str, List[str]],
            encoding_format: Optional[str] = None,
            dimensions: Optional[int] = None,
            user: Optional[str] = None,
            # Brokle specific parameters
            routing_strategy: Optional[str] = None,
            cache_strategy: Optional[str] = None,
            custom_tags: Optional[Dict[str, Any]] = None,
            **kwargs
        ) -> CreateEmbeddingResponse:
            """Create embeddings with Brokle routing."""
            # Prepare request data
            request_data = {
                "model": model,
                "input": input,
                "encoding_format": encoding_format,
                "dimensions": dimensions,
                "user": user,
                # Brokle parameters
                "routing_strategy": routing_strategy,
                "cache_strategy": cache_strategy,
                "custom_tags": custom_tags,
                **kwargs
            }
            
            # Remove None values
            request_data = {k: v for k, v in request_data.items() if v is not None}
            
            # Make request through Brokle
            response = self.brokle.embeddings.create_sync(**request_data)
            
            # Convert to OpenAI format
            return self._convert_to_openai_format(response)
        
        def _convert_to_openai_format(self, response: Any) -> CreateEmbeddingResponse:
            """Convert Brokle response to OpenAI format."""
            return CreateEmbeddingResponse(
                object="list",
                data=response.data,
                model=response.model,
                usage=response.usage
            )
    
    class Models:
        """Models API (pass-through to OpenAI)."""
        
        def __init__(self, brokle: Brokle):
            self.brokle = brokle
        
        def list(self) -> Dict[str, Any]:
            """List available models."""
            # This would call Brokle's models endpoint
            return {"object": "list", "data": []}
        
        def retrieve(self, model: str) -> Dict[str, Any]:
            """Retrieve model information."""
            # This would call Brokle's model info endpoint
            return {"id": model, "object": "model"}


class AsyncOpenAI:
    """Async OpenAI-compatible client that routes through Brokle."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        **kwargs
    ):
        """Initialize async OpenAI-compatible client."""
        if not HAS_OPENAI:
            raise ImportError("OpenAI package is required for OpenAI compatibility. Install with: pip install openai")
        
        # Initialize Brokle client
        self.brokle = Brokle(
            public_key=api_key,
            host=base_url,
            secret_key=project,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs
        )
        
        # Initialize OpenAI client sections
        self.chat = self.ChatCompletions(self.brokle)
        self.completions = self.Completions(self.brokle)
        self.embeddings = self.Embeddings(self.brokle)
        self.models = self.Models(self.brokle)
    
    class ChatCompletions:
        """Async chat completions with Brokle integration."""
        
        def __init__(self, brokle: Brokle):
            self.brokle = brokle
            self.completions = self
        
        @observe(as_type="generation")
        async def create(
            self,
            *,
            model: str,
            messages: List[Dict[str, str]],
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            top_p: Optional[float] = None,
            frequency_penalty: Optional[float] = None,
            presence_penalty: Optional[float] = None,
            stop: Optional[Union[str, List[str]]] = None,
            stream: Optional[bool] = None,
            n: Optional[int] = None,
            logit_bias: Optional[Dict[str, float]] = None,
            user: Optional[str] = None,
            # Brokle specific parameters
            routing_strategy: Optional[str] = None,
            cache_strategy: Optional[str] = None,
            cache_similarity_threshold: Optional[float] = None,
            max_cost_usd: Optional[float] = None,
            evaluation_metrics: Optional[List[str]] = None,
            custom_tags: Optional[Dict[str, Any]] = None,
            **kwargs
        ) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
            """Create async chat completion with Brokle routing."""
            # Prepare request data
            request_data = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "stop": stop,
                "stream": stream,
                "n": n,
                "logit_bias": logit_bias,
                "user": user,
                # Brokle parameters
                "routing_strategy": routing_strategy,
                "cache_strategy": cache_strategy,
                "cache_similarity_threshold": cache_similarity_threshold,
                "max_cost_usd": max_cost_usd,
                "evaluation_metrics": evaluation_metrics,
                "custom_tags": custom_tags,
                **kwargs
            }
            
            # Remove None values
            request_data = {k: v for k, v in request_data.items() if v is not None}
            
            # Make request through Brokle
            response = await self.brokle.chat.create(**request_data)
            
            # Convert to OpenAI format
            return self._convert_to_openai_format(response, stream=stream)
        
        def _convert_to_openai_format(self, response: Any, stream: bool = False) -> Any:
            """Convert Brokle response to OpenAI format."""
            if stream:
                return self._create_stream_response(response)
            else:
                return self._create_chat_completion(response)
        
        def _create_chat_completion(self, response: Any) -> ChatCompletion:
            """Create OpenAI ChatCompletion object."""
            return ChatCompletion(
                id=response.id,
                object="chat.completion",
                created=response.created,
                model=response.model,
                choices=response.choices,
                usage=response.usage,
                system_fingerprint=getattr(response, 'system_fingerprint', None)
            )
        
        async def _create_stream_response(self, response: Any) -> AsyncIterator[ChatCompletionChunk]:
            """Create async streaming response."""
            # This would handle async streaming conversion
            # For now, just return the response
            async for chunk in response:
                yield chunk
    
    class Completions:
        """Async text completions with Brokle integration."""
        
        def __init__(self, brokle: Brokle):
            self.brokle = brokle
        
        @observe(as_type="generation")
        async def create(
            self,
            *,
            model: str,
            prompt: str,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            frequency_penalty: Optional[float] = None,
            presence_penalty: Optional[float] = None,
            stop: Optional[Union[str, List[str]]] = None,
            stream: Optional[bool] = None,
            n: Optional[int] = None,
            logprobs: Optional[int] = None,
            echo: Optional[bool] = None,
            suffix: Optional[str] = None,
            user: Optional[str] = None,
            # Brokle specific parameters
            routing_strategy: Optional[str] = None,
            cache_strategy: Optional[str] = None,
            cache_similarity_threshold: Optional[float] = None,
            max_cost_usd: Optional[float] = None,
            evaluation_metrics: Optional[List[str]] = None,
            custom_tags: Optional[Dict[str, Any]] = None,
            **kwargs
        ) -> Union[Completion, AsyncIterator[Completion]]:
            """Create async completion with Brokle routing."""
            # Prepare request data
            request_data = {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "stop": stop,
                "stream": stream,
                "n": n,
                "logprobs": logprobs,
                "echo": echo,
                "suffix": suffix,
                "user": user,
                # Brokle parameters
                "routing_strategy": routing_strategy,
                "cache_strategy": cache_strategy,
                "cache_similarity_threshold": cache_similarity_threshold,
                "max_cost_usd": max_cost_usd,
                "evaluation_metrics": evaluation_metrics,
                "custom_tags": custom_tags,
                **kwargs
            }
            
            # Remove None values
            request_data = {k: v for k, v in request_data.items() if v is not None}
            
            # Make request through Brokle
            response = await self.brokle.completions.create(**request_data)
            
            # Convert to OpenAI format
            return self._convert_to_openai_format(response, stream=stream)
        
        def _convert_to_openai_format(self, response: Any, stream: bool = False) -> Any:
            """Convert Brokle response to OpenAI format."""
            if stream:
                return self._create_stream_response(response)
            else:
                return self._create_completion(response)
        
        def _create_completion(self, response: Any) -> Completion:
            """Create OpenAI Completion object."""
            return Completion(
                id=response.id,
                object="text_completion",
                created=response.created,
                model=response.model,
                choices=response.choices,
                usage=response.usage
            )
        
        async def _create_stream_response(self, response: Any) -> AsyncIterator[Completion]:
            """Create async streaming response."""
            async for chunk in response:
                yield chunk
    
    class Embeddings:
        """Async embeddings with Brokle integration."""
        
        def __init__(self, brokle: Brokle):
            self.brokle = brokle
        
        @observe(as_type="generation")
        async def create(
            self,
            *,
            model: str,
            input: Union[str, List[str]],
            encoding_format: Optional[str] = None,
            dimensions: Optional[int] = None,
            user: Optional[str] = None,
            # Brokle specific parameters
            routing_strategy: Optional[str] = None,
            cache_strategy: Optional[str] = None,
            custom_tags: Optional[Dict[str, Any]] = None,
            **kwargs
        ) -> CreateEmbeddingResponse:
            """Create async embeddings with Brokle routing."""
            # Prepare request data
            request_data = {
                "model": model,
                "input": input,
                "encoding_format": encoding_format,
                "dimensions": dimensions,
                "user": user,
                # Brokle parameters
                "routing_strategy": routing_strategy,
                "cache_strategy": cache_strategy,
                "custom_tags": custom_tags,
                **kwargs
            }
            
            # Remove None values
            request_data = {k: v for k, v in request_data.items() if v is not None}
            
            # Make request through Brokle
            response = await self.brokle.embeddings.create(**request_data)
            
            # Convert to OpenAI format
            return self._convert_to_openai_format(response)
        
        def _convert_to_openai_format(self, response: Any) -> CreateEmbeddingResponse:
            """Convert Brokle response to OpenAI format."""
            return CreateEmbeddingResponse(
                object="list",
                data=response.data,
                model=response.model,
                usage=response.usage
            )
    
    class Models:
        """Async models API (pass-through to OpenAI)."""
        
        def __init__(self, brokle: Brokle):
            self.brokle = brokle
        
        async def list(self) -> Dict[str, Any]:
            """List available models."""
            # This would call Brokle's models endpoint
            return {"object": "list", "data": []}
        
        async def retrieve(self, model: str) -> Dict[str, Any]:
            """Retrieve model information."""
            # This would call Brokle's model info endpoint
            return {"id": model, "object": "model"}
    
    async def close(self) -> None:
        """Close the async client."""
        await self.brokle.close()
    
    async def __aenter__(self) -> 'AsyncOpenAI':
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()