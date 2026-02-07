# Brokle Python SDK - Complete API Reference

The Brokle Python SDK provides three integration patterns for adding AI observability, routing, and optimization to your applications.

## 🎯 Three Integration Patterns Overview

- **Pattern 1: Wrapper Functions** - Explicit wrapping of existing AI clients
- **Pattern 2: Universal Decorator** - Framework-agnostic `@observe()` decorator
- **Pattern 3: Native SDK** - Full platform features with OpenAI-compatible interface

---

# Pattern 1: Wrapper Functions API

## `wrap_openai(client) -> OpenAIType`

Wrap an existing OpenAI client with Brokle observability.

**Parameters:**
- `client` (OpenAI | AsyncOpenAI): The OpenAI client instance to wrap

**Returns:**
- The same client instance with instrumented methods (identical interface)

**Raises:**
- `ProviderError`: If OpenAI SDK not installed or client is invalid

**How it works:**
The wrapper instruments the client's `chat.completions.create` method to automatically create OTEL spans with GenAI semantic attributes. It resolves the Brokle client at call time via `get_client()`, so you must initialize Brokle (via `Brokle(api_key=...)` or `BROKLE_API_KEY` env var) before making wrapped calls.

**Example:**
```python
from openai import OpenAI
from brokle import Brokle, wrap_openai

# 1. Initialize Brokle client
brokle = Brokle(api_key="bk_...")

# 2. Wrap OpenAI client (single argument)
client = wrap_openai(OpenAI(api_key="sk-..."))

# 3. Use exactly like normal OpenAI client
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)
```

**Setting trace attributes (tags, session_id, user_id):**

These are trace-level concerns — set them via `@observe` or `start_as_current_span`, not on the wrapper:

```python
# Option 1: @observe decorator
@observe(session_id="session_123", user_id="user_456", tags=["production"])
def chat(message: str):
    return client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": message}],
    )

# Option 2: Context manager
with brokle.start_as_current_span("chat_session") as span:
    span.update_trace(session_id="session_123", tags=["production"])
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
    )
```

## `wrap_anthropic(client) -> AnthropicType`

Wrap an existing Anthropic client with Brokle observability.

**Parameters:**
- `client` (Anthropic | AsyncAnthropic): The Anthropic client instance to wrap

**Returns:**
- The same client instance with instrumented methods (identical interface)

**Example:**
```python
from anthropic import Anthropic
from brokle import Brokle, wrap_anthropic

brokle = Brokle(api_key="bk_...")
client = wrap_anthropic(Anthropic(api_key="sk-ant-..."))

message = client.messages.create(
    model="claude-3-opus-20240229",
    messages=[{"role": "user", "content": "Hello"}]
)
```

---

# Pattern 2: Universal Decorator API

## `@observe(**kwargs) -> Callable`

Universal decorator for function observability with AI-aware intelligence.

**Parameters:**
- `name` (str, optional): Custom span name (defaults to function name)
- `capture_inputs` (bool, default=True): Whether to capture function inputs
- `capture_outputs` (bool, default=True): Whether to capture function outputs
- `capture_errors` (bool, default=True): Whether to capture exceptions
- `session_id` (str, optional): Session identifier for grouping related calls
- `user_id` (str, optional): User identifier for user-scoped analytics
- `tags` (List[str], optional): List of tags for categorization
- `metadata` (Dict[str, Any], optional): Custom metadata dictionary
- `evaluation_enabled` (bool, default=True): Whether to enable automatic evaluation
- `max_input_length` (int, default=10000): Maximum length for input serialization
- `max_output_length` (int, default=10000): Maximum length for output serialization

**Returns:**
- Decorated function with comprehensive observability and AI intelligence

**Features:**
- **Automatic hierarchical tracing**: Nested function calls create proper span hierarchy
- **AI provider detection**: Automatically detects OpenAI/Anthropic usage in functions
- **Privacy controls**: Sensitive parameter redaction and length limits
- **Performance**: <1ms overhead per function call
- **Framework agnostic**: Works with any Python function

**Example:**
```python
from brokle import observe

@observe(name="ai-workflow", tags=["ai", "production"])
def process_user_query(client: OpenAI, query: str) -> str:
    # Automatically detects OpenAI usage and extracts AI metrics
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": query}]
    )
    return response.choices[0].message.content

@observe(capture_inputs=False)  # For sensitive data
def analyze_document(client: Anthropic, document: str) -> dict:
    # Privacy controls prevent input capture
    return client.messages.create(
        model="claude-3-opus-20240229",
        messages=[{"role": "user", "content": document}]
    )

# Automatic hierarchical tracing
@observe(name="parent-workflow")
def complex_workflow(data: str) -> str:
    # Parent span created automatically
    step1 = process_step_one(data)  # Child span
    step2 = process_step_two(step1)  # Child span
    return step2

@observe(name="step-one")
def process_step_one(data: str) -> str:
    # Automatically becomes child of parent-workflow
    return f"processed: {data}"

@observe(name="step-two")
def process_step_two(data: str) -> str:
    # Automatically becomes child of parent-workflow
    return f"final: {data}"
```

## `trace_workflow(name, session_id=None, user_id=None, metadata=None)`

Context manager for tracing complex workflows.

**Parameters:**
- `name` (str): Workflow name
- `session_id` (str, optional): Session identifier
- `user_id` (str, optional): User identifier
- `metadata` (Dict[str, Any], optional): Custom metadata

**Example:**
```python
from brokle import trace_workflow

with trace_workflow("user-onboarding", user_id="user123"):
    step1_result = process_signup(user_data)
    step2_result = send_welcome_email(step1_result)
    return step2_result
```

## Specialized Decorators

### `observe_llm(name=None, model=None, **kwargs) -> Callable`

Specialized decorator for LLM function calls with LLM-specific metadata.

**Example:**
```python
from brokle import observe_llm

@observe_llm(name="story-generation", model="gpt-4")
def generate_story(prompt: str) -> str:
    # LLM-specific observability
    return llm_call(prompt)
```

### `observe_retrieval(name=None, index_name=None, **kwargs) -> Callable`

Specialized decorator for retrieval/search operations.

**Example:**
```python
from brokle import observe_retrieval

@observe_retrieval(name="vector-search", index_name="documents")
def search_documents(query: str) -> List[str]:
    # Retrieval-specific observability
    return vector_db.search(query)
```

---

# Pattern 3: Native SDK API

## Core Client Classes

### `class Brokle(HTTPBase)`

Synchronous Brokle client with OpenAI-compatible interface and advanced platform features.

**Constructor Parameters:**
- `api_key` (str, optional): Brokle API key (or use BROKLE_API_KEY env var)
- `host` (str, optional): Brokle host URL (default: http://localhost:8080)
- `environment` (str, optional): Environment name (or use BROKLE_ENVIRONMENT env var)
- `timeout` (float, optional): Request timeout in seconds (default: 60)
- `**kwargs`: Additional configuration options

**Usage:**
```python
from brokle import Brokle

# Explicit configuration
with Brokle(
    api_key="bk_...",
    host="http://localhost:8080",
    environment="production"
) as client:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}],
        routing_strategy="cost_optimized"  # Brokle extension
    )
```

### `class AsyncBrokle(HTTPBase)`

Asynchronous Brokle client with identical interface to sync client.

**Usage:**
```python
from brokle import AsyncBrokle

async with AsyncBrokle(api_key="bk_...") as client:
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}],
        routing_strategy="cost_optimized",
        cache_strategy="semantic"
    )
```

### `get_client() -> Brokle`

Get or create a singleton Brokle client instance from environment variables.

**Environment Variables:**
- `BROKLE_API_KEY`: API key
- `BROKLE_HOST`: Host URL
- `BROKLE_ENVIRONMENT`: Environment name
- `BROKLE_OTEL_ENABLED`: Enable OpenTelemetry
- `BROKLE_TELEMETRY_ENABLED`: Enable telemetry
- `BROKLE_CACHE_ENABLED`: Enable caching

**Example:**
```python
from brokle import get_client

# Uses environment variables for configuration
client = get_client()
response = client.chat.completions.create(...)
```

## Resource APIs

### Chat Completions API

#### `client.chat.completions.create(**kwargs) -> ChatCompletionResponse`

Create chat completion with OpenAI-compatible interface plus Brokle extensions.

**Standard OpenAI Parameters:**
- `model` (str): Model name
- `messages` (List[Dict[str, str]]): List of messages
- `temperature` (float, optional): Sampling temperature (0-2)
- `max_tokens` (int, optional): Maximum tokens to generate
- `top_p` (float, optional): Nucleus sampling parameter (0-1)
- `frequency_penalty` (float, optional): Frequency penalty (-2 to 2)
- `presence_penalty` (float, optional): Presence penalty (-2 to 2)
- `stop` (Union[str, List[str]], optional): Stop sequences
- `stream` (bool, optional): Whether to stream response

**Brokle Extensions:**
- `routing_strategy` (str, optional): Routing strategy
  - `"cost_optimized"`: Minimize cost
  - `"latency_optimized"`: Minimize latency
  - `"quality_optimized"`: Maximize quality
  - `"balanced"`: Balance cost, latency, quality
- `cache_strategy` (str, optional): Cache strategy
  - `"semantic"`: Semantic similarity caching
  - `"exact"`: Exact match caching
  - `"disabled"`: Disable caching
- `environment` (str, optional): Environment override
- `tags` (List[str], optional): Request tags for analytics
- `**kwargs`: Additional parameters

**Response Object:**
```python
class ChatCompletionResponse(BaseModel):
    # Standard OpenAI fields
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

    # Brokle platform metadata (industry standard pattern)
    brokle: Optional[BrokleMetadata] = None

class BrokleMetadata(BaseModel):
    provider: str                    # Actual provider used
    request_id: str                 # Unique request ID
    latency_ms: float               # Response latency
    cost_usd: Optional[float]       # Estimated cost
    input_tokens: Optional[int]     # Input tokens used
    output_tokens: Optional[int]    # Output tokens used
    total_tokens: Optional[int]     # Total tokens used
    cache_hit: bool                 # Whether cache was hit
    cache_similarity_score: Optional[float] # Cache similarity score
    quality_score: Optional[float]  # Quality score (0-1)
    routing_strategy: Optional[str] # Routing strategy used
    routing_reason: Optional[str]   # Routing decision reason
```

**Example:**
```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    temperature=0.7,
    max_tokens=150,
    routing_strategy="cost_optimized",
    cache_strategy="semantic",
    tags=["chatbot", "production"]
)

print(f"Response: {response.choices[0].message.content}")

# Industry standard pattern: Access platform metadata via response.brokle.*
if response.brokle:
    print(f"Provider: {response.brokle.provider}")
    print(f"Cost: ${response.brokle.cost_usd}")
    print(f"Cache hit: {response.brokle.cache_hit}")
    print(f"Quality: {response.brokle.quality_score}")
```

### Backward Compatibility

For existing code using direct field access, compatibility properties are provided with deprecation warnings:

```python
# Legacy pattern (deprecated but still works)
response = client.chat.completions.create(...)
print(response.provider)        # Works but shows deprecation warning
print(response.cost_usd)        # Works but shows deprecation warning

# Recommended pattern (industry standard)
if response.brokle:
    print(response.brokle.provider)     # Clean, no warnings
    print(response.brokle.cost_usd)     # Clean, no warnings
```

**Available Legacy Properties:**
- `response.request_id` → `response.brokle.request_id`
- `response.provider` → `response.brokle.provider`
- `response.cost_usd` → `response.brokle.cost_usd`
- `response.cache_hit` → `response.brokle.cache_hit`
- `response.quality_score` → `response.brokle.quality_score`
- `response.input_tokens` → `response.brokle.input_tokens`
- `response.output_tokens` → `response.brokle.output_tokens`
- `response.total_tokens` → `response.brokle.total_tokens`
- `response.latency_ms` → `response.brokle.latency_ms`
- `response.routing_reason` → `response.brokle.routing_reason`

**Migration Timeline:**
- **Current**: Both patterns work, legacy shows warnings
- **Next Major Version**: Legacy patterns will be removed

**Migrating Your Code:**
```python
# Before (deprecated)
if response.provider == "openai" and response.cost_usd < 0.01:
    print(f"Cheap OpenAI request: {response.cache_hit}")

# After (recommended)
if response.brokle and response.brokle.provider == "openai" and response.brokle.cost_usd < 0.01:
    print(f"Cheap OpenAI request: {response.brokle.cache_hit}")
```

### Embeddings API

#### `client.embeddings.create(**kwargs) -> EmbeddingResponse`

Create embeddings with OpenAI-compatible interface.

**Parameters:**
- `input` (Union[str, List[str]]): Input text(s) to embed
- `model` (str): Embedding model name
- `routing_strategy` (str, optional): Brokle routing strategy
- `**kwargs`: Additional parameters

**Example:**
```python
response = client.embeddings.create(
    input=["Hello world", "How are you?"],
    model="text-embedding-ada-002",
    routing_strategy="cost_optimized"
)

embeddings = response.data[0].embedding
```

### Models API

#### `client.models.list() -> ModelListResponse`

List available models across all providers.

**Example:**
```python
models = client.models.list()
for model in models.data:
    print(f"Model: {model.id}, Provider: {model.owned_by}")
```

## Configuration Management

### `class Config`

Configuration management with validation and environment variable support.

**Attributes:**
- `api_key` (str): API key
- `host` (str): Host URL
- `environment` (str): Environment name
- `timeout` (float): Request timeout
- `telemetry_enabled` (bool): Telemetry enabled
- `otel_enabled` (bool): OpenTelemetry enabled
- `cache_enabled` (bool): Caching enabled

**Example:**
```python
from brokle import Config

config = Config(
    api_key="bk_...",
    environment="production",
    telemetry_enabled=True
)
```

### `class AuthManager`

Authentication and API key management.

**Methods:**
- `validate_api_key() -> bool`: Validate API key with backend
- `get_project_info() -> dict`: Get project information
- `refresh_token()`: Refresh authentication token

---

# Evaluation Framework API

## Core Functions

### `evaluate(data, evaluators, **kwargs) -> EvaluationResult`

Synchronous evaluation of AI responses.

**Parameters:**
- `data` (List[Dict]): Evaluation data with inputs, outputs, and expected results
- `evaluators` (List[BaseEvaluator]): List of evaluator instances
- `**kwargs`: Additional evaluation options

**Example:**
```python
from brokle import evaluate, AccuracyEvaluator, RelevanceEvaluator

data = [
    {
        "input": "What is the capital of France?",
        "output": "Paris",
        "expected": "Paris"
    }
]

result = evaluate(
    data=data,
    evaluators=[
        AccuracyEvaluator(),
        RelevanceEvaluator()
    ]
)

print(f"Accuracy: {result.metrics['accuracy']}")
print(f"Relevance: {result.metrics['relevance']}")
```

### `aevaluate(data, evaluators, **kwargs) -> EvaluationResult`

Asynchronous evaluation of AI responses.

**Example:**
```python
result = await aevaluate(
    data=data,
    evaluators=[AccuracyEvaluator(), RelevanceEvaluator()]
)
```

## Evaluator Classes

### `class AccuracyEvaluator(BaseEvaluator)`

Evaluate response accuracy against expected outputs.

### `class RelevanceEvaluator(BaseEvaluator)`

Evaluate response relevance to input queries.

### `class CostEfficiencyEvaluator(BaseEvaluator)`

Evaluate cost efficiency of AI responses.

### `class LatencyEvaluator(BaseEvaluator)`

Evaluate response latency performance.

### `class QualityEvaluator(BaseEvaluator)`

Comprehensive quality evaluation.

---

# Error Handling

## Exception Classes

### `class BrokleError(Exception)`

Base exception for all Brokle SDK errors.

### `class AuthenticationError(BrokleError)`

Raised for authentication failures.

### `class ValidationError(BrokleError)`

Raised for input validation errors.

### `class NetworkError(BrokleError)`

Raised for network connectivity issues.

### `class RateLimitError(BrokleError)`

Raised when rate limits are exceeded.

### `class ProviderError(BrokleError)`

Raised for AI provider-specific errors.

### `class CacheError(BrokleError)`

Raised for caching-related errors.

### `class EvaluationError(BrokleError)`

Raised for evaluation framework errors.

**Example Error Handling:**
```python
from brokle import Brokle, AuthenticationError, RateLimitError

try:
    with Brokle(api_key="invalid") as client:
        response = client.chat.completions.create(...)
except AuthenticationError as e:
    print(f"Authentication failed: {e}")
except RateLimitError as e:
    print(f"Rate limit exceeded: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

# Environment Configuration

## Required Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `BROKLE_API_KEY` | API key for authentication | Yes | None |

## Optional Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BROKLE_HOST` | Backend host URL | `http://localhost:8080` |
| `BROKLE_ENVIRONMENT` | Environment tag | `default` |
| `BROKLE_TELEMETRY_ENABLED` | Enable telemetry | `true` |
| `BROKLE_OTEL_ENABLED` | Enable OpenTelemetry | `true` |
| `BROKLE_CACHE_ENABLED` | Enable caching | `true` |
| `BROKLE_TIMEOUT` | Request timeout (seconds) | `60` |

## Environment Setup Example

```bash
# Required
export BROKLE_API_KEY="bk_your_api_key_here"

# Optional
export BROKLE_HOST="https://api.brokle.com"
export BROKLE_ENVIRONMENT="production"
export BROKLE_TELEMETRY_ENABLED="true"
export BROKLE_CACHE_ENABLED="true"
```

---

# Performance and Best Practices

## Performance Characteristics

- **Wrapper Functions**: <3ms overhead, maintains original client performance
- **Universal Decorator**: <1ms overhead per function call with comprehensive tracing
- **Native SDK**: Sub-100ms response times, 30-50% cost reduction through optimization

## Best Practices

### Pattern Selection
- **Start with Pattern 1**: Easy migration, zero code changes beyond imports
- **Add Pattern 2**: Enhanced observability for custom workflows
- **Scale with Pattern 3**: Full platform features for production systems

### Configuration
```python
# Production configuration
client = Brokle(
    api_key=os.getenv("BROKLE_API_KEY"),
    environment="production",
    timeout=30,  # Shorter timeout for production
)

# Always use context managers for proper cleanup
async with AsyncBrokle() as client:
    response = await client.chat.completions.create(...)
```

### Error Handling
```python
from brokle import Brokle, BrokleError
import backoff

@backoff.on_exception(backoff.expo, BrokleError, max_tries=3)
async def robust_ai_call():
    async with AsyncBrokle() as client:
        return await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            routing_strategy="quality_optimized"
        )
```

---

*Complete API Reference for Brokle Python SDK*