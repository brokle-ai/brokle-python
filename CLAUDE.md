# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Brokle Platform Python SDK** - an OpenTelemetry-native SDK that provides comprehensive observability, tracing, and auto-instrumentation for AI applications. It's designed as part of the larger Brokle platform ecosystem.

**Key Features:**
- Three integration patterns: wrapper functions, @observe decorator, and native SDK
- Comprehensive observability with OpenTelemetry integration
- Hierarchical tracing with automatic span linking
- Real-time analytics and quality scoring
- Sub-3ms overhead for high-performance applications

## Development Commands

### Essential Setup
```bash
# Install for development
make install-dev

# Full development setup (includes pre-commit hooks)
make dev-setup
```

### Testing
```bash
# Run all tests
make test

# Run tests with verbose output
make test-verbose

# Run tests with coverage report
make test-coverage

# Run specific test file
make test-specific TEST=test_config.py

# Quick config tests only
make quick-test

# Integration tests (client functionality)
make integration-test
```

### Code Quality
```bash
# Format code (black + isort)
make format

# Run linter (flake8)
make lint

# Type checking (mypy)
make type-check

# Full development check (lint + type-check + coverage)
make dev-check
```

### Build and Publish
```bash
# Clean build artifacts
make clean

# Build distribution packages
make build

# Publish to test PyPI
make publish-test

# Publish to PyPI
make publish
```

## Architecture Overview

### Package Structure
The SDK is organized as a modular Python package with 53+ Python files:

```
brokle/
├── ai_platform/          # AI platform abstraction layer
├── _client/              # Core HTTP client implementation
├── evaluation/           # Response evaluation framework
├── integrations/         # Auto-instrumentation for various libraries
├── observability/        # Observability primitives (trace, span, score)
├── _task_manager/       # Background task management
├── testing/             # Testing utilities
├── types/               # Type definitions and attributes
├── _utils/              # Internal utilities
├── wrappers/            # SDK wrapper functions (OpenAI, Anthropic)
├── auth.py              # Authentication management
├── client.py            # Main Brokle client
├── config.py            # Configuration management
├── decorators.py        # @observe decorator
├── exceptions.py        # Custom exception hierarchy
└── __init__.py          # Public API exports
```

### Three Integration Patterns

1. **Wrapper Functions** (`brokle.wrappers`):
   - Explicit wrapping of existing SDK clients (OpenAI, Anthropic)
   - Usage: `wrap_openai(OpenAI(...))` or `wrap_anthropic(Anthropic(...))`
   - Automatic observability and telemetry for wrapped clients
   - Preserves original SDK API while adding Brokle features

2. **Decorator Pattern** (`brokle.decorators`):
   - `@observe()` decorator for comprehensive observability
   - Automatic telemetry and tracing for any function
   - Configurable capture options with `as_trace=True` parameter
   - Works with sync and async functions

3. **Native SDK** (`brokle.client`):
   - Full platform feature access via `Brokle` client
   - Async/await support throughout
   - Context manager support with automatic resource cleanup
   - Direct access to observability primitives (trace, span, score)

### Core Components

**Client Layer** (`client.py`, `_client/`):
- Async HTTP client with connection pooling
- Authentication and request signing
- Environment tag support

**Configuration** (`config.py`):
- Environment variable and programmatic configuration
- Validation with Pydantic models
- Support for multiple environments and projects

**Authentication** (`auth.py`):
- API key management and validation
- Project-scoped authentication
- Token refresh and error handling

**AI Platform** (`ai_platform/`):
- Quality scoring and evaluation
- Telemetry aggregation and batching
- Provider detection and metadata extraction

## Testing Strategy

### Test Files
- `test_config.py` - Configuration and environment validation
- `test_client.py` - Core client functionality
- `test_auth.py` - Authentication and authorization
- `test_wrappers.py` - Wrapper functions (OpenAI, Anthropic)
- `test_exceptions.py` - Error handling and custom exceptions

### Environment Testing
The SDK includes comprehensive environment tag validation rules:
- Max 40 characters, lowercase only
- Cannot start with "brokle" prefix
- Default environment is "default"
- Environment sent in request body (not headers)
- Authentication header: `X-API-Key`

### Manual Testing
Use the provided manual test scripts:
```bash
# Interactive testing
python test_manual.py --interactive

# Direct testing with API key
python test_manual.py --api-key "bk_your_key_here"

# Integration testing with backend
python test_integration.py --api-key "bk_your_key_here"
```

## Configuration and Environment

### Environment Variables
```bash
# Master Switch
BROKLE_ENABLED=true                            # Master switch (default: true). Set to false to completely disable SDK.

# Required (when enabled)
BROKLE_API_KEY="bk_your_secret"

# Optional Configuration
BROKLE_BASE_URL="https://api.brokle.com"      # Default: http://localhost:8080
BROKLE_ENVIRONMENT="production"                # Default: "default" (lowercase, max 40 chars)

# Tracing Control
BROKLE_TRACING_ENABLED=true                    # Enable/disable tracing (default: true)
BROKLE_SAMPLE_RATE=1.0                         # Sampling rate 0.0-1.0 (default: 1.0)

# Release Tracking
BROKLE_RELEASE="v1.2.3"                        # Release version for analytics

# Batch Configuration
BROKLE_FLUSH_AT=100                            # Max batch size before flush (default: 100)
BROKLE_FLUSH_INTERVAL=5.0                      # Max delay before flush in seconds (default: 5.0)

# Debug
BROKLE_DEBUG=false                             # Enable debug logging (default: false)

# HTTP
BROKLE_TIMEOUT=30                              # HTTP timeout in seconds (default: 30)
```

### Programmatic Configuration

The SDK offers **three initialization patterns**. Choose based on your use case:

| Pattern | Best For | Config Source |
|---------|----------|---------------|
| Direct params | Quick setup, scripts | Hardcoded values |
| `get_client()` | 12-factor apps, serverless | Environment variables |
| `BrokleConfig` | Testing, DI, programmatic | Config object |

**Pattern 1: Direct Parameters (Simplest)**
```python
from brokle import Brokle

# Minimal configuration
client = Brokle(api_key="bk_your_secret")

# Full configuration with parameters
client = Brokle(
    api_key="bk_your_secret",
    base_url="http://localhost:8080",
    environment="production",
    debug=True,                                # Enable debug logging
    tracing_enabled=True,                      # Enable tracing (False to disable)
    release="v1.2.3",                          # Release version for analytics
    sample_rate=0.5,                           # Sample 50% of traces
    mask=lambda data: mask_pii(data),          # Custom masking function
    flush_at=200,                              # Batch size before flush
    flush_interval=10.0,                       # Batch interval in seconds
    timeout=60,                                # HTTP timeout in seconds
)

# Use the client
with client.start_as_current_span("my-operation") as span:
    span.update(output="Done")

# Close when done
client.close()
```
✅ Use when: Quick scripts, explicit configuration, learning the SDK

**Pattern 2: Environment-Based Singleton (`get_client()`)**
```python
from brokle import get_client

# Get or create singleton from BROKLE_* environment variables
client = get_client()

# All calls to get_client() return the same instance
client2 = get_client()  # Same instance as above

# With overrides (env vars + explicit values)
client = get_client(release="v2.0.0")  # Override specific values

# Use the client
with client.start_as_current_span("my-operation") as span:
    span.update(output="Done")

# Singleton is automatically cleaned up on process exit
```
✅ Use when: 12-factor apps, serverless, Docker/K8s deployments

**Pattern 3: Config Object (`BrokleConfig`)**
```python
from brokle import Brokle, BrokleConfig

# Build config programmatically
config = BrokleConfig(
    api_key="bk_your_secret",
    release="v1.2.3",
    transport="grpc",
    grpc_endpoint="localhost:4317",
)
client = Brokle(config=config)

# Or from environment with modifications
config = BrokleConfig.from_env()
# Modify config before creating client
client = Brokle(config=config)
```
✅ Use when: Testing (mock config), dependency injection, dynamic configuration

**Decision Guide:**
- Need quick setup? → **Direct params**
- Running in container/serverless? → **`get_client()`**
- Writing tests or need DI? → **`BrokleConfig`**

**Common Configuration Patterns:**

*Disable Tracing (all calls become no-ops):*
```python
client = Brokle(api_key="bk_your_secret", tracing_enabled=False)
```

*Sample 10% of Traces (cost optimization):*
```python
client = Brokle(api_key="bk_your_secret", sample_rate=0.1)
```

*Custom Masking for Privacy:*
```python
import re

def mask_pii(data):
    """Mask emails and credit cards."""
    if isinstance(data, str):
        data = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', data)
        data = re.sub(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', '[CARD]', data)
    elif isinstance(data, dict):
        return {k: mask_pii(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [mask_pii(item) for item in data]
    return data

client = Brokle(api_key="bk_your_secret", mask=mask_pii)
```

**Integration with Wrappers:**
```python
from openai import OpenAI
from brokle.wrappers import wrap_openai
from brokle import get_client

# Initialize Brokle singleton from environment
get_client()  # Reads BROKLE_API_KEY, BROKLE_BASE_URL, etc.

# Wrap OpenAI client (automatically uses singleton for telemetry)
openai_client = wrap_openai(OpenAI(api_key="..."))

# All calls automatically tracked
response = openai_client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)
```

**Streaming with Token Usage:**

To capture token usage in streaming responses, you **must** pass `stream_options={"include_usage": True}`. This is an OpenAI API requirement, not a Brokle limitation.

```python
# Streaming WITHOUT usage tracking (usage will be empty)
stream = openai_client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)

# Streaming WITH usage tracking (recommended)
stream = openai_client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
    stream_options={"include_usage": True},  # Required for usage in streaming!
)

# Note: The final chunk will have empty choices - handle appropriately
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

See [OpenAI docs](https://platform.openai.com/docs/api-reference/chat/create#chat-create-stream_options) for more details.

**Integration with Decorators:**
```python
from brokle import observe, get_client

# Initialize singleton
get_client()

# Decorator automatically uses singleton for telemetry
@observe
def process_request(user_input: str):
    return f"Processed: {user_input}"

# Function calls automatically tracked
result = process_request("hello")

# Flush telemetry in short-lived applications
brokle = get_client()
brokle.flush()
```

### Verify Connection (`auth_check()`)

You can verify your connection to the Brokle server using `auth_check()`. This is useful for testing but should **not be used in production** as it adds latency.

**⚠️ Warning**: This is a synchronous blocking call that should only be used in development/testing.

**Usage**:
```python
from brokle import get_client

# Initialize client
brokle = get_client()

# Verify connection (development/testing only)
if brokle.auth_check():
    print("✅ Brokle client is authenticated and ready!")
else:
    print("❌ Authentication failed. Please check your credentials and base_url.")
```

**When to use**:
- ✅ Local development setup verification
- ✅ CLI tools and scripts
- ✅ CI/CD pipeline validation
- ❌ Production code (adds synchronous HTTP call latency)

**What happens**:
- Makes a synchronous POST request to `/v1/auth/validate-key`
- Validates your `BROKLE_API_KEY` against the backend
- Returns `True` if authenticated, `False` otherwise

**Error handling**:
```python
from brokle import Brokle

brokle = Brokle(api_key="bk_test")

try:
    if brokle.auth_check():
        print("Connection verified!")
    else:
        print("Invalid API key or server unreachable")
except Exception as e:
    print(f"Auth check failed with error: {e}")
```

## Instrumentation Patterns

The Brokle SDK provides three flexible instrumentation patterns for observability.

### 1. Observe Decorator (Automatic Tracing)

**Zero-configuration function tracing:**
```python
from brokle import observe

@observe()
def process_request(user_input: str):
    return f"Processed: {user_input}"

# Automatic tracing with input/output capture
result = process_request("hello")
```

**Full decorator options:**
```python
@observe(
    name="custom-name",                   # Override function name
    as_type="generation",                 # Type: span, generation, agent, tool, chain, retriever, etc.
    session_id="session-123",             # Group related traces
    user_id="user-456",                   # Track per-user analytics
    tags=["production", "critical"],      # Filterable tags
    metadata={"key": "value"},            # Custom metadata
    level="WARNING",                      # Log level: DEBUG, DEFAULT, WARNING, ERROR
    capture_input=True,                   # Capture function args (default: True)
    capture_output=True,                  # Capture return value (default: True)
)
async def my_operation(arg1, arg2):      # Works with sync and async
    return result
```

**Automatic nesting:**
```python
@observe(name="parent")
def parent_operation():
    # Nested calls automatically linked
    return child_operation()

@observe(name="child", as_type="tool")
def child_operation():
    return "result"
```

### 2. Context Managers (Explicit Control)

**Span context manager:**
```python
from brokle import get_client

brokle = get_client()

with brokle.start_as_current_span("operation") as span:
    result = perform_work()
    span.update(output=result, metadata={"status": "success"})
```

**Generation context manager (for LLM calls):**
```python
with brokle.start_as_current_generation(
    name="gpt4-chat",
    model="gpt-4",
    input="What is AI?",
) as generation:
    response = openai_client.chat.completions.create(...)

    generation.update(
        output=response.choices[0].message.content,
        usage={
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        },
        model_parameters={"temperature": 0.7},
    )
```

**All context manager parameters:**
```python
with brokle.start_as_current_span(
    name="operation",
    as_type="span",                       # Span type
    input="input data",
    metadata={"key": "value"},
    level="DEFAULT",
    session_id="session-123",             # Trace-level attributes
    user_id="user-456",
    tags=["tag1", "tag2"],
    model="gpt-4",                        # For generation type
    model_parameters={...},
    usage={...},
) as span:
    pass
```

**Automatic nesting:**
```python
with brokle.start_as_current_span("parent") as parent:
    # Child automatically nested
    with brokle.start_as_current_span("child") as child:
        child.update(output="child result")
    parent.update(output="parent result")
```

### 3. Manual Spans (Maximum Control)

**Manual span with explicit lifecycle:**
```python
span = brokle.start_span(name="operation", input="data")

try:
    result = perform_work()
    span.update(output=result)
except Exception as e:
    span.update(level="ERROR", metadata={"error": str(e)})
    raise
finally:
    span.end()  # Always end span
```

**Manual generation:**
```python
generation = brokle.start_generation(
    name="llm-call",
    model="gpt-4",
    input="prompt",
)

try:
    response = call_llm()
    generation.update(
        output=response,
        usage={"input_tokens": 10, "output_tokens": 20},
    )
finally:
    generation.end()
```

**Manual nesting:**
```python
parent = brokle.start_span(name="parent")
child = brokle.start_span(name="child", parent=parent)  # Explicit parent
child.end()
parent.end()
```

### Updating Spans

**Update during execution:**
```python
# Incremental updates (e.g., streaming)
with brokle.start_as_current_generation("stream") as gen:
    output = ""
    for chunk in stream:
        output += chunk
        gen.update(output=output)

# Update metadata
span.update(metadata={"status": "processing", "count": 42})

# Update level
span.update(level="ERROR" if error else "DEFAULT")
```

**Update trace-level attributes:**
```python
# Update trace attributes from any span
span.update_trace(
    session_id="new-session",
    user_id="new-user",
    tags=["additional"],              # Additive (merged)
    metadata={"trace_key": "value"},  # Merged at top level
)

# Convenience methods on client
brokle.update_current_trace(session_id="session-123", tags=["prod"])
brokle.update_current_span(output="result", metadata={"updated": True})
```

### Setting Trace Attributes

**Session ID (group related traces):**
```python
@observe(session_id="checkout-session-789")
def process_checkout():
    pass

# Or update dynamically
span.update_trace(session_id="checkout-session-789")
brokle.update_current_trace(session_id="checkout-session-789")
```

**User ID (track per-user analytics):**
```python
@observe(user_id="user-123")
def user_action():
    pass

span.update_trace(user_id="user-123")
```

**Tags (filterable labels, additive):**
```python
@observe(tags=["production", "critical"])
def operation():
    span.update_trace(tags=["experiment"])  # Merged with existing
    # Result: ["production", "critical", "experiment"]
```

**Metadata (custom key-value data, merged):**
```python
@observe(metadata={"version": "v1.0", "region": "us-east"})
def operation():
    span.update_trace(metadata={"feature": "enabled"})
    # Result: {"version": "v1.0", "region": "us-east", "feature": "enabled"}
```

### Trace Input/Output

**Decorator (automatic capture):**
```python
@observe(capture_input=True, capture_output=True)
def process(text: str, count: int):
    return text * count

# Automatically captures:
# - input: {"text": "hi", "count": 3}
# - output: "hihihi"
```

**Context manager/manual (explicit):**
```python
with brokle.start_as_current_span("op", input={"query": "..."}) as span:
    result = process()
    span.update(output=result)

# Trace-level input/output
brokle.set_trace_input({"user_query": "..."})
brokle.set_trace_output({"response": "..."})
```

### Trace and Span IDs

**Automatic ULID generation:**
```python
with brokle.start_as_current_span("op") as span:
    trace_id = span.trace_id          # "01HQXYZ..." (26 chars, sortable)
    span_id = span.span_id
```

**Custom trace ID (link to existing trace):**
```python
with brokle.start_as_current_span("op", trace_id="01HQXYZ...") as span:
    pass
```

**W3C Trace Context propagation:**
```python
trace_context = request.headers.get("traceparent")

with brokle.start_as_current_span("op", trace_context=trace_context) as span:
    pass  # Links to distributed trace
```

**Cross-service linking:**
```python
# Service A: propagate trace_id
with brokle.start_as_current_span("service-a") as span:
    requests.post(url, headers={"X-Trace-ID": span.trace_id})

# Service B: continue same trace
trace_id = request.headers.get("X-Trace-ID")
with brokle.start_as_current_span("service-b", trace_id=trace_id):
    pass
```

### Client Lifecycle Management

**Flush (send pending data):**
```python
brokle = get_client()

with brokle.start_as_current_span("op") as span:
    span.update(output="result")

brokle.flush()  # Blocks until all data sent
```

**Shutdown (flush + cleanup):**
```python
brokle.shutdown()  # Calls flush() + closes resources
```

**Context manager (automatic cleanup):**
```python
with Brokle(api_key="bk_secret") as brokle:
    with brokle.start_as_current_span("op") as span:
        pass
    # Automatically flushed and shutdown on exit
```

**Singleton pattern:**
```python
brokle = get_client()  # Singleton for long-running apps

# Use throughout application
with brokle.start_as_current_span("op1"):
    pass

brokle.flush()  # Optional periodic flush

# Singleton auto-cleaned on process exit
```

**Serverless/short-lived apps:**
```python
def lambda_handler(event, context):
    brokle = get_client()

    with brokle.start_as_current_span("lambda") as span:
        result = process(event)
        span.update(output=result)

    brokle.flush()  # CRITICAL: flush before exit
    return result
```

**Best practices:**
- ✅ Use context manager for automatic cleanup
- ✅ Call `flush()` before exit in serverless/CLI apps
- ✅ Use singleton for long-running applications
- ✅ Configure `flush_at`/`flush_interval` for throughput
- ❌ Don't call `shutdown()` on singleton in long-running apps

## Key Development Patterns

### Error Handling
The SDK uses a comprehensive exception hierarchy:
- `BrokleError` - Base exception
- `AuthenticationError` - Auth failures
- `ValidationError` - Input validation
- `ProviderError` - LLM provider issues
- `QuotaExceededError` - Usage limits
- `RateLimitError` - Rate limiting

### Async/Await Support
All client operations support async/await:
```python
async with Brokle() as client:
    response = await client.chat.create(...)
```

### OpenTelemetry Integration
Comprehensive tracing with custom attributes:
```python
from brokle.types.attributes import BrokleOtelSpanAttributes
span.set_attribute(BrokleOtelSpanAttributes.BROKLE_SESSION_ID, "session-123")
```

## Dependencies and Build System

### Core Dependencies
- `httpx>=0.25.0` - HTTP client
- `pydantic>=2.0.0` - Data validation
- `opentelemetry-*` - Comprehensive observability
- `python-dotenv>=1.0.0` - Environment management
- `backoff>=2.2.1` - Retry logic

### Optional Dependencies
- `openai[>=1.0.0]` - OpenAI wrapper support
- `anthropic>=0.5.0` - Anthropic wrapper support
- `google-generativeai>=0.3.0` - Google AI integration

### Development Tools
- `pytest` with asyncio support and coverage
- `black` + `isort` for formatting
- `flake8` for linting
- `mypy` for type checking
- `pre-commit` for git hooks

## Performance Considerations

- **Sub-3ms Telemetry Overhead**: Minimal impact on response times
- **Connection Pooling**: Efficient HTTP connection management
- **Background Processing**: Non-blocking telemetry submission
- **Batch Operations**: Efficient bulk processing

## Backend Integration

This SDK is designed to work with the Brokle platform backend:
- Default backend: `http://localhost:8080`
- Authentication via `/v1/auth/validate-key`
- Environment tags in request headers
- Project-scoped API key validation

The SDK validates environment tags using rules and sends appropriate headers to the backend for proper request routing and scoping.
