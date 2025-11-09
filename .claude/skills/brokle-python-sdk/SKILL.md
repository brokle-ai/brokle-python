---
name: brokle-python-sdk
description: Use this skill when developing, debugging, or implementing features for the Brokle Python SDK. This includes @observe decorator, context managers (start_as_current_span, start_as_current_generation), wrappers (wrap_openai, wrap_anthropic), client configuration, OTEL architecture, or working with the Python SDK codebase. Triggers: Python SDK, brokle python, @observe, get_client(), wrap_openai, GenAI attributes, OTLP.
---

# Brokle Python SDK Development Skill

Comprehensive guidance for developing the Brokle Python SDK - an OTEL-native observability SDK for AI applications.

## Overview

The Brokle Python SDK is built on **OpenTelemetry SDK** with full GenAI 1.28+ semantic conventions compliance. It provides three integration patterns for comprehensive observability of AI applications.

**Architecture**: OTEL-native with TracerProvider → BatchSpanProcessor → OTLP/HTTP Exporter (Protobuf+Gzip)

## Public API Surface

From `brokle/__init__.py`:
```python
# Core classes
Brokle              # Main client class
BrokleConfig        # Configuration dataclass

# Client functions
get_client()        # Singleton pattern with env vars
reset_client()      # Testing utility

# Decorators
observe             # @observe decorator for functions

# Attributes (GenAI 1.28+ compliant)
Attrs               # Convenience alias
BrokleOtelSpanAttributes  # Full attribute constants

# Types
ObservationType     # generation, span, event, tool, chain, retriever
ObservationLevel    # DEBUG, DEFAULT, WARNING, ERROR
LLMProvider         # openai, anthropic, google, cohere, etc.
OperationType       # chat, embeddings, completion
ScoreDataType       # numeric, boolean, categorical
```

## OTEL-Native Architecture

### Architecture Flow
```
User Code
  ↓
@observe / start_as_current_span()
  ↓
TracerProvider (TraceIdRatioBased sampler)
  ↓
BrokleSpanProcessor (BatchSpanProcessor)
  ↓
OTLPSpanExporter (Protobuf + Gzip)
  ↓
HTTP POST /v1/otlp/traces
  ↓
Brokle Backend
```

### Key Components

**TracerProvider Setup**:
- Resource with service name and attributes
- TraceIdRatioBased sampler (trace-level sampling, not span-level)
- Registered globally for OTEL ecosystem compatibility

**BatchSpanProcessor Configuration**:
- `max_queue_size`: 2048 spans
- `schedule_delay_millis`: `flush_interval * 1000` (default 5000ms)
- `max_export_batch_size`: `flush_at` (default 100)
- `export_timeout_millis`: `timeout * 1000` (default 30000ms)

**OTLP/HTTP Exporter**:
- Endpoint: `{base_url}/v1/otlp/traces`
- Headers: `X-API-Key`, `X-Brokle-Environment`
- Compression: Gzip (automatic)
- Format: Protobuf

**Automatic Lifecycle**:
- Atexit handler registered for cleanup
- Automatic flush on process exit
- No manual shutdown required in long-running apps

## Three Integration Patterns

### Pattern 1: @observe Decorator (Zero Config)

**Basic Usage**:
```python
from brokle import observe

@observe()
def process_request(user_input: str) -> str:
    return f"Processed: {user_input}"

# Automatic tracing with input/output capture
result = process_request("hello")
```

**Full Options**:
```python
@observe(
    name="custom-operation-name",     # Override function name
    as_type="generation",              # Type: span, generation, event, tool, chain, retriever
    session_id="session-123",          # Group related traces
    user_id="user-456",                # Track per-user analytics
    tags=["production", "critical"],   # Filterable tags
    metadata={"feature": "checkout"},  # Custom metadata
    level="WARNING",                   # Log level: DEBUG, DEFAULT, WARNING, ERROR
    version="1.0",                     # A/B testing support
    capture_input=True,                # Capture function args (default: True)
    capture_output=True,               # Capture return value (default: True)
)
async def my_operation(arg1: str, arg2: int) -> dict:
    # Works with sync and async functions
    return {"result": "success"}
```

**Automatic Nesting**:
```python
@observe(name="parent")
def parent_operation():
    # Child automatically nested under parent
    return child_operation()

@observe(name="child", as_type="tool")
def child_operation():
    return "result"
```

### Pattern 2: Context Managers (Explicit Control)

**Generic Span**:
```python
from brokle import get_client

client = get_client()

with client.start_as_current_span(
    name="my-operation",
    version="1.0",  # A/B testing
) as span:
    result = perform_work()
    span.set_attribute("output", result)
    span.set_attribute(Attrs.USER_ID, "user-123")
```

**LLM Generation (GenAI 1.28+ Compliant)**:
```python
with client.start_as_current_generation(
    name="gpt4-chat",
    model="gpt-4",
    provider="openai",
    input_messages=[{"role": "user", "content": "What is AI?"}],
    model_parameters={"temperature": 0.7, "max_tokens": 100},
    version="1.0",  # A/B testing
) as generation:
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "What is AI?"}],
        temperature=0.7,
    )

    generation.set_attribute(
        Attrs.GEN_AI_OUTPUT_MESSAGES,
        json.dumps([{"role": "assistant", "content": response.choices[0].message.content}])
    )
    generation.set_attribute(Attrs.GEN_AI_USAGE_INPUT_TOKENS, response.usage.prompt_tokens)
    generation.set_attribute(Attrs.GEN_AI_USAGE_OUTPUT_TOKENS, response.usage.completion_tokens)
```

**Automatic Nesting**:
```python
with client.start_as_current_span("parent") as parent:
    # Child automatically nested
    with client.start_as_current_span("child") as child:
        child.set_attribute("child_data", "value")
    parent.set_attribute("parent_data", "value")
```

### Pattern 3: OpenAI/Anthropic Wrappers

**OpenAI Wrapper**:
```python
from openai import OpenAI
from brokle.wrappers import wrap_openai
from brokle import get_client

# Initialize Brokle singleton
get_client()

# Wrap OpenAI client
openai_client = wrap_openai(OpenAI(api_key="sk-..."))

# All calls automatically tracked with GenAI 1.28+ attributes
response = openai_client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
)
# Automatically captures: model, provider, messages, tokens, parameters
```

**Anthropic Wrapper**:
```python
from anthropic import Anthropic
from brokle.wrappers import wrap_anthropic
from brokle import get_client

get_client()
anthropic_client = wrap_anthropic(Anthropic(api_key="sk-ant-..."))

response = anthropic_client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
)
```

## Configuration

### Two Configuration Patterns

**Pattern 1: Explicit Configuration**
```python
from brokle import Brokle

client = Brokle(
    api_key="bk_your_secret",
    base_url="http://localhost:8080",
    environment="production",
    debug=True,
    tracing_enabled=True,
    release="v1.2.3",
    sample_rate=0.1,              # Sample 10% of traces
    mask=lambda data: mask_pii(data),  # Custom masking function
    flush_at=200,                 # Batch size before flush
    flush_interval=10.0,          # Batch interval in seconds
    timeout=60,                   # HTTP timeout in seconds
)
```

**Pattern 2: Environment-Based Singleton**
```python
from brokle import get_client

# Reads from BROKLE_* environment variables
client = get_client()

# All subsequent calls return same instance
client2 = get_client()  # Same instance
```

### Environment Variables

```bash
# Required
BROKLE_API_KEY=bk_your_secret

# Optional
BROKLE_BASE_URL=http://localhost:8080
BROKLE_ENVIRONMENT=production
BROKLE_RELEASE=v1.2.3
BROKLE_TRACING_ENABLED=true
BROKLE_SAMPLE_RATE=1.0              # 0.0 to 1.0
BROKLE_DEBUG=false
BROKLE_FLUSH_AT=100                 # Batch size
BROKLE_FLUSH_INTERVAL=5.0           # Seconds
BROKLE_TIMEOUT=30                   # Seconds
```

### Validation Rules

**API Key**:
- Format: `bk_` + 40 alphanumeric characters (43 total)
- Example: `bk_1234567890abcdefghijklmnopqrstuvwxyz1234`

**Environment**:
- Max 40 characters
- Alphanumeric + hyphens + underscores only
- Default: `"default"`

**Sample Rate**:
- Range: 0.0 to 1.0
- 0.0 = sample nothing, 1.0 = sample everything
- Trace-level sampling (entire traces sampled together)

**Flush Configuration**:
- `flush_at`: 1 to 1000 spans
- `flush_interval`: 0.1 to 60.0 seconds

## GenAI 1.28+ Attributes (OTEL Standard)

### Provider & Operation
```python
from brokle import Attrs

# OTEL Standard Attributes
Attrs.GEN_AI_PROVIDER_NAME        # "openai", "anthropic", "google"
Attrs.GEN_AI_OPERATION_NAME       # "chat", "embeddings", "completion"
```

### Request Parameters
```python
Attrs.GEN_AI_REQUEST_MODEL        # "gpt-4", "claude-3-sonnet"
Attrs.GEN_AI_REQUEST_TEMPERATURE
Attrs.GEN_AI_REQUEST_MAX_TOKENS
Attrs.GEN_AI_REQUEST_TOP_P
Attrs.GEN_AI_REQUEST_FREQUENCY_PENALTY
Attrs.GEN_AI_REQUEST_PRESENCE_PENALTY
```

### Messages (JSON Format)
```python
# Input messages
Attrs.GEN_AI_INPUT_MESSAGES
# Format: JSON array
# Example: '[{"role": "user", "content": "Hello"}]'

# Output messages
Attrs.GEN_AI_OUTPUT_MESSAGES
# Format: JSON array
# Example: '[{"role": "assistant", "content": "Hi there!"}]'

# System instructions
Attrs.GEN_AI_SYSTEM_INSTRUCTIONS
```

### Usage Metrics
```python
Attrs.GEN_AI_USAGE_INPUT_TOKENS
Attrs.GEN_AI_USAGE_OUTPUT_TOKENS
```

### Brokle Custom Attributes
```python
# Observation type
Attrs.BROKLE_OBSERVATION_TYPE     # generation, span, event, tool

# Usage metrics
Attrs.BROKLE_USAGE_TOTAL_TOKENS
Attrs.BROKLE_USAGE_LATENCY_MS

# Environment & versioning
Attrs.BROKLE_ENVIRONMENT          # Production, staging, dev
Attrs.BROKLE_VERSION              # A/B testing support
Attrs.BROKLE_RELEASE              # Deployment tracking

# Filterable metadata (root-level)
Attrs.USER_ID                     # user.id (OTEL standard)
Attrs.SESSION_ID                  # session.id (OTEL standard)
Attrs.TAGS                        # tags (array)
Attrs.METADATA                    # metadata (object)
```

## Common Usage Patterns

### Nested Spans (Automatic Hierarchy)
```python
with client.start_as_current_span("parent") as parent:
    parent.set_attribute("parent_data", "value")

    # Child automatically nested under parent
    with client.start_as_current_span("child") as child:
        child.set_attribute("child_data", "value")
```

### Serverless/CLI Pattern
```python
def lambda_handler(event, context):
    client = get_client()

    with client.start_as_current_span("lambda-handler", version="1.0") as span:
        result = process_event(event)
        span.set_attribute("result", result)

    # CRITICAL: flush before exit in serverless/short-lived apps
    client.flush()

    return result
```

### Version Attribute (A/B Testing)
```python
# All span creation methods support version parameter
with client.start_as_current_span("operation", version="v2") as span:
    # Test new algorithm
    result = new_algorithm()

# Decorator support
@observe(version="experiment-1")
def experimental_feature():
    pass

# Generation support
with client.start_as_current_generation(
    name="chat",
    model="gpt-4",
    provider="openai",
    version="control",  # A/B test control group
) as gen:
    pass
```

### Custom PII Masking
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

client = Brokle(api_key="bk_...", mask=mask_pii)
```

## Development Commands

### Installation
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

# Run with verbose output
make test-verbose

# Run with coverage report
make test-coverage

# Run specific test file
make test-specific TEST=test_config.py

# Integration tests
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

## Testing Patterns

### Unit Tests with pytest
```python
import pytest
from brokle import Brokle, BrokleConfig, get_client, reset_client

def test_api_key_validation():
    """Test API key format validation."""
    with pytest.raises(ValueError, match="must start with 'bk_'"):
        Brokle(api_key="invalid_key")

def test_client_initialization():
    """Test client initializes with valid config."""
    client = Brokle(api_key="bk_test_key_1234567890123456789012345678")
    assert client.config.api_key.startswith("bk_")

def test_singleton_pattern():
    """Test get_client returns same instance."""
    reset_client()  # Clean state
    client1 = get_client()
    client2 = get_client()
    assert client1 is client2

@pytest.fixture
def brokle_client():
    """Fixture for Brokle client."""
    client = Brokle(
        api_key="bk_test_1234567890123456789012345678901234567890",
        tracing_enabled=False,  # Disable for tests
    )
    yield client
    # Cleanup after test
    client.shutdown()
```

### Decorator Tests
```python
from brokle import observe

@observe(name="test-function")
def decorated_function(x: int) -> int:
    return x * 2

def test_decorator_execution():
    """Test decorator doesn't break function execution."""
    result = decorated_function(5)
    assert result == 10

@observe(capture_input=True, capture_output=True)
async def async_function(x: int) -> int:
    return x * 2

@pytest.mark.asyncio
async def test_async_decorator():
    """Test decorator works with async functions."""
    result = await async_function(5)
    assert result == 10
```

## Troubleshooting

### API Key Issues
**Problem**: `ValueError: Invalid API key format`
**Solution**: Ensure API key format is `bk_` + 40 alphanumeric characters (43 total)

### Environment Tag Validation
**Problem**: `ValueError: Invalid environment tag`
**Solution**:
- Max 40 characters
- Only alphanumeric, hyphens, underscores
- Cannot start with "brokle" prefix

### Trace-Level Sampling
**Problem**: Some spans missing from traces
**Solution**: SDK uses trace-level sampling (TraceIdRatioBased), not span-level. Entire traces are sampled together. If `sample_rate=0.1`, 10% of complete traces are kept.

### Serverless Flush
**Problem**: Data not appearing in Brokle
**Solution**: Always call `client.flush()` before function exit in serverless/CLI apps:
```python
def lambda_handler(event, context):
    client = get_client()
    # ... processing ...
    client.flush()  # CRITICAL
    return result
```

### Atexit Cleanup
**Problem**: Spans not exported on program exit
**Solution**: SDK automatically registers atexit handler. No manual cleanup needed for long-running apps. For short-lived apps, use `client.flush()`.

## Key Architecture Decisions

1. **OTEL-Native**: Built on OpenTelemetry SDK (not custom implementation)
2. **Trace-Level Sampling**: TraceIdRatioBased ensures entire traces sampled together
3. **Batch Processing**: BatchSpanProcessor for efficiency
4. **OTLP/HTTP**: Industry-standard export protocol with Protobuf + Gzip
5. **Automatic Cleanup**: Atexit handler for graceful shutdown
6. **Version Support**: A/B testing via version attribute in all span methods

## Reference

- **SDK Location**: `sdk/python/`
- **Package Structure**: `brokle/` (53+ Python files)
- **Documentation**: `CLAUDE.md`, `README.md`
- **Examples**: `examples/` directory
- **Tests**: `tests/` directory
