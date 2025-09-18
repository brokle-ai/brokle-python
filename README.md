# Brokle Platform Python SDK

A comprehensive Python SDK for the Brokle Platform that provides intelligent routing, cost optimization, semantic caching, and comprehensive observability for AI applications.

## Features

- **🔄 Intelligent Routing**: ML-based provider selection with cost, quality, and latency optimization
- **💰 Cost Optimization**: 30-50% reduction in LLM costs through smart routing
- **🧠 Semantic Caching**: Vector similarity-based response caching
- **📊 Real-time Analytics**: Comprehensive telemetry and business intelligence
- **🎯 Response Evaluation**: Automated quality assessment and feedback loops
- **🔍 OpenTelemetry Integration**: Industry-standard distributed tracing
- **⚡ Three Integration Patterns**: Drop-in replacement, decorator, and native SDK

## Installation

```bash
pip install brokle
```

### Optional Dependencies

```bash
# For OpenAI compatibility
pip install brokle[openai]

# For development
pip install brokle[dev]

# For all features
pip install brokle[all]
```

## Quick Start

### 1. Configuration

Set up your environment variables:

```bash
export BROKLE_API_KEY="ak_your_api_key_here"
export BROKLE_HOST="http://localhost:8000"
export BROKLE_PROJECT_ID="proj_your_project_id"
export BROKLE_ENVIRONMENT="production"
```

Or configure programmatically:

```python
import brokle

brokle.configure(
    api_key="ak_your_api_key_here",
    host="http://localhost:8000",
    project_id="proj_your_project_id"
)
```

### 2. OpenAI Drop-in Replacement

Zero code changes beyond import:

```python
# Before
# from openai import OpenAI

# After
from brokle.openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}],
    
    # Brokle specific features
    routing_strategy="cost_optimized",
    cache_strategy="semantic",
    max_cost_usd=0.01
)

print(response.choices[0].message.content)
print(f"Cost: ${response.cost_usd:.4f}")
print(f"Provider: {response.provider}")
```

### 3. @observe Decorator

Comprehensive observability with decorators:

```python
from brokle import observe
from brokle.openai import openai

@observe()
def generate_story(prompt: str) -> str:
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        routing_strategy="quality_optimized"
    )
    return response.choices[0].message.content

@observe(
    name="main-workflow",
    user_id="user123",
    session_id="session456",
    capture_input=True,
    capture_output=True
)
def main():
    return generate_story("Tell me a story about AI")

story = main()
print(story)
```

### 4. Native SDK (Full Features)

Access all Brokle Platform features:

```python
from brokle import Brokle
import asyncio

async def main():
    async with Brokle() as client:
        # Chat completion with advanced features
        response = await client.chat.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Explain quantum computing"}],
            routing_strategy="balanced",
            cache_strategy="semantic",
            cache_similarity_threshold=0.8,
            evaluation_metrics=["relevance", "accuracy", "clarity"],
            custom_tags={"topic": "quantum", "difficulty": "intermediate"}
        )
        
        print(f"Response: {response.choices[0].message.content}")
        print(f"Cost: ${response.cost_usd:.4f}")
        print(f"Quality Score: {response.quality_score}")
        print(f"Cached: {response.cached}")
        
        # Get analytics
        analytics = await client.analytics.get_real_time_metrics()
        print(f"Analytics: {analytics}")
        
        # Submit feedback
        await client.evaluation.submit_feedback(
            response_id=response.id,
            feedback_type="thumbs_up",
            comment="Great explanation!"
        )

asyncio.run(main())
```

## Integration Patterns

### 1. OpenAI Drop-in Replacement

Perfect for existing OpenAI codebases:

```python
from brokle.openai import OpenAI, AsyncOpenAI

# Synchronous
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    routing_strategy="cost_optimized"
)

# Asynchronous
async with AsyncOpenAI() as client:
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        routing_strategy="quality_optimized"
    )
```

### 2. Decorator Pattern

For comprehensive observability:

```python
from brokle import observe

@observe(as_type="generation")
def llm_call(prompt: str):
    # Your LLM call here
    return result

@observe(
    name="custom-name",
    user_id="user123",
    session_id="session456",
    tags=["important", "workflow"],
    metadata={"version": "1.0"}
)
def business_logic():
    # Your business logic here
    return result
```

### 3. Native SDK

For full platform access:

```python
from brokle import Brokle

async with Brokle() as client:
    # All LLM operations
    chat_response = await client.chat.create(...)
    completion_response = await client.completions.create(...)
    embedding_response = await client.embeddings.create(...)
    
    # Analytics
    metrics = await client.analytics.get_metrics(...)
    
    # Evaluation
    evaluation = await client.evaluation.evaluate_response(...)
```

## Advanced Features

### Intelligent Routing

```python
# Cost-optimized routing
response = await client.chat.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    routing_strategy="cost_optimized",
    max_cost_usd=0.01
)

# Quality-optimized routing
response = await client.chat.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Complex question"}],
    routing_strategy="quality_optimized",
    evaluation_metrics=["accuracy", "depth"]
)

# Latency-optimized routing
response = await client.chat.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Quick question"}],
    routing_strategy="latency_optimized"
)
```

### Semantic Caching

```python
response = await client.chat.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What is machine learning?"}],
    cache_strategy="semantic",
    cache_similarity_threshold=0.8,
    cache_ttl=3600  # 1 hour
)

print(f"Cache hit: {response.cache_hit}")
print(f"Similarity score: {response.cache_similarity_score}")
```

### Response Evaluation

```python
response = await client.chat.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Explain photosynthesis"}],
    evaluation_metrics=["relevance", "accuracy", "clarity", "completeness"]
)

print(f"Quality score: {response.quality_score}")
print(f"Evaluation scores: {response.evaluation_scores}")

# Submit feedback
await client.evaluation.submit_feedback(
    response_id=response.id,
    feedback_type="rating",
    feedback_value=4.5,
    comment="Very clear explanation!"
)
```

### Analytics and Monitoring

```python
from datetime import datetime, timedelta

# Real-time metrics
metrics = await client.analytics.get_real_time_metrics()

# Historical analytics
end_date = datetime.now()
start_date = end_date - timedelta(days=7)

analytics = await client.analytics.get_metrics(
    start_date=start_date.isoformat(),
    end_date=end_date.isoformat(),
    group_by=["provider", "model"],
    metrics=["request_count", "total_cost", "average_latency"],
    granularity="daily"
)
```

## Development & Testing

### Running Tests

```bash
# Install development dependencies
make install-dev

# Run all tests
make test

# Run tests with coverage
make test-coverage

# Run specific test file
make test-specific TEST=test_config.py

# Run integration tests
make integration-test
```

### Code Quality

```bash
# Format code
make format

# Run linter
make lint

# Type checking
make type-check

# Full development check
make dev-check
```

### Test Structure

The SDK includes comprehensive tests:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **Mock Tests**: Test external API interactions
- **Error Handling Tests**: Test exception scenarios
- **Async Tests**: Test asynchronous functionality

### Key Test Files

- `test_config.py` - Configuration and environment handling
- `test_client.py` - Core client functionality
- `test_decorators.py` - @observe decorator functionality
- `test_openai_client.py` - OpenAI compatibility layer
- `test_auth.py` - Authentication and authorization
- `test_exceptions.py` - Error handling and custom exceptions

## Configuration

### Environment Variables

```bash
# Core configuration
BROKLE_API_KEY="ak_your_api_key"
BROKLE_HOST="http://localhost:8000"
BROKLE_PROJECT_ID="proj_your_project_id"
BROKLE_ENVIRONMENT="production"

# OpenTelemetry
BROKLE_OTEL_ENABLED=true
BROKLE_OTEL_ENDPOINT="http://localhost:4317"
BROKLE_OTEL_SERVICE_NAME="my-app"

# Features
BROKLE_TELEMETRY_ENABLED=true
BROKLE_CACHE_ENABLED=true
BROKLE_ROUTING_ENABLED=true
BROKLE_EVALUATION_ENABLED=true

# Performance
BROKLE_TIMEOUT=30
BROKLE_MAX_RETRIES=3
BROKLE_TELEMETRY_BATCH_SIZE=100
```

### Programmatic Configuration

```python
import brokle

brokle.configure(
    api_key="ak_your_api_key",
    host="http://localhost:8000",
    project_id="proj_your_project_id",
    environment="default",
    otel_enabled=True,
    telemetry_enabled=True,
    cache_enabled=True,
    routing_enabled=True,
    evaluation_enabled=True
)
```

## OpenTelemetry Integration

The SDK provides comprehensive OpenTelemetry integration:

```python
from brokle.types.attributes import BrokleOtelSpanAttributes
from brokle.core.telemetry import get_tracer

tracer = get_tracer()

with tracer.start_as_current_span("my-operation") as span:
    # Your code here
    span.set_attribute(BrokleOtelSpanAttributes.USER_ID, "user123")
    span.set_attribute(BrokleOtelSpanAttributes.SESSION_ID, "session456")
    
    # Brokle specific attributes
    span.set_attribute(BrokleOtelSpanAttributes.ROUTING_STRATEGY, "cost_optimized")
    span.set_attribute(BrokleOtelSpanAttributes.COST_USD, 0.0045)
```

## Error Handling

```python
from brokle.utils.error_handler import (
    BrokleError,
    AuthenticationError,
    RateLimitError,
    QuotaExceededError,
    ProviderError
)

try:
    response = await client.chat.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        max_cost_usd=0.001  # Very low limit
    )
except QuotaExceededError as e:
    print(f"Quota exceeded: {e.message}")
    print(f"Current usage: {e.current_usage}")
    print(f"Quota limit: {e.quota_limit}")
except RateLimitError as e:
    print(f"Rate limited: {e.message}")
    print(f"Retry after: {e.retry_after} seconds")
except ProviderError as e:
    print(f"Provider error: {e.message}")
    print(f"Provider: {e.provider}")
    print(f"Provider error code: {e.provider_error_code}")
except BrokleError as e:
    print(f"Brokle error: {e.message}")
    print(f"Error code: {e.error_code}")
    print(f"Request ID: {e.request_id}")
```

## Examples

Check the `examples/` directory for comprehensive usage examples:

- [`openai_dropin.py`](examples/openai_dropin.py) - OpenAI drop-in replacement examples
- [`decorator_usage.py`](examples/decorator_usage.py) - @observe decorator examples
- [`native_features.py`](examples/native_features.py) - Native SDK feature examples

## Performance

The SDK is designed for high performance:

- **Sub-3ms Telemetry Overhead**: Minimal impact on response times
- **Background Processing**: Non-blocking telemetry submission
- **Connection Pooling**: Efficient HTTP connection management
- **Async Support**: Full async/await support throughout
- **Batch Processing**: Efficient bulk operations

## Development

### Setup

```bash
git clone https://github.com/brokle-ai/brokle-python
cd brokle-python
pip install -e .[dev]
```

### Running Tests

```bash
pytest
```

### Code Quality

```bash
black .
isort .
flake8 .
mypy .
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run quality checks
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: https://docs.brokle.com/sdk/python
- **Issues**: https://github.com/brokle-ai/brokle-python/issues
- **Community**: https://discord.gg/brokle
- **Email**: support@brokle.com

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and changes.

---

**Brokle Platform** - Intelligent routing, cost optimization, and comprehensive observability for AI applications.