# Brokle SDK - Quick Reference

## 🎯 Three Integration Patterns

### Pattern 1: Wrapper Functions (Explicit Enhancement)
**Best for**: Migrating existing applications with zero code changes

```python
from openai import OpenAI
from anthropic import Anthropic
from brokle import wrap_openai, wrap_anthropic

# Wrap existing clients
openai_client = wrap_openai(OpenAI(api_key="sk-..."), tags=["production"])
anthropic_client = wrap_anthropic(Anthropic(api_key="sk-ant-..."), tags=["claude"])

# Use exactly like normal clients
response = openai_client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
# ✨ Enhanced with comprehensive observability
```

### Pattern 2: Universal Decorator (Framework-Agnostic)
**Best for**: Custom workflows with automatic hierarchical tracing

```python
from brokle import observe
import openai

client = openai.OpenAI()

@observe(name="parent-workflow")
def my_ai_workflow(prompt: str) -> str:
    """Parent span created automatically."""
    # Child spans created automatically from nested calls
    analysis = analyze_input(prompt)
    result = generate_response(analysis)
    return result

@observe(name="input-analysis")
def analyze_input(data: str) -> str:
    """Child span - automatically linked to parent."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Analyze: {data}"}]
    )
    return response.choices[0].message.content

@observe(name="response-generation")
def generate_response(analysis: str) -> str:
    """Child span - automatically linked to parent."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Generate response: {analysis}"}]
    )
    return response.choices[0].message.content

# Automatic hierarchical tracing - no manual workflow management needed
result = my_ai_workflow("User input data")
# ✨ Complete span hierarchy: parent → analyze_input + generate_response
```

### Pattern 3: Native SDK (Full Platform Features)
**Best for**: New applications with intelligent routing and optimization

```python
from brokle import Brokle, AsyncBrokle

# Sync client with context manager
with Brokle(
    api_key="ak_...",
    project_id="proj_...",
    host="http://localhost:8080"
) as client:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}],
        routing_strategy="cost_optimized",  # Smart routing
        cache_strategy="semantic",          # 30-50% cost reduction
        tags=["production", "chatbot"]
    )

# Async client
import asyncio

async def main():
    async with AsyncBrokle(api_key="ak_...") as client:
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello async!"}],
            routing_strategy="latency_optimized",
            cache_strategy="semantic"
        )
        return response.choices[0].message.content

asyncio.run(main())
```

---

## 🔧 Configuration

### Environment Variables

```bash
# Required
export BROKLE_API_KEY="ak_your_api_key_here"
export BROKLE_PROJECT_ID="proj_your_project_id"

# Optional
export BROKLE_HOST="http://localhost:8080"
export BROKLE_ENVIRONMENT="production"
export BROKLE_TELEMETRY_ENABLED="true"
export BROKLE_OTEL_ENABLED="true"
export BROKLE_CACHE_ENABLED="true"
export BROKLE_TIMEOUT="60"
```

### Programmatic Configuration

```python
from brokle import Brokle, get_client

# Explicit configuration
client = Brokle(
    api_key="ak_...",
    project_id="proj_...",
    host="http://localhost:8080",
    environment="production",
    timeout=30
)

# Environment-based singleton (uses BROKLE_* env vars)
client = get_client()

# Wrapper configuration
from brokle import wrap_openai
from openai import OpenAI

wrapped_client = wrap_openai(
    OpenAI(api_key="sk-..."),
    capture_content=True,
    capture_metadata=True,
    tags=["production", "ai"],
    session_id="session_123",
    user_id="user_456"
)
```

---

## 🚀 Pattern Benefits & Use Cases

### Pattern 1: Wrapper Functions
- ✅ **Zero Code Changes**: Beyond import and wrapping
- ✅ **Perfect Migration**: Drop-in replacement for existing code
- ✅ **Multi-Provider**: Works with OpenAI, Anthropic, etc.
- ✅ **Instant Observability**: Comprehensive AI metrics immediately

**Use Cases:**
- Migrating existing applications
- Legacy code enhancement
- Team training (familiar interfaces)
- Quick observability wins

### Pattern 2: Universal Decorator
- ✅ **Framework Agnostic**: Works with any AI library or custom logic
- ✅ **Automatic Hierarchical Tracing**: No manual span management
- ✅ **AI Intelligence**: Auto-detects OpenAI/Anthropic usage
- ✅ **Privacy Controls**: Configurable input/output capture

**Use Cases:**
- Custom business logic observability
- Complex multi-step AI workflows
- Framework-independent tracing
- Workflow optimization analysis

### Pattern 3: Native SDK
- ✅ **Intelligent Routing**: 250+ providers with smart selection
- ✅ **30-50% Cost Reduction**: Semantic caching and optimization
- ✅ **Quality Scoring**: Built-in response evaluation
- ✅ **Production Scale**: Enterprise-grade performance

**Use Cases:**
- New applications built from scratch
- Cost-sensitive production systems
- Quality-critical AI applications
- Advanced routing and optimization

---

## 🔍 Common Issues & Solutions

### Pattern 1: Import Order & Configuration

```python
# ❌ Wrong - missing wrapper
from openai import OpenAI
client = OpenAI()  # No Brokle enhancement

# ✅ Correct - with wrapper
from openai import OpenAI
from brokle import wrap_openai
client = wrap_openai(OpenAI(api_key="sk-..."))

# ❌ Wrong - double wrapping
client = wrap_openai(wrap_openai(OpenAI()))  # Warning issued

# ✅ Correct - check if already wrapped
if not hasattr(client, '_brokle_instrumented'):
    client = wrap_openai(client)
```

### Pattern 2: Decorator Configuration

```python
# ❌ Wrong - deprecated parameters
@observe(as_type="generation")  # Not supported
def my_function():
    pass

# ✅ Correct - current syntax
@observe(
    name="llm-generation",
    tags=["generation", "ai"],
    capture_inputs=True,
    capture_outputs=True
)
def my_function():
    pass

# Privacy controls
@observe(capture_inputs=False)  # For sensitive data
def process_sensitive(api_key: str):
    pass
```

### Pattern 3: Context Management

```python
# ❌ Wrong - missing context manager
client = Brokle(api_key="ak_...")
response = client.chat.completions.create(...)
# Missing cleanup

# ✅ Correct - with context manager
with Brokle(api_key="ak_...") as client:
    response = client.chat.completions.create(...)
# Automatic cleanup

# ✅ Correct - manual cleanup
client = Brokle(api_key="ak_...")
try:
    response = client.chat.completions.create(...)
finally:
    client.close()
```

### Environment Configuration Issues

```python
# Check configuration
from brokle import get_client

try:
    client = get_client()
    print(f"API Key: {client.config.api_key[:10]}...")
    print(f"Project: {client.config.project_id}")
    print(f"Host: {client.config.host}")
except Exception as e:
    print(f"Configuration error: {e}")

# Test connection
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Test connection"}],
    max_tokens=10
)
print("✅ Connection successful")
```

### Debugging Observability

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check telemetry status
from brokle import get_client
client = get_client()
print(f"Telemetry enabled: {client.config.telemetry_enabled}")
print(f"OpenTelemetry enabled: {client.config.otel_enabled}")

# Verify spans are being created
from brokle import observe

@observe(name="debug-test")
def test_function():
    print("Function executed")
    return "test result"

result = test_function()
# Check logs for span creation messages
```

---

## ⚡ Performance Tips

### Caching Strategy

```python
# High cache hit scenarios
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What is Python?"}],  # Common question
    cache_strategy="semantic",      # Matches similar questions
    temperature=0.7,                # Consistent temperature
    tags=["faq", "education"]
)

# Low cache hit scenarios
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Write unique creative story"}],
    cache_strategy="disabled",      # Don't cache unique content
    temperature=1.0,                # High creativity
    tags=["creative", "unique"]
)
```

### Routing Optimization

```python
# Cost-optimized routing
summary = client.chat.completions.create(
    model="gpt-4",  # May route to cheaper equivalent
    messages=[{"role": "user", "content": "Summarize this document..."}],
    routing_strategy="cost_optimized",
    max_tokens=200  # Shorter = cheaper
)

# Quality-optimized routing
analysis = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Technical analysis required..."}],
    routing_strategy="quality_optimized",
    temperature=0.3  # More deterministic
)

# Latency-optimized routing
quick_response = client.chat.completions.create(
    model="gpt-3.5-turbo",  # Faster model
    messages=[{"role": "user", "content": "Quick question..."}],
    routing_strategy="latency_optimized",
    max_tokens=50
)
```

### Async Performance

```python
import asyncio
from brokle import AsyncBrokle

async def batch_requests():
    async with AsyncBrokle() as client:
        # Concurrent requests for better throughput
        tasks = [
            client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": f"Question {i}"}],
                routing_strategy="cost_optimized"
            )
            for i in range(10)
        ]

        # Process with concurrency limit
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent

        async def limited_request(task):
            async with semaphore:
                return await task

        results = await asyncio.gather(
            *[limited_request(task) for task in tasks]
        )

        return results

# Run batch processing
results = asyncio.run(batch_requests())
```

---

## 💡 Quick Tips

### Pattern Selection Strategy
1. **Start with Pattern 1**: Easy migration, zero learning curve
2. **Add Pattern 2**: Enhanced observability for business logic
3. **Scale with Pattern 3**: Full platform features when ready

### Migration Path
```python
# Stage 1: Wrapper functions (immediate value)
client = wrap_openai(OpenAI())

# Stage 2: Add decorators (enhanced tracing)
@observe()
def my_workflow():
    return client.chat.completions.create(...)

# Stage 3: Native SDK (full optimization)
with Brokle() as native_client:
    return native_client.chat.completions.create(
        routing_strategy="cost_optimized"
    )
```

### Error Handling

```python
from brokle import BrokleError, AuthenticationError, RateLimitError

try:
    response = client.chat.completions.create(...)
except AuthenticationError:
    print("Check your API keys")
except RateLimitError:
    print("Rate limit exceeded, implement backoff")
except BrokleError as e:
    print(f"Brokle error: {e}")
```

### Environment Setup Validation

```bash
# Quick validation script
python -c "
from brokle import get_client
try:
    client = get_client()
    print('✅ Configuration valid')
    print(f'Host: {client.config.host}')
    print(f'Project: {client.config.project_id}')
except Exception as e:
    print(f'❌ Configuration error: {e}')
"
```

---

## 🔗 Next Steps

- **📖 [Complete API Reference](API_REFERENCE.md)** - Full documentation
- **🎯 [Integration Patterns Guide](INTEGRATION_PATTERNS_GUIDE.md)** - Detailed examples
- **💻 [Example Files](../examples/)** - Working code examples

---

**Simple. Powerful. Three patterns to fit your needs.**

*Choose your pattern, configure once, and scale as needed with comprehensive AI observability and optimization.*