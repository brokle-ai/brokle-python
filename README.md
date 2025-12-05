# Brokle Python SDK

**OpenTelemetry-native observability for AI applications.**

The Brokle Python SDK provides comprehensive observability, tracing, and auto-instrumentation for LLM applications. Choose your integration level:

## 🎯 Three Integration Patterns

**Pattern 1: Wrapper Functions**
Wrap existing SDK clients (OpenAI, Anthropic) for automatic observability and platform features.

**Pattern 2: Universal Decorator**
Framework-agnostic `@observe()` decorator with automatic hierarchical tracing. Works with any AI library.

**Pattern 3: Native SDK (Sync & Async)**
Full platform capabilities with OpenAI-compatible interface. Context manager support with automatic resource cleanup.

## Installation

```bash
pip install brokle
```

### Setup

```bash
export BROKLE_API_KEY="bk_your_api_key_here"
export BROKLE_HOST="http://localhost:8080"
```

## Quick Start

### Pattern 1: Wrapper Functions

```python
# Wrap existing SDK clients for automatic observability
from openai import OpenAI
from anthropic import Anthropic
from brokle import wrap_openai, wrap_anthropic

# OpenAI wrapper
openai_client = wrap_openai(
    OpenAI(api_key="sk-..."),
    tags=["production"],
    session_id="user_session_123"
)

# Anthropic wrapper
anthropic_client = wrap_anthropic(
    Anthropic(api_key="sk-ant-..."),
    tags=["claude", "analysis"]
)

response = openai_client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
# ✅ Automatic Brokle observability and tracing
```

### Pattern 2: Universal Decorator

```python
# Automatic hierarchical tracing with just @observe()
from brokle import observe
import openai

client = openai.OpenAI()

@observe(name="parent-workflow")
def main_workflow(data: str):
    # Parent span automatically created
    result1 = analyze_data(data)
    result2 = summarize_findings(result1)
    return f"Final result: {result1} -> {result2}"

@observe(name="data-analysis")
def analyze_data(data: str):
    # Child span automatically linked to parent
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Analyze: {data}"}]
    )
    return response.choices[0].message.content

@observe(name="summarization")
def summarize_findings(analysis: str):
    # Another child span automatically linked to parent
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Summarize: {analysis}"}]
    )
    return response.choices[0].message.content

# Automatic hierarchical tracing - no manual workflow management needed
result = main_workflow("User behavior data from Q4 2024")
# ✅ Complete span hierarchy: parent -> analyze_data + summarize_findings
```

### Pattern 3: Native SDK

**Sync Client:**
```python
from brokle import Brokle

# Context manager (recommended)
with Brokle(
    api_key="bk_...",
    host="http://localhost:8080"
) as client:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}],
        tags=["production"]  # Analytics tags
    )
    print(f"Response: {response.choices[0].message.content}")
```

**Async Client:**
```python
from brokle import AsyncBrokle
import asyncio

async def main():
    async with AsyncBrokle(
        api_key="bk_...",
    ) as client:
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}],
            tags=["async", "production"]  # Analytics tags
        )
        print(f"Response: {response.choices[0].message.content}")

asyncio.run(main())
```


## Why Choose Brokle?

- **⚡ <3ms Overhead**: High-performance observability
- **📊 Complete Visibility**: Real-time analytics and quality scoring
- **🔧 OpenTelemetry Native**: Standards-based tracing and metrics
- **🛠️ Three Patterns**: Start simple, scale as needed

## Next Steps

- **📖 [Integration Patterns Guide](docs/INTEGRATION_PATTERNS_GUIDE.md)** - Detailed examples
- **⚡ [Quick Reference](docs/QUICK_REFERENCE.md)** - Fast setup guide
- **🔧 [API Reference](docs/API_REFERENCE.md)** - Complete documentation
- **💻 [Examples](examples/)** - Pattern-based code examples

## Examples

Check the `examples/` directory:
- [`pattern1_wrapper_functions.py`](examples/pattern1_wrapper_functions.py) - Wrapper functions
- [`pattern2_decorator.py`](examples/pattern2_decorator.py) - Universal decorator

## Support

- **Issues**: [GitHub Issues](https://github.com/brokle-ai/brokle-python/issues)
- **Docs**: [docs.brokle.com](https://docs.brokle.com/sdk/python)
- **Email**: support@brokle.com

---

**Simple. Powerful. OpenTelemetry-native observability for AI.**
