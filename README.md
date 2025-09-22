# Brokle Platform Python SDK

**Three integration patterns. One powerful platform.**

The Brokle Python SDK provides intelligent routing across 250+ LLM providers, semantic caching (30-50% cost reduction), and comprehensive observability. Choose your integration level:

## 🎯 Three Patterns

**Pattern 1: Drop-in Replacement**
Zero code changes beyond import. Perfect for existing OpenAI codebases.

**Pattern 2: Universal Decorator**
Framework-agnostic `@observe()` decorator. Works with any AI library.

**Pattern 3: Native SDK**
Full platform capabilities: intelligent routing, semantic caching, cost optimization.

## Installation

```bash
pip install brokle
```

### Setup

```bash
export BROKLE_API_KEY="ak_your_api_key_here"
export BROKLE_PROJECT_ID="proj_your_project_id"
export BROKLE_HOST="http://localhost:8080"
```

## Quick Start

### Pattern 1: Drop-in Replacement

```python
# Zero code changes beyond import
from brokle.openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
# Automatically tracked: cost, tokens, performance
```

### Pattern 2: Universal Decorator

```python
from brokle import observe

@observe()
def ai_workflow(prompt: str):
    # Any AI library call gets tracked
    return some_llm_call(prompt)

result = ai_workflow("Analyze this data")
# Complete workflow observability
```

### Pattern 3: Native SDK

```python
from brokle import Brokle
import asyncio

async def main():
    async with Brokle() as client:
        response = await client.chat.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}],
            routing_strategy="cost_optimized",  # Smart routing
            cache_strategy="semantic"           # 30-50% savings
        )
        print(f"Cost: ${response.cost_usd:.4f}")

asyncio.run(main())
```


## Why Choose Brokle?

- **🚀 30-50% Cost Reduction**: Intelligent routing and semantic caching
- **⚡ <3ms Overhead**: High-performance observability
- **🔄 250+ Providers**: Route across all major LLM providers
- **📊 Complete Visibility**: Real-time analytics and quality scoring
- **🛠️ Three Patterns**: Start simple, scale as needed

## Next Steps

- **📖 [Integration Patterns Guide](docs/INTEGRATION_PATTERNS_GUIDE.md)** - Detailed examples
- **⚡ [Quick Reference](docs/QUICK_REFERENCE.md)** - Fast setup guide
- **🔧 [API Reference](docs/API_REFERENCE.md)** - Complete documentation
- **💻 [Examples](examples/)** - Pattern-based code examples

## Examples

Check the `examples/` directory:
- [`pattern1_openai_dropin.py`](examples/pattern1_openai_dropin.py) - Drop-in replacement
- [`pattern2_decorator.py`](examples/pattern2_decorator.py) - Universal decorator
- [`pattern3_native_sdk.py`](examples/pattern3_native_sdk.py) - Native SDK features

## Support

- **Issues**: [GitHub Issues](https://github.com/brokle-ai/brokle-python/issues)
- **Docs**: [docs.brokle.com](https://docs.brokle.com/sdk/python)
- **Email**: support@brokle.com

---

**Simple. Powerful. Three patterns to fit your needs.**
