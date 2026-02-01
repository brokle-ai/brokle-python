# Brokle Python SDK

**OpenTelemetry-native observability for AI applications.**

The Brokle Python SDK provides comprehensive observability, tracing, and evaluation for LLM applications.

## ⚡ Quick Start: Evaluation

Run evaluations with a single function call (similar to Braintrust/LangSmith):

```python
from brokle import evaluate, ExactMatch, Contains

# Define your task
def my_llm_task(item):
    # Your LLM call here
    return {"output": f"Response to: {item['input']}"}

# Run evaluation
results = evaluate(
    task=my_llm_task,
    data=[
        {"input": "Hello", "expected": "Response to: Hello"},
        {"input": "World", "expected": "Response to: World"},
    ],
    evaluators=[ExactMatch(), Contains(substring="Response")],
    experiment_name="my-first-eval",
)

print(f"Experiment: {results.experiment_name}")
print(f"View at: {results.url}")
for name, stats in results.summary.items():
    print(f"  {name}: mean={stats['mean']:.3f}")
```

## 🎯 Four Integration Patterns

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

## Privacy and Data Masking

Brokle supports client-side data masking to protect sensitive information before transmission. Masking is applied to input/output data and metadata **before** it leaves your application.

### Basic Usage

```python
import re
from brokle import Brokle

def mask_emails(data):
    """Mask email addresses in any data structure."""
    if isinstance(data, str):
        return re.sub(r'\b[\w.]+@[\w.]+\b', '[EMAIL]', data)
    elif isinstance(data, dict):
        return {k: mask_emails(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [mask_emails(item) for item in data]
    return data

# Configure masking at client initialization
client = Brokle(api_key="bk_secret", mask=mask_emails)

# All input/output automatically masked
with client.start_as_current_span(
    "process",
    input="Contact john@example.com"
) as span:
    pass
# Transmitted as: input="Contact [EMAIL]"
```

### Using Built-in Helpers

The SDK includes pre-built masking utilities for common PII patterns:

```python
from brokle import Brokle
from brokle.utils.masking import MaskingHelper

# Option 1: Mask all common PII (recommended)
client = Brokle(
    api_key="bk_secret",
    mask=MaskingHelper.mask_pii  # Masks emails, phones, SSN, credit cards, API keys
)

# Option 2: Mask specific PII types
client = Brokle(api_key="bk_secret", mask=MaskingHelper.mask_emails)
client = Brokle(api_key="bk_secret", mask=MaskingHelper.mask_phones)
client = Brokle(api_key="bk_secret", mask=MaskingHelper.mask_api_keys)

# Option 3: Field-based masking
client = Brokle(
    api_key="bk_secret",
    mask=MaskingHelper.field_mask(['password', 'ssn', 'api_key'])
)

# Option 4: Combine multiple strategies
combined_mask = MaskingHelper.combine_masks(
    MaskingHelper.mask_emails,
    MaskingHelper.mask_phones,
    MaskingHelper.field_mask(['password', 'secret_token'])
)
client = Brokle(api_key="bk_secret", mask=combined_mask)
```

### What Gets Masked

Masking applies to these span attributes:
- `input.value` - Generic input data
- `output.value` - Generic output data
- `gen_ai.input.messages` - LLM chat messages
- `gen_ai.output.messages` - LLM response messages
- `metadata` - Custom metadata

**Structural attributes are NOT masked** (model names, token counts, metrics, timestamps, environment tags).

### Error Handling

If your masking function throws an exception, Brokle returns:
```
"<fully masked due to failed mask function>"
```

This ensures sensitive data is **never transmitted** even if masking fails (security-first design).

### Custom Pattern Masking

Create custom masking for your specific needs:

```python
from brokle.utils.masking import MaskingHelper

# Mask IPv4 addresses
mask_ip = MaskingHelper.custom_pattern_mask(
    r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
    '[IP_ADDRESS]'
)

client = Brokle(api_key="bk_secret", mask=mask_ip)
```

### Security Best Practices

1. **Client-side masking**: Data is masked before leaving your application
2. **Test your masks**: Verify patterns catch your specific PII in development
3. **Fail-safe defaults**: Exceptions result in full masking (never sends unmasked data)
4. **Performance**: Masking adds <1ms overhead per span

For more examples, see [`examples/masking_basic.py`](examples/masking_basic.py) and [`examples/masking_helpers.py`](examples/masking_helpers.py).

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

## 🛡️ Graceful Degradation

Tracer errors never break your application. If tracing fails, your function still executes normally:

```python
from brokle import observe

@observe()
def my_critical_function(x):
    return x * 2

# Works even if BROKLE_API_KEY is missing or server is down
result = my_critical_function(5)  # Returns 10, logs warning
```

## 📦 Migration from Other SDKs

### From Langfuse

```python
# Langfuse
from langfuse.decorators import observe
@observe()
def my_function(): pass

# Brokle (same pattern!)
from brokle import observe
@observe()
def my_function(): pass
```

### From Braintrust

```python
# Braintrust
from braintrust import Eval
Eval("my-project", data=dataset, task=fn, scores=[score])

# Brokle
from brokle import evaluate
evaluate(task=fn, data=dataset, evaluators=[scorer], experiment_name="my-project")
```

### From LangSmith

```python
# LangSmith
from langsmith import traceable
@traceable
def my_function(): pass

# Brokle
from brokle import observe
@observe()
def my_function(): pass
```

## 🔧 Enhanced Error Messages

Errors include actionable guidance:

```python
from brokle import AuthenticationError, ConnectionError, ValidationError

# Errors include hints to help you fix issues
try:
    client = get_client()
except AuthenticationError as e:
    print(e.hint)  # Shows how to fix authentication issues
except ConnectionError as e:
    print(e.hint)  # Shows how to fix connection issues
except ValidationError as e:
    print(e.hint)  # Shows how to fix validation issues
```

Available error classes (following Langfuse naming pattern):
- `BrokleError` - Base error class
- `AuthenticationError` - API key invalid/missing
- `ConnectionError` - Server unreachable
- `ValidationError` - Invalid request data
- `RateLimitError` - Too many requests
- `NotFoundError` - Resource not found
- `ServerError` - Server-side error

## Support

- **Issues**: [GitHub Issues](https://github.com/brokle-ai/brokle-python/issues)
- **Docs**: [docs.brokle.com](https://docs.brokle.com/sdk/python)
- **Email**: support@brokle.com

---

**Simple. Powerful. OpenTelemetry-native observability for AI.**
