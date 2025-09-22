# Integration Patterns Quick Reference

## 🎯 Three Patterns

### Pattern 1: Drop-in Replacement (Zero Code Changes)
```python
# Instead of: from openai import OpenAI
from brokle.openai import OpenAI  # Instant observability

client = OpenAI()
response = client.chat.completions.create(...)  # Automatically tracked
```

### Pattern 2: Universal Decorator (Framework-Agnostic)
```python
from brokle import observe

@observe()
def my_ai_workflow(prompt: str):
    # Any AI call here gets tracked
    return some_llm_call(prompt)
```

### Pattern 3: Native SDK (Full Features)
```python
from brokle import Brokle

async with Brokle() as client:
    response = await client.chat.create(
        model="gpt-4",
        routing_strategy="cost_optimized",  # Smart routing
        cache_strategy="semantic"           # 30-50% cost reduction
    )
```

## 🔧 Configuration

### Environment Variables
```bash
export BROKLE_API_KEY="ak_your_api_key"
export BROKLE_PROJECT_ID="proj_your_project"
export BROKLE_ENVIRONMENT="production"
```

### Programmatic Setup
```python
from brokle import Brokle, get_client

# Dedicated client
client = Brokle(api_key="ak_...", project_id="proj_...")

# Shared singleton (uses env vars)
client = get_client()
```

## 🚀 Pattern Benefits

### Pattern 1: Drop-in Replacement
- ✅ Zero code changes beyond import
- ✅ Works with existing OpenAI/Anthropic code
- ✅ Instant observability
- ✅ Perfect for migration

### Pattern 2: Universal Decorator
- ✅ Framework-agnostic (@observe any function)
- ✅ Works with any AI library
- ✅ Minimal code changes
- ✅ Complete workflow tracing

### Pattern 3: Native SDK
- ✅ Intelligent routing (250+ providers)
- ✅ Semantic caching (30-50% cost reduction)
- ✅ Cost optimization
- ✅ Full platform features

## 🔍 Common Issues

### Pattern 1 (Drop-in): Import Order
```python
# ✅ Correct
from brokle.openai import OpenAI

# ❌ Wrong - use original import
from openai import OpenAI
```

### API Key Issues
```python
# Check configuration
from brokle import get_client
print(get_client().config.api_key)
```

### Pattern 3 (Native): Async Usage
```python
# ✅ Always use async context manager
async with Brokle() as client:
    response = await client.chat.create(...)
```

## 💡 Quick Tips

- **Start with Pattern 1**: Easy migration, zero code changes
- **Upgrade to Pattern 2**: Add @observe for custom functions
- **Scale with Pattern 3**: Full platform features when needed
- **Performance**: <3ms overhead, 30-50% cost reduction
- **Compatibility**: Works with any existing AI workflow

---

*Simple. Powerful. Three patterns to fit your needs.*
