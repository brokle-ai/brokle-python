# 🎯 Brokle 3-Pattern Architecture → Features Mapping

## Control Plane Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    BROKLE CONTROL PLANE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Pattern 1: Wrapper Functions                                  │
│  ┌──────────────────────────────────────────────────────┐      │
│  │ wrap_openai() / wrap_anthropic()                     │      │
│  │ - Telemetry ONLY (captures requests/responses)       │      │
│  │ - NO gateway routing (calls OpenAI directly)         │      │
│  │ - NO structured traces                               │      │
│  │ Use case: "I just want to see what my app is doing"  │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                 │
│  Pattern 2: Universal Decorator                                │
│  ┌──────────────────────────────────────────────────────┐      │
│  │ @observe()                                            │      │
│  │ - Telemetry + Span management                        │      │
│  │ - Creates traces/observations                        │      │
│  │ - NO gateway routing                                 │      │
│  │ Use case: "I want to track my multi-step pipeline"   │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                 │
│  Pattern 3: Native SDK                                         │
│  ┌──────────────────────────────────────────────────────┐      │
│  │ brokle.chat.completions.create()                     │      │
│  │ - Gateway routing (calls Brokle backend)             │      │
│  │ - Auto-telemetry, caching, cost optimization         │      │
│  │ - CAN ALSO do structured traces                      │      │
│  │ Use case: "I want the complete control plane"        │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✅ Pattern 1: Wrapper Functions (Observability ONLY)

```python
from brokle import wrap_openai
import openai

client = wrap_openai(openai.OpenAI(api_key="sk_..."))
response = client.chat.completions.create(model="gpt-4", messages=[...])
```

- ✅ Request/response captured  
- ✅ Sent to Brokle telemetry backend  
- ❌ NO routing through Brokle  
- ❌ NO caching / cost optimization  

---

## ✅ Pattern 2: Universal Decorator (Structured Observability)

```python
from brokle import Brokle, observe

brokle = Brokle(api_key="bk_proj_...")

@observe(name="rag-pipeline")
def customer_support_rag(question: str):
    with brokle.observation(name="embed", type="embedding"):
        embedding = embed_text(question)

    with brokle.observation(name="search", type="retrieval"):
        docs = pinecone.query(embedding)

    with brokle.observation(name="llm", type="llm"):
        import openai
        response = openai.chat.completions.create(...)

    return response
```

- ✅ Creates trace hierarchy  
- ✅ Tracks each step separately  
- ✅ Add quality scores  
- ❌ NO routing / caching / cost optimization  

---

## ✅ Pattern 3: Native SDK (Full Control Plane)

### Option A: URL Drop-in (Gateway Only, No SDK Needed)

```python
import openai
openai.api_base = "https://brokle.dev/v1"
openai.api_key = "bk_proj_..."

response = openai.chat.completions.create(model="gpt-4", messages=[...])
```

### Option B: Brokle Native SDK

```python
from brokle import Brokle

brokle = Brokle(api_key="bk_proj_...")
response = brokle.chat.completions.create(model="gpt-4", messages=[...])

trace = brokle.trace(name="workflow")
with trace.observation(name="llm", type="llm"):
    response = brokle.chat.completions.create(...)
trace.score(name="quality", value=0.9)
```

- ✅ Gateway routing, caching, optimization  
- ✅ Structured observability (optional)  

---

## 🎯 Feature Comparison

| Feature           | Pattern 1 | Pattern 2 | Pattern 3    |
|-------------------|-----------|-----------|--------------|
| Gateway Routing   | ❌         | ❌         | ✅            |
| Caching           | ❌         | ❌         | ✅            |
| Cost Optimization | ❌         | ❌         | ✅            |
| Basic Telemetry   | ✅         | ✅         | ✅            |
| Structured Traces | ❌         | ✅         | ✅ (optional) |
| Quality Scores    | ❌         | ✅         | ✅ (optional) |

---

## 📦 SDK Structure

```python
# brokle/__init__.py
from .client import Brokle
from .wrappers import wrap_openai, wrap_anthropic
from .decorators import observe

__all__ = [
    "Brokle",           # Native SDK (Pattern 3)
    "wrap_openai",      # Wrapper (Pattern 1)
    "wrap_anthropic",   # Wrapper (Pattern 1)
    "observe",          # Decorator (Pattern 2)
]
```

---

## ✅ Final Summary

- Brokle SDK should include ALL 3 patterns:
  1. `wrap_openai()` → Observability only  
  2. `@observe()` → Structured traces  
  3. `Brokle` → Gateway + observability  

- URL Drop-in is NOT part of SDK, it’s a **backend feature**.  
- Native SDK = full developer experience + gateway + observability.  

---
