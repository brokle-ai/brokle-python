## Brokle OpenTelemetry SDK

**OpenTelemetry-native observability for AI applications.**

This SDK leverages the industry-standard OpenTelemetry framework to provide comprehensive observability for LLM applications, following OpenTelemetry GenAI 1.28+ semantic conventions.

### ✨ Key Features

- **🎯 OTEL-Native**: Built on OpenTelemetry SDK, not custom implementations
- **🚀 Zero Configuration**: Automatic instrumentation with SDK wrappers
- **📊 GenAI 1.28+ Compliant**: Follows OpenTelemetry GenAI semantic conventions
- **🔧 Multiple Integration Patterns**: Decorators, wrappers, or manual instrumentation
- **🌐 Multi-Provider Support**: OpenAI, Anthropic, and more
- **🔒 Privacy-Safe**: Built-in masking functions for sensitive data
- **📦 50% Less Code**: Eliminated custom batching/queuing logic

### 🚀 Quick Start

#### Installation

```bash
pip install brokle
```

#### Basic Usage

```python
from brokle import Brokle

# Initialize client
client = Brokle(api_key="bk_your_secret")

# Create a span
with client.start_as_current_span("my-operation") as span:
    span.set_attribute("output", "Hello, world!")

# Flush (important for short-lived apps)
client.flush()
```

#### Environment Variables

```bash
# Required
export BROKLE_API_KEY="bk_your_secret"

# Optional
export BROKLE_BASE_URL="https://api.brokle.ai"
export BROKLE_ENVIRONMENT="production"
export BROKLE_DEBUG="true"
```

Then use the singleton:

```python
from brokle import get_client

client = get_client()  # Reads from environment variables
```

### 🎯 Integration Patterns

#### Pattern 1: Decorators (Simplest)

```python
from brokle.decorators import observe

@observe(user_id="user-123")
def process_request(input_text: str):
    return f"Processed: {input_text}"

result = process_request("hello")  # Automatically traced
```

#### Pattern 2: SDK Wrappers (Zero Config for LLMs)

```python
from brokle import get_client
from brokle.wrappers import wrap_openai
import openai

# Initialize Brokle
brokle = get_client()

# Wrap OpenAI client
client = wrap_openai(openai.OpenAI(api_key="..."))

# All calls automatically traced with GenAI attributes
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)

brokle.flush()
```

#### Pattern 3: Context Managers (Recommended for Complex Workflows)

```python
from brokle import get_client
from brokle.types import Attrs

client = get_client()

# Generic span
with client.start_as_current_span("process-request") as span:
    span.set_attribute(Attrs.USER_ID, "user-123")
    # Your logic here

# LLM generation (OTEL 1.28+ compliant)
with client.start_as_current_generation(
    name="chat",
    model="gpt-4",
    provider="openai",
    input_messages=[{"role": "user", "content": "Hello"}],
    model_parameters={"temperature": 0.7}
) as gen:
    # Make LLM call
    gen.set_attribute(Attrs.GEN_AI_OUTPUT_MESSAGES, '[...]')
    gen.set_attribute(Attrs.GEN_AI_USAGE_INPUT_TOKENS, 10)
    gen.set_attribute(Attrs.GEN_AI_USAGE_OUTPUT_TOKENS, 50)

client.flush()
```

### 📊 OpenTelemetry GenAI 1.28+ Compliance

This SDK follows OpenTelemetry GenAI semantic conventions:

#### Standard Attributes

```python
from brokle.types import Attrs

# OTEL GenAI standard attributes
Attrs.GEN_AI_PROVIDER_NAME          # "openai", "anthropic"
Attrs.GEN_AI_OPERATION_NAME         # "chat", "embeddings"
Attrs.GEN_AI_REQUEST_MODEL          # "gpt-4"
Attrs.GEN_AI_RESPONSE_MODEL         # "gpt-4-0613"
Attrs.GEN_AI_INPUT_MESSAGES         # JSON array
Attrs.GEN_AI_OUTPUT_MESSAGES        # JSON array
Attrs.GEN_AI_USAGE_INPUT_TOKENS     # Token counts
Attrs.GEN_AI_USAGE_OUTPUT_TOKENS

# Provider-specific attributes
Attrs.OPENAI_REQUEST_N
Attrs.OPENAI_RESPONSE_SYSTEM_FINGERPRINT
Attrs.ANTHROPIC_REQUEST_TOP_K

# Brokle custom attributes
Attrs.BROKLE_USAGE_TOTAL_TOKENS     # Convenience metric
Attrs.BROKLE_USAGE_COST_USD         # Cost tracking
```

### 🔗 Framework Integrations

#### LangChain Integration

Automatic tracing for LangChain applications via callback handlers:

```python
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from brokle.integrations import BrokleLangChainCallback

# Create Brokle callback
callback = BrokleLangChainCallback(
    user_id="user-123",
    session_id="session-456"
)

# Use with LangChain
llm = ChatOpenAI(callbacks=[callback])
chain = LLMChain(llm=llm, callbacks=[callback])
result = chain.run("Hello")  # Automatically traced

# Flush telemetry
from brokle import get_client
get_client().flush()
```

**Installation**:
```bash
pip install brokle[langchain]
```

**Features**:
- ✅ Automatic parent-child span hierarchy
- ✅ LLM call tracing with GenAI attributes
- ✅ Chain execution tracing
- ✅ Tool execution tracing
- ✅ Error handling

---

#### LlamaIndex Integration

Automatic tracing via global handler registration:

```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from brokle.integrations import set_global_handler

# Set Brokle as global handler
set_global_handler("brokle", user_id="user-123")

# Use LlamaIndex normally
documents = SimpleDirectoryReader("data/").load_data()
index = VectorStoreIndex.from_documents(documents)
response = index.as_query_engine().query("Question")  # Automatically traced

# Flush telemetry
from brokle import get_client
get_client().flush()
```

**Installation**:
```bash
pip install brokle[llamaindex]
```

**Features**:
- ✅ Query execution tracing
- ✅ Retrieval tracing with scores
- ✅ LLM generation tracing
- ✅ Document processing tracing
- ✅ Embedding generation tracing

---

### 🔧 Configuration

All configuration can be set programmatically or via environment variables:

| Parameter | Type | Default | Env Var | Description |
|-----------|------|---------|---------|-------------|
| `api_key` | `str` | - | `BROKLE_API_KEY` | **Required**. Brokle API key |
| `base_url` | `str` | `"http://localhost:8080"` | `BROKLE_BASE_URL` | API base URL |
| `environment` | `str` | `"default"` | `BROKLE_ENVIRONMENT` | Environment tag |
| `tracing_enabled` | `bool` | `True` | `BROKLE_TRACING_ENABLED` | Enable/disable tracing |
| `sample_rate` | `float` | `1.0` | `BROKLE_SAMPLE_RATE` | Trace-level sampling rate (0.0-1.0) |
| `flush_at` | `int` | `100` | `BROKLE_FLUSH_AT` | Batch size |
| `flush_interval` | `float` | `5.0` | `BROKLE_FLUSH_INTERVAL` | Flush interval (seconds) |

### 📊 Trace-Level Sampling

**Sample 10% of traces** (cost optimization):
```python
client = Brokle(
    api_key="bk_your_secret",
    sample_rate=0.1  # Sample 10% of traces (entire traces, not individual spans)
)
```

**How it works**:
- Uses OpenTelemetry's `TraceIdRatioBased` sampler
- Sampling decision based on trace_id hash (deterministic)
- All spans within a sampled trace are exported together
- All spans in non-sampled traces are dropped together
- No partial traces in backend

**Example**:
```python
# With sample_rate=0.5, approximately 50% of traces will be sampled
client = Brokle(sample_rate=0.5)

for i in range(100):
    with client.start_as_current_span(f"trace-{i}") as parent:
        with client.start_as_current_span(f"child-{i}-1"):
            pass
        with client.start_as_current_span(f"child-{i}-2"):
            pass

# Expected result: ~50 complete traces (parent + 2 children each)
# Not: Random mix of 150 individual spans from different traces
```

---

### 🔒 Privacy & Masking

```python
import re

def mask_pii(data):
    """Mask PII data before sending."""
    if isinstance(data, str):
        # Mask email addresses
        data = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', data)
    return data

client = Brokle(
    api_key="bk_your_secret",
    mask=mask_pii
)
```

### 🌟 Advanced Features

#### Nested Spans (Parent-Child Hierarchy)

```python
with client.start_as_current_span("parent-operation") as parent:
    # Child span 1
    with client.start_as_current_span("child-1") as child1:
        child1.set_attribute("index", 1)

    # Child span 2
    with client.start_as_current_span("child-2") as child2:
        child2.set_attribute("index", 2)
```

#### Error Handling

Errors are automatically captured and recorded:

```python
try:
    with client.start_as_current_span("risky-operation") as span:
        raise ValueError("Something went wrong")
except ValueError:
    pass  # Error automatically recorded in span
```

#### Mixed Provider Usage

```python
from brokle.wrappers import wrap_openai, wrap_anthropic

# Wrap multiple providers
openai_client = wrap_openai(openai.OpenAI(...))
anthropic_client = wrap_anthropic(anthropic.Anthropic(...))

# Both automatically traced with proper attributes
with client.start_as_current_span("comparison") as span:
    response1 = openai_client.chat.completions.create(...)
    response2 = anthropic_client.messages.create(...)
```

### 📖 Examples

See the `examples/` directory for comprehensive examples:

- `basic_usage.py` - Core SDK patterns (spans, generations, decorators)
- `wrapper_usage.py` - LLM wrapper examples (OpenAI, Anthropic)
- `langchain_integration.py` - LangChain automatic tracing
- `llamaindex_integration.py` - LlamaIndex automatic tracing

### 🏗️ Architecture

```
SDK → OpenTelemetry SDK → OTLP/HTTP → Brokle Backend → ClickHouse
     (Industry Standard)   (Protobuf)   (Processing)     (Analytics)
```

**Key Components:**

1. **Brokle Client**: Initializes OpenTelemetry TracerProvider
2. **OTLP Exporter**: Sends data to `/v1/traces` endpoint (OpenTelemetry standard)
3. **Span Processor**: Handles batching and sampling
4. **SDK Wrappers**: Automatic instrumentation for LLM SDKs

### 🆚 Comparison with Legacy SDK

| Feature | Legacy SDK | OTEL-Native SDK |
|---------|-----------|-----------------|
| **Transport** | Custom `/v1/ingest/batch` | Standard OTLP |
| **Batching** | Custom queue (~300 lines) | OpenTelemetry SDK |
| **Code Size** | 62+ files | ~17 files |
| **Maintenance** | High | Low |
| **Ecosystem** | Isolated | OTEL-compatible |
| **Format** | Custom JSON | OTLP Protobuf/JSON |

### 🤝 Contributing

This SDK is part of the [Brokle](https://github.com/brokle/brokle) open-source project.

### 📄 License

Apache 2.0 - See LICENSE file for details.

### 🔗 Resources

- [Documentation](https://docs.brokle.ai)
- [OpenTelemetry GenAI Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [GitHub Repository](https://github.com/brokle/brokle)
- [Discord Community](https://discord.gg/brokle)

### 🐛 Troubleshooting

**Q: Data not showing up in dashboard?**
- Ensure you call `client.flush()` for short-lived apps
- Check BROKLE_API_KEY is set correctly
- Verify BROKLE_BASE_URL points to your backend

**Q: ImportError for opentelemetry?**
```bash
pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
```

**Q: Spans not nested correctly?**
- Use `start_as_current_span()` context manager (sets current span in context)
- Avoid manual `start_span()` unless you need full control

---

**Built with ❤️ by the Brokle team**
