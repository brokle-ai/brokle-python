# Observability Guide - New 3-Pattern Architecture

The Brokle SDK provides three elegant patterns for adding observability to your AI applications, inspired by industry best practices.

## 🎯 Three Integration Patterns

### Pattern 1: Wrapper Functions

Explicit wrapping for enhanced AI observability:

```python
# Explicit wrapping approach
from openai import OpenAI
from anthropic import Anthropic
from brokle import wrap_openai, wrap_anthropic

# Wrap your AI clients for enhanced observability
openai_client = wrap_openai(
    OpenAI(api_key="sk-..."),
    tags=["production"],
    session_id="user_session_123"
)

anthropic_client = wrap_anthropic(
    Anthropic(api_key="sk-ant-..."),
    tags=["claude", "analysis"]
)

# Use wrapped clients exactly like normal
response = openai_client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
# ✨ Enhanced with comprehensive AI-specific observability
```

### Pattern 2: Universal Decorator (Framework-Agnostic)

Add observability to any function:

```python
from brokle import observe

@observe()
def my_ai_workflow(query: str) -> str:
    # Any AI code here gets automatic observability
    response = some_llm_call(query)
    analysis = another_ai_service(response)
    return analysis

result = my_ai_workflow("Analyze this data")
# ✨ Complete workflow traced automatically
```

### Pattern 3: Native Brokle SDK (Full Platform Features)

Get intelligent routing, caching, and optimization:

```python
from brokle import Brokle

client = Brokle(
    api_key="ak_your_brokle_key",
    project_id="proj_your_project"
)

# Intelligent routing across 250+ providers
response = await client.chat.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
# ✨ Full platform: routing + caching + optimization + observability
```

## 🔧 Configuration

### Environment Variables
```bash
BROKLE_API_KEY="ak_your_api_key"
BROKLE_PROJECT_ID="proj_your_project"
BROKLE_ENVIRONMENT="production"
BROKLE_TELEMETRY_ENABLED=true
```

### Programmatic Configuration
```python
from brokle.client import get_client

# Configure globally
client = get_client(
    api_key="ak_your_key",
    project_id="proj_your_project",
    environment="production"
)
```

## 📊 LangChain Integration

Use the callback handler for LangChain workflows:

```python
from brokle.langchain import BrokleCallbackHandler

handler = BrokleCallbackHandler(session_id="my-session")

# Use with any LangChain component
chain.run(input_text, callbacks=[handler])
```

## 🎯 Migration from Old System

If you were using the old auto-instrumentation system:

```python
# OLD - Remove this
import brokle.auto_instrumentation as brokle_ai
brokle_ai.auto_instrument()

# NEW - Use drop-in replacements
from brokle.openai import OpenAI
from brokle.anthropic import Anthropic
```

## 📈 Benefits

- **Zero Code Changes**: Drop-in replacements work with existing code
- **Performance**: <1ms overhead with comprehensive observability
- **Flexibility**: Choose the right pattern for your use case
- **Production Ready**: Enterprise-grade error handling and fallbacks
- **Industry Standard**: Modern observability patterns developers recognize

## 🔍 Debugging

```python
# Check if observability is working
from brokle.client import get_client

client = get_client()
print(f"Telemetry enabled: {client.config.telemetry_enabled}")
print(f"Project: {client.config.project_id}")
```

The new architecture provides the same comprehensive observability with a much cleaner, more maintainable approach.