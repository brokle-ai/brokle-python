# Integration Patterns Guide - Brokle SDK

The Brokle Python SDK provides three elegant patterns for adding AI observability, routing, and optimization to your applications. Choose the pattern that best fits your use case and scale as needed.

## 🎯 Three Integration Patterns Overview

### Pattern 1: Wrapper Functions
**Best for**: Migrating existing applications with minimal code changes
- Wrap existing AI clients (OpenAI, Anthropic) for instant observability
- Zero code changes beyond import statements
- Maintains identical API interface
- Perfect for legacy code migration

### Pattern 2: Universal Decorator
**Best for**: Custom AI workflows and business logic
- Framework-agnostic `@observe()` decorator for any function
- Automatic hierarchical tracing (like Langfuse)
- Works with any AI library or custom logic
- Ideal for complex multi-step workflows

### Pattern 3: Native SDK
**Best for**: New applications wanting full platform features
- Complete AI platform: routing, caching, optimization
- OpenAI-compatible interface with Brokle extensions
- Advanced features: cost optimization, quality scoring
- Built for production scale and performance

---

# Pattern 1: Wrapper Functions

## Overview

Explicitly wrap existing AI client instances to add comprehensive observability and platform features while maintaining the exact same API interface.

### Key Benefits
- ✅ **Zero Code Changes**: Beyond import and wrapping, use clients identically
- ✅ **Perfect Migration**: Ideal for existing applications
- ✅ **Drop-in Replacement**: No learning curve for existing teams
- ✅ **Comprehensive Observability**: Full tracing, metrics, and analytics

## OpenAI Wrapper

### Basic Usage

```python
from openai import OpenAI
from brokle import wrap_openai

# Wrap existing OpenAI client
openai_client = wrap_openai(OpenAI(api_key="sk-..."))

# Use exactly like normal OpenAI client
response = openai_client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
# ✨ Enhanced with comprehensive AI-specific observability
```

### Advanced Configuration

```python
from openai import OpenAI, AsyncOpenAI
from brokle import wrap_openai

# Sync client with full configuration
openai_client = wrap_openai(
    OpenAI(api_key="sk-..."),
    capture_content=True,              # Capture request/response content
    capture_metadata=True,             # Capture model, tokens, etc.
    tags=["production", "chatbot"],    # Tags for analytics
    session_id="user_session_123",     # Group related calls
    user_id="user_456"                 # User-scoped analytics
)

# Async client support
async_client = wrap_openai(
    AsyncOpenAI(api_key="sk-..."),
    tags=["async", "streaming"],
    capture_content=True
)

# Use async client
async def chat_async():
    response = await async_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello async world!"}]
    )
    return response.choices[0].message.content
```

### Privacy Controls

```python
# Sensitive data handling
sensitive_client = wrap_openai(
    OpenAI(),
    capture_content=False,    # Don't capture request/response content
    capture_metadata=True,    # Still capture tokens, model, timing
    tags=["sensitive", "pii"]
)

# Use for sensitive operations
response = sensitive_client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Process this PII data: ..."}]
)
# ✅ Metadata captured, content privacy preserved
```

## Anthropic Wrapper

### Basic Usage

```python
from anthropic import Anthropic
from brokle import wrap_anthropic

# Wrap Anthropic client
anthropic_client = wrap_anthropic(
    Anthropic(api_key="sk-ant-..."),
    tags=["claude", "analysis"]
)

# Use exactly like normal Anthropic client
message = anthropic_client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    messages=[{"role": "user", "content": "Analyze this data..."}]
)

print(message.content[0].text)
# ✨ Enhanced with Brokle observability
```

### Advanced Usage

```python
from anthropic import Anthropic, AsyncAnthropic
from brokle import wrap_anthropic

# Production configuration
claude_client = wrap_anthropic(
    Anthropic(),
    capture_content=True,
    tags=["production", "document-analysis"],
    session_id="analysis_session_789",
    user_id="analyst_123"
)

# Async Anthropic client
async_claude = wrap_anthropic(
    AsyncAnthropic(),
    tags=["async", "streaming", "claude"],
    capture_metadata=True
)

# Document analysis workflow
def analyze_documents(documents: list[str]) -> list[str]:
    analyses = []

    for doc in documents:
        message = claude_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": f"Analyze this document and provide key insights: {doc}"
            }]
        )
        analyses.append(message.content[0].text)

    return analyses

# Each call automatically tracked with session context
results = analyze_documents(["doc1...", "doc2...", "doc3..."])
```

## Multi-Provider Workflows

```python
from openai import OpenAI
from anthropic import Anthropic
from brokle import wrap_openai, wrap_anthropic

# Wrap multiple providers with consistent configuration
openai_client = wrap_openai(
    OpenAI(),
    tags=["multi-provider", "openai"],
    session_id="comparison_session"
)

anthropic_client = wrap_anthropic(
    Anthropic(),
    tags=["multi-provider", "anthropic"],
    session_id="comparison_session"  # Same session for comparison
)

def compare_providers(prompt: str) -> dict:
    """Compare responses from multiple AI providers."""

    # OpenAI response
    openai_response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    # Anthropic response
    anthropic_response = anthropic_client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "openai": openai_response.choices[0].message.content,
        "anthropic": anthropic_response.content[0].text
    }

# Both providers tracked in same session for easy comparison
results = compare_providers("Explain quantum computing in simple terms")
# ✅ Complete observability across providers with session correlation
```

---

# Pattern 2: Universal Decorator

## Overview

The `@observe()` decorator provides framework-agnostic observability for any Python function with automatic hierarchical tracing. Works like Langfuse's `@observe()` but with enhanced AI intelligence.

### Key Benefits
- ✅ **Automatic Hierarchical Tracing**: Nested functions create proper span hierarchy
- ✅ **AI-Aware Intelligence**: Automatically detects AI provider usage
- ✅ **Framework Agnostic**: Works with any AI library or business logic
- ✅ **Zero Manual Management**: No context managers or manual span handling
- ✅ **Privacy Controls**: Configurable input/output capture with sensitive data protection

## Basic Hierarchical Tracing

### Simple Workflow Example

```python
from brokle import observe
import openai

# Initialize OpenAI client (will be auto-detected by @observe)
client = openai.OpenAI()

@observe(name="ai-content-pipeline")
def create_content(topic: str) -> dict:
    """Main content creation pipeline - creates parent span automatically."""

    # Step 1: Generate outline (child span)
    outline = create_outline(topic)

    # Step 2: Write content (child span)
    content = write_content(outline)

    # Step 3: Review content (child span)
    review = review_content(content)

    return {
        "topic": topic,
        "outline": outline,
        "content": content,
        "review": review,
        "status": "completed"
    }

@observe(name="outline-generation")
def create_outline(topic: str) -> str:
    """Generate outline - automatically becomes child span."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": f"Create an outline for content about: {topic}"
        }]
    )
    return response.choices[0].message.content

@observe(name="content-writing")
def write_content(outline: str) -> str:
    """Write content - automatically becomes child span."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": f"Write detailed content based on this outline: {outline}"
        }]
    )
    return response.choices[0].message.content

@observe(name="content-review")
def review_content(content: str) -> str:
    """Review content - automatically becomes child span."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": f"Review and provide feedback on this content: {content}"
        }]
    )
    return response.choices[0].message.content

# Usage - creates perfect span hierarchy automatically
result = create_content("Machine Learning Fundamentals")

# ✨ Automatic span hierarchy:
# └── ai-content-pipeline (parent)
#     ├── outline-generation (child)
#     ├── content-writing (child)
#     └── content-review (child)
```

### Complex Multi-Level Hierarchy

```python
from brokle import observe
import openai
from anthropic import Anthropic

openai_client = openai.OpenAI()
claude_client = Anthropic()

@observe(name="document-analysis-system")
def analyze_document_system(document: str) -> dict:
    """Complex document analysis with multi-level hierarchy."""

    # Phase 1: Multi-step preprocessing
    preprocessed = preprocess_document(document)

    # Phase 2: Multi-provider analysis
    analysis = multi_provider_analysis(preprocessed)

    # Phase 3: Generate final report
    report = generate_final_report(analysis)

    return {
        "original_length": len(document),
        "preprocessed": preprocessed,
        "analysis": analysis,
        "report": report
    }

@observe(name="document-preprocessing")
def preprocess_document(document: str) -> dict:
    """Multi-step preprocessing - child of document-analysis-system."""

    # Sub-step 1: Extract key information
    extracted = extract_key_info(document)

    # Sub-step 2: Clean and structure
    structured = structure_content(extracted)

    return {
        "extracted": extracted,
        "structured": structured
    }

@observe(name="key-extraction")
def extract_key_info(document: str) -> str:
    """Extract key info - grandchild span."""
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": f"Extract key information from: {document}"
        }]
    )
    return response.choices[0].message.content

@observe(name="content-structuring")
def structure_content(extracted_info: str) -> str:
    """Structure content - grandchild span."""
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": f"Structure this information: {extracted_info}"
        }]
    )
    return response.choices[0].message.content

@observe(name="multi-provider-analysis")
def multi_provider_analysis(preprocessed: dict) -> dict:
    """Multi-provider analysis - child of document-analysis-system."""

    # OpenAI analysis
    openai_analysis = openai_deep_analysis(preprocessed["structured"])

    # Claude analysis
    claude_analysis = claude_deep_analysis(preprocessed["structured"])

    # Compare results
    comparison = compare_analyses(openai_analysis, claude_analysis)

    return {
        "openai": openai_analysis,
        "claude": claude_analysis,
        "comparison": comparison
    }

@observe(name="openai-analysis")
def openai_deep_analysis(content: str) -> str:
    """OpenAI analysis - grandchild span."""
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": f"Provide deep analysis of: {content}"
        }]
    )
    return response.choices[0].message.content

@observe(name="claude-analysis")
def claude_deep_analysis(content: str) -> str:
    """Claude analysis - grandchild span."""
    message = claude_client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": f"Analyze this content deeply: {content}"
        }]
    )
    return message.content[0].text

@observe(name="analysis-comparison")
def compare_analyses(openai_result: str, claude_result: str) -> str:
    """Compare analyses - grandchild span."""
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": f"Compare these analyses: OpenAI: {openai_result} Claude: {claude_result}"
        }]
    )
    return response.choices[0].message.content

@observe(name="report-generation")
def generate_final_report(analysis: dict) -> str:
    """Generate final report - child of document-analysis-system."""
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": f"Create comprehensive report from: {analysis}"
        }]
    )
    return response.choices[0].message.content

# Creates perfect 3-level hierarchy automatically:
# └── document-analysis-system (root)
#     ├── document-preprocessing (child)
#     │   ├── key-extraction (grandchild)
#     │   └── content-structuring (grandchild)
#     ├── multi-provider-analysis (child)
#     │   ├── openai-analysis (grandchild)
#     │   ├── claude-analysis (grandchild)
#     │   └── analysis-comparison (grandchild)
#     └── report-generation (child)

result = analyze_document_system("Long technical document content...")
```

## Advanced Decorator Configuration

### Privacy and Sensitive Data

```python
@observe(
    name="secure-analysis",
    capture_inputs=False,     # Don't capture sensitive inputs
    capture_outputs=True,     # Capture outputs (if safe)
    tags=["secure", "pii"],
    user_id="user123"
)
def process_sensitive_data(api_key: str, personal_data: dict) -> dict:
    """Handle sensitive data with privacy controls."""
    # Function logic with sensitive parameters
    # Inputs won't be captured due to capture_inputs=False
    result = {"status": "processed", "items": len(personal_data)}
    return result

@observe(
    capture_inputs=True,
    capture_outputs=False,    # Don't capture potentially sensitive outputs
    max_input_length=1000     # Limit input capture length
)
def process_large_dataset(dataset: list) -> str:
    """Process large dataset with controlled capture."""
    # Large inputs truncated to 1000 chars
    # Outputs not captured for privacy
    return f"Processed {len(dataset)} items"
```

### Session and User Tracking

```python
@observe(
    name="user-workflow",
    session_id="session_abc123",
    user_id="user_xyz789",
    tags=["workflow", "production"],
    metadata={"version": "2.0", "feature": "analysis"}
)
def user_specific_workflow(user_input: str) -> dict:
    """Workflow with user and session context."""

    # All child functions automatically inherit session/user context
    step1 = process_user_input(user_input)
    step2 = analyze_with_ai(step1)
    step3 = generate_response(step2)

    return {"result": step3, "user_context": "preserved"}

@observe(name="input-processing")
def process_user_input(data: str) -> str:
    """Child function inherits parent session/user context."""
    return f"processed: {data}"

@observe(name="ai-analysis")
def analyze_with_ai(data: str) -> str:
    """AI analysis with inherited context."""
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Analyze: {data}"}]
    )
    return response.choices[0].message.content

@observe(name="response-generation")
def generate_response(analysis: str) -> str:
    """Response generation with inherited context."""
    return f"Based on analysis: {analysis}, here's the response..."
```

### Async Support

```python
import asyncio
from brokle import observe

@observe(name="async-workflow")
async def async_ai_workflow(queries: list[str]) -> list[str]:
    """Async workflow with concurrent AI calls."""

    # Process queries concurrently
    tasks = [process_single_query(query) for query in queries]
    results = await asyncio.gather(*tasks)

    # Combine results
    final_result = await combine_results(results)

    return final_result

@observe(name="single-query-processing")
async def process_single_query(query: str) -> str:
    """Process single query - child span."""
    from openai import AsyncOpenAI

    async_client = AsyncOpenAI()
    response = await async_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": query}]
    )
    return response.choices[0].message.content

@observe(name="result-combination")
async def combine_results(results: list[str]) -> str:
    """Combine results - child span."""
    from openai import AsyncOpenAI

    async_client = AsyncOpenAI()
    combined_input = " ".join(results)

    response = await async_client.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": f"Combine and summarize these results: {combined_input}"
        }]
    )
    return response.choices[0].message.content

# Usage
async def main():
    queries = [
        "What is machine learning?",
        "Explain neural networks",
        "What are transformers?"
    ]
    result = await async_ai_workflow(queries)
    print(result)

# Run async workflow
asyncio.run(main())
```

## Specialized Decorators

### LLM-Specific Observability

```python
from brokle import observe_llm

@observe_llm(
    name="story-generation",
    model="gpt-4",
    tags=["creative", "content"],
    metadata={"genre": "sci-fi", "length": "short"}
)
def generate_story(prompt: str, style: str) -> str:
    """Generate story with LLM-specific observability."""
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "system",
            "content": f"You are a {style} writer."
        }, {
            "role": "user",
            "content": f"Write a story: {prompt}"
        }]
    )
    return response.choices[0].message.content

# Enhanced with LLM-specific metadata
story = generate_story("A robot learns to love", "science fiction")
```

### Retrieval/Search Operations

```python
from brokle import observe_retrieval

@observe_retrieval(
    name="document-search",
    index_name="company_docs",
    tags=["search", "retrieval"],
    metadata={"search_type": "semantic"}
)
def search_company_documents(query: str, top_k: int = 5) -> list[str]:
    """Search documents with retrieval-specific observability."""
    # Hypothetical vector search
    results = vector_db.search(
        query=query,
        index="company_docs",
        top_k=top_k
    )
    return [doc.content for doc in results]

# Enhanced with retrieval-specific metadata
docs = search_company_documents("machine learning best practices")
```

---

# Pattern 3: Native SDK

## Overview

The Native SDK provides full Brokle platform features including intelligent routing, semantic caching, cost optimization, and quality scoring - all through an OpenAI-compatible interface enhanced with Brokle extensions.

### Key Benefits
- ✅ **Intelligent Routing**: Automatic routing across 250+ LLM providers
- ✅ **30-50% Cost Reduction**: Through smart routing and semantic caching
- ✅ **Quality Optimization**: Built-in response quality scoring and improvement
- ✅ **OpenAI Compatible**: Familiar interface with powerful extensions
- ✅ **Production Ready**: Built for enterprise scale and reliability

## Basic Usage

### Synchronous Client

```python
from brokle import Brokle

# Context manager (recommended for proper cleanup)
with Brokle(
    api_key="bk_your_api_key",
    host="http://localhost:8080",
    environment="production"
) as client:

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": "Explain quantum computing"
        }],
        # Standard OpenAI parameters
        temperature=0.7,
        max_tokens=500,

        # Brokle extensions
        routing_strategy="cost_optimized",  # Smart routing
        cache_strategy="semantic",          # Semantic caching
        tags=["education", "quantum"]       # Analytics tags
    )

    print(f"Response: {response.choices[0].message.content}")
    # Industry standard pattern: Access platform metadata via response.brokle.*
    if response.brokle:
        print(f"Provider used: {response.brokle.provider}")
        print(f"Cost: ${response.brokle.cost_usd}")
        print(f"Cache hit: {response.brokle.cache_hit}")
        print(f"Quality score: {response.brokle.quality_score}")
```

### Asynchronous Client

```python
from brokle import AsyncBrokle
import asyncio

async def main():
    async with AsyncBrokle(
        api_key="bk_your_api_key",
    ) as client:

        # Concurrent requests for better performance
        tasks = [
            client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": f"Question {i}: What is AI?"}],
                routing_strategy="latency_optimized",  # Minimize response time
                cache_strategy="semantic"               # Enable caching
            )
            for i in range(5)
        ]

        responses = await asyncio.gather(*tasks)

        for i, response in enumerate(responses):
            metadata = response.brokle
            print(f"Response {i}: Provider={metadata.provider}, "
                  f"Latency={metadata.latency_ms}ms, "
                  f"Cache hit={metadata.cache_hit}")

asyncio.run(main())
```

## Advanced Routing Strategies

### Cost Optimization

```python
from brokle import Brokle

with Brokle() as client:
    # Minimize cost while maintaining quality
    response = client.chat.completions.create(
        model="gpt-4",  # Requested model (may be substituted)
        messages=[{"role": "user", "content": "Summarize this article..."}],
        routing_strategy="cost_optimized",      # Prioritize cost savings
        cache_strategy="semantic",              # Enable semantic caching
        tags=["summarization", "cost-sensitive"]
    )

    # Check what actually happened
    metadata = response.brokle
    print(f"Requested: gpt-4, Actual provider: {metadata.provider}")
    print(f"Cost savings: {((1.0 - metadata.cost_usd/expected_cost) * 100):.1f}%")
    print(f"Quality maintained: {metadata.quality_score >= 0.8}")
```

### Quality Optimization

```python
# Prioritize highest quality responses
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{
        "role": "user",
        "content": "Write a technical analysis of quantum algorithms"
    }],
    routing_strategy="quality_optimized",       # Best quality available
    cache_strategy="disabled",                  # Fresh responses
    tags=["technical", "high-quality"],
    temperature=0.3                             # More deterministic
)

print(f"Quality score: {response.brokle.quality_score}")
print(f"Provider: {response.brokle.provider}")
```

### Latency Optimization

```python
# Minimize response time for real-time applications
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Quick fact about Mars"}],
    routing_strategy="latency_optimized",       # Fastest response
    cache_strategy="exact",                     # Fast cache lookups
    max_tokens=100,                            # Shorter responses
    tags=["realtime", "quick-facts"]
)

print(f"Response time: {response.brokle.latency_ms}ms")
```

### Balanced Strategy

```python
# Balance cost, quality, and latency
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Explain machine learning"}],
    routing_strategy="balanced",                # Optimize all factors
    cache_strategy="semantic",                  # Smart caching
    tags=["education", "balanced"]
)

metadata = response.brokle
print(f"Balanced result - Cost: ${metadata.cost_usd}, "
      f"Latency: {metadata.latency_ms}ms, "
      f"Quality: {metadata.quality_score}")
```

## Semantic Caching

### Cache Strategies

```python
# Semantic similarity caching (recommended)
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What is machine learning?"}],
    cache_strategy="semantic",                  # Matches similar questions
    tags=["caching", "education"]
)

# Later request with similar meaning
similar_response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Can you explain ML to me?"}],
    cache_strategy="semantic"
)

# Check cache performance
if similar_response.brokle and similar_response.brokle.cache_hit:
    print("Cache hit! Instant response with semantic matching")
    print(f"Cache similarity: {similar_response.brokle.cache_similarity_score}")
```

### Exact Match Caching

```python
# Exact match caching for deterministic responses
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    cache_strategy="exact",                     # Exact question match only
    temperature=0,                              # Deterministic responses
    tags=["math", "deterministic"]
)

# Exact same question will hit cache
exact_match = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What is 2+2?"}],  # Exact match
    cache_strategy="exact",
    temperature=0
)

if exact_match.brokle:
    print(f"Exact cache hit: {exact_match.brokle.cache_hit}")
```

## Environment-Based Configuration

### Using Environment Variables

```bash
# Set environment variables
export BROKLE_API_KEY="bk_your_api_key"
export BROKLE_HOST="https://api.brokle.com"
export BROKLE_ENVIRONMENT="production"
export BROKLE_CACHE_ENABLED="true"
```

```python
from brokle import get_client

# Uses environment variables automatically
client = get_client()

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
# ✅ Configured from environment variables
```

### Multiple Environments

```python
from brokle import Brokle

# Development environment
dev_client = Brokle(
    environment="development",
    cache_strategy="disabled",      # Fresh responses in dev
    routing_strategy="latency_optimized"
)

# Staging environment
staging_client = Brokle(
    environment="staging",
    cache_strategy="semantic",
    routing_strategy="balanced"
)

# Production environment
prod_client = Brokle(
    environment="production",
    cache_strategy="semantic",      # Full caching in prod
    routing_strategy="cost_optimized"
)

# Different behavior per environment
for env_name, client in [("dev", dev_client), ("staging", staging_client), ("prod", prod_client)]:
    with client:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test message"}],
            tags=[env_name, "testing"]
        )
        print(f"{env_name}: {response.brokle.provider}")
```

## Production Patterns

### Error Handling and Retry Logic

```python
from brokle import Brokle, RateLimitError, NetworkError, ProviderError
import backoff
import logging

logging.basicConfig(level=logging.INFO)

@backoff.on_exception(
    backoff.expo,
    (NetworkError, RateLimitError),
    max_tries=3,
    max_time=60
)
async def robust_ai_call(prompt: str) -> str:
    """Production-ready AI call with retry logic."""
    try:
        async with AsyncBrokle() as client:
            response = await client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                routing_strategy="quality_optimized",
                cache_strategy="semantic",
                timeout=30,                     # Reasonable timeout
                tags=["production", "robust"]
            )

            # Log success metrics
            metadata = response.brokle
            logging.info(f"AI call successful: provider={metadata.provider}, "
                        f"latency={metadata.latency_ms}ms, "
                        f"cost=${metadata.cost_usd}")

            return response.choices[0].message.content

    except ProviderError as e:
        logging.error(f"Provider error: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise

# Usage with error handling
async def main():
    try:
        result = await robust_ai_call("Explain quantum computing")
        print(f"Result: {result}")
    except Exception as e:
        print(f"Final error: {e}")

asyncio.run(main())
```

### Batch Processing

```python
import asyncio
from brokle import AsyncBrokle

async def batch_process_documents(documents: list[str]) -> list[dict]:
    """Efficiently process multiple documents with Brokle."""

    async with AsyncBrokle() as client:
        # Create tasks for concurrent processing
        tasks = []
        for i, doc in enumerate(documents):
            task = process_single_document(client, doc, i)
            tasks.append(task)

        # Process with concurrency limit
        semaphore = asyncio.Semaphore(10)  # Max 10 concurrent requests

        async def limited_task(task):
            async with semaphore:
                return await task

        # Execute all tasks
        results = await asyncio.gather(
            *[limited_task(task) for task in tasks],
            return_exceptions=True
        )

        # Filter successful results
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Document {i} failed: {result}")
            else:
                successful_results.append(result)

        return successful_results

async def process_single_document(client: AsyncBrokle, document: str, doc_id: int) -> dict:
    """Process a single document with full observability."""

    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "system",
            "content": "Analyze the following document and provide key insights."
        }, {
            "role": "user",
            "content": document
        }],
        routing_strategy="cost_optimized",      # Batch processing optimization
        cache_strategy="semantic",              # Leverage caching
        tags=["batch", "document-analysis", f"doc-{doc_id}"]
    )

    return {
        "doc_id": doc_id,
        "analysis": response.choices[0].message.content,
        "metadata": {
            "provider": response.brokle.provider,
            "cost": response.brokle.cost_usd,
            "latency": response.brokle.latency_ms,
            "cache_hit": response.brokle.cache_hit
        }
    }

# Usage
async def main():
    documents = [f"Document {i} content..." for i in range(50)]
    results = await batch_process_documents(documents)

    # Analyze batch results
    total_cost = sum(r["metadata"]["cost"] for r in results)
    cache_hits = sum(1 for r in results if r["metadata"]["cache_hit"])
    avg_latency = sum(r["metadata"]["latency"] for r in results) / len(results)

    print(f"Processed {len(results)} documents")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Cache hit rate: {cache_hits/len(results)*100:.1f}%")
    print(f"Average latency: {avg_latency:.1f}ms")

asyncio.run(main())
```


---

# Best Practices

## Pattern Selection Guide

### Start with Pattern 1 (Wrappers)
- **Perfect for migration**: Minimal code changes
- **Existing teams**: Familiar OpenAI/Anthropic interfaces
- **Legacy applications**: Drop-in enhancement
- **Quick wins**: Instant observability and routing

### Add Pattern 2 (Decorator)
- **Custom workflows**: Business logic observability
- **Complex pipelines**: Multi-step AI workflows
- **Framework agnostic**: Works with any AI library
- **Hierarchical tracing**: Automatic parent-child relationships

### Scale with Pattern 3 (Native SDK)
- **New applications**: Built from ground up
- **Advanced features**: Full platform capabilities
- **Production scale**: Enterprise-grade performance
- **Cost optimization**: Maximum efficiency

## Configuration Best Practices

### Environment-Based Configuration
```python
# Use environment variables for different deployments
import os

# Development
if os.getenv("ENV") == "development":
    client = Brokle(
        environment="development",
        cache_strategy="disabled",  # Fresh responses
        routing_strategy="latency_optimized"
    )

# Production
elif os.getenv("ENV") == "production":
    client = Brokle(
        environment="production",
        cache_strategy="semantic",  # Full caching
        routing_strategy="cost_optimized"
    )
```

### Proper Resource Management
```python
# ✅ Always use context managers
async with AsyncBrokle() as client:
    response = await client.chat.completions.create(...)

# ✅ Or explicit cleanup
client = Brokle()
try:
    response = client.chat.completions.create(...)
finally:
    client.close()
```

### Observability Tagging Strategy
```python
# Consistent tagging for analytics
base_tags = ["production", "brokle-app"]

# Feature-specific tags
content_tags = base_tags + ["content", "generation"]
analysis_tags = base_tags + ["analysis", "document"]

client = wrap_openai(OpenAI(), tags=content_tags)
```

## Performance Optimization

### Caching Strategy
```python
# High cache hit operations
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Common question"}],
    cache_strategy="semantic",      # Semantic matching
    temperature=0.7,                # Consistent for caching
    tags=["cacheable", "faq"]
)

# Low cache hit operations
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Unique creative request"}],
    cache_strategy="disabled",      # Don't cache unique content
    temperature=1.0,                # High creativity
    tags=["unique", "creative"]
)
```

### Routing Optimization
```python
# Cost-sensitive operations
summary_response = client.chat.completions.create(
    model="gpt-4",  # May be routed to cheaper equivalent
    messages=[{"role": "user", "content": "Summarize this..."}],
    routing_strategy="cost_optimized",
    max_tokens=200  # Shorter responses cost less
)

# Quality-critical operations
analysis_response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Technical analysis..."}],
    routing_strategy="quality_optimized",
    temperature=0.3  # More deterministic
)
```

The Brokle SDK provides comprehensive AI observability and optimization through three flexible integration patterns. Choose the pattern that fits your needs and scale as your requirements grow.