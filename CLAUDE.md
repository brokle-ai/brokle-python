# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Brokle Platform Python SDK** - a comprehensive Python SDK that provides intelligent routing, cost optimization, semantic caching, and observability for AI applications. It's designed as part of the larger Brokle platform ecosystem.

**Key Features:**
- OpenAI drop-in replacement with advanced routing and caching
- Three integration patterns: drop-in replacement, decorator, and native SDK
- Comprehensive observability with OpenTelemetry integration
- Cost optimization (30-50% reduction in LLM costs)
- Semantic caching with vector similarity
- Real-time analytics and evaluation framework

## Development Commands

### Essential Setup
```bash
# Install for development
make install-dev

# Full development setup (includes pre-commit hooks)
make dev-setup
```

### Testing
```bash
# Run all tests
make test

# Run tests with verbose output
make test-verbose

# Run tests with coverage report
make test-coverage

# Run specific test file
make test-specific TEST=test_config.py

# Quick config tests only
make quick-test

# Integration tests (client functionality)
make integration-test
```

### Code Quality
```bash
# Format code (black + isort)
make format

# Run linter (flake8)
make lint

# Type checking (mypy)
make type-check

# Full development check (lint + type-check + coverage)
make dev-check
```

### Build and Publish
```bash
# Clean build artifacts
make clean

# Build distribution packages
make build

# Publish to test PyPI
make publish-test

# Publish to PyPI
make publish
```

## Architecture Overview

### Package Structure
The SDK is organized as a modular Python package with 53+ Python files:

```
brokle/
├── ai_platform/          # AI platform abstraction layer
├── _client/              # Core HTTP client implementation
├── evaluation/           # Response evaluation framework
├── integrations/         # Auto-instrumentation for various libraries
├── openai/              # OpenAI compatibility layer
├── _task_manager/       # Background task management
├── testing/             # Testing utilities
├── types/               # Type definitions and attributes
├── _utils/              # Internal utilities
├── auth.py              # Authentication management
├── client.py            # Main Brokle client
├── config.py            # Configuration management
├── decorators.py        # @observe decorator
├── exceptions.py        # Custom exception hierarchy
└── __init__.py          # Public API exports
```

### Three Integration Patterns

1. **OpenAI Drop-in Replacement** (`brokle.openai`):
   - Zero-code changes beyond import
   - Full compatibility with OpenAI SDK
   - Enhanced with Brokle-specific features

2. **Decorator Pattern** (`brokle.decorators`):
   - `@observe()` decorator for comprehensive observability
   - Automatic telemetry and tracing
   - Configurable capture options

3. **Native SDK** (`brokle.client`):
   - Full platform feature access
   - Async/await support throughout
   - Advanced routing, caching, and evaluation

### Core Components

**Client Layer** (`client.py`, `_client/`):
- Async HTTP client with connection pooling
- Authentication and request signing
- Environment tag support (Langfuse-style validation)

**Configuration** (`config.py`):
- Environment variable and programmatic configuration
- Validation with Pydantic models
- Support for multiple environments and projects

**Authentication** (`auth.py`):
- API key management and validation
- Project-scoped authentication
- Token refresh and error handling

**AI Platform** (`ai_platform/`):
- Intelligent routing across 250+ LLM providers
- Semantic caching with similarity matching
- Quality scoring and evaluation
- Cost optimization strategies

## Testing Strategy

### Test Files
- `test_config.py` - Configuration and environment validation
- `test_client.py` - Core client functionality
- `test_auth.py` - Authentication and authorization
- `test_openai_client.py` - OpenAI compatibility layer
- `test_exceptions.py` - Error handling and custom exceptions

### Environment Testing
The SDK includes comprehensive environment tag validation following Langfuse-style rules:
- Max 40 characters, lowercase only
- Cannot start with "brokle" prefix
- Default environment is "default"
- Headers: `X-Environment`, `X-Project-ID`, `X-API-Key`

### Manual Testing
Use the provided manual test scripts:
```bash
# Interactive testing
python test_manual.py --interactive

# Direct testing with API key
python test_manual.py --api-key "ak_your_key_here"

# Integration testing with backend
python test_integration.py --api-key "ak_your_key_here"
```

## Configuration and Environment

### Environment Variables
```bash
BROKLE_API_KEY="ak_your_api_key"
BROKLE_HOST="http://localhost:8080"
BROKLE_PROJECT_ID="proj_your_project_id"
BROKLE_ENVIRONMENT="production"          # Environment tag (lowercase, max 40 chars)
BROKLE_OTEL_ENABLED=true
BROKLE_TELEMETRY_ENABLED=true
BROKLE_CACHE_ENABLED=true
```

### Programmatic Configuration
```python
import brokle

brokle.configure(
    api_key="ak_your_key",
    host="http://localhost:8080",
    project_id="proj_your_project",
    environment="production"  # Validates Langfuse-style rules
)
```

## Key Development Patterns

### Error Handling
The SDK uses a comprehensive exception hierarchy:
- `BrokleError` - Base exception
- `AuthenticationError` - Auth failures
- `ValidationError` - Input validation
- `ProviderError` - LLM provider issues
- `QuotaExceededError` - Usage limits
- `RateLimitError` - Rate limiting

### Async/Await Support
All client operations support async/await:
```python
async with Brokle() as client:
    response = await client.chat.create(...)
```

### OpenTelemetry Integration
Comprehensive tracing with custom attributes:
```python
from brokle.types.attributes import BrokleOtelSpanAttributes
span.set_attribute(BrokleOtelSpanAttributes.ROUTING_STRATEGY, "cost_optimized")
```

## Dependencies and Build System

### Core Dependencies
- `httpx>=0.25.0` - HTTP client
- `pydantic>=2.0.0` - Data validation
- `opentelemetry-*` - Comprehensive observability
- `python-dotenv>=1.0.0` - Environment management
- `backoff>=2.2.1` - Retry logic

### Optional Dependencies
- `openai[>=1.0.0]` - OpenAI compatibility
- `anthropic>=0.5.0` - Anthropic integration
- `google-generativeai>=0.3.0` - Google AI integration

### Development Tools
- `pytest` with asyncio support and coverage
- `black` + `isort` for formatting
- `flake8` for linting
- `mypy` for type checking
- `pre-commit` for git hooks

## Performance Considerations

- **Sub-3ms Telemetry Overhead**: Minimal impact on response times
- **Connection Pooling**: Efficient HTTP connection management
- **Background Processing**: Non-blocking telemetry submission
- **Batch Operations**: Efficient bulk processing

## Backend Integration

This SDK is designed to work with the Brokle platform backend:
- Default backend: `http://localhost:8080`
- Authentication via `/api/v1/auth/validate`
- Environment tags in request headers
- Project-scoped API key validation

The SDK validates environment tags using Langfuse-style rules and sends appropriate headers to the backend for proper request routing and scoping.