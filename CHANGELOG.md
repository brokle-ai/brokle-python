# Changelog

All notable changes to the Brokle Platform Python SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2024-01-15

### Added
- Initial release of Brokle Platform Python SDK
- Three integration patterns:
  - OpenAI drop-in replacement
  - @observe decorator (LangFuse-style)
  - Native SDK with full platform features
- Core features:
  - Intelligent routing (cost, quality, latency optimization)
  - Semantic caching with vector similarity
  - Real-time analytics and monitoring
  - Response evaluation and feedback loops
  - OpenTelemetry integration
  - Background telemetry processing
  - Comprehensive error handling
- OpenAI-compatible API support:
  - Chat completions
  - Text completions
  - Embeddings
  - Streaming responses
  - Async support
- Advanced features:
  - Cost optimization with spending limits
  - Provider-specific routing
  - A/B testing framework
  - Custom tagging and metadata
  - Session and user tracking
- Full async/await support throughout
- Type safety with Pydantic models
- Comprehensive documentation and examples

### Dependencies
- httpx>=0.25.0
- pydantic>=2.0.0
- opentelemetry-api>=1.20.0
- opentelemetry-sdk>=1.20.0
- opentelemetry-instrumentation>=0.41b0
- opentelemetry-exporter-otlp>=1.20.0
- opentelemetry-semantic-conventions>=0.41b0
- typing-extensions>=4.0.0
- python-dotenv>=1.0.0
- backoff>=2.2.1

### Optional Dependencies
- openai>=1.0.0 (for OpenAI compatibility)
- wrapt>=1.15.0 (for OpenAI compatibility)