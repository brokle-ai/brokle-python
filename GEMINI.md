# Brokle Python SDK

This document provides context for the Brokle Python SDK, a comprehensive library for interacting with the Brokle Platform.

## 1. Project Overview

The Brokle Python SDK enables developers to integrate their Python applications with the Brokle Platform, a powerful AI infrastructure solution that provides:

*   **Intelligent Routing:** Optimizes for cost, latency, and quality when routing requests to different LLM providers.
*   **Cost Optimization:** Reduces LLM-related expenses.
*   **Semantic Caching:** Caches LLM responses to reduce redundant calls.
*   **Real-time Analytics:** Offers insights into application performance and usage.
*   **Response Evaluation:** Assesses the quality of LLM responses.
*   **OpenTelemetry Integration:** Provides distributed tracing for enhanced observability.

The SDK is designed to be easy to use and offers three integration patterns:

1.  **OpenAI Drop-in Replacement:** Allows for seamless integration with existing OpenAI codebases.
2.  **`@observe` Decorator:** Provides a simple way to add observability to functions.
3.  **Native SDK:** Offers full access to all Brokle Platform features.

## 2. Project Structure

The project is organized as follows:

*   `brokle/`: The main source code directory.
    *   `_client/`: Core client implementation, including OpenTelemetry integration.
    *   `_task_manager/`: Background task management for non-blocking telemetry.
    *   `_utils/`: Utility functions for caching, error handling, and more.
    *   `ai_platform/`: Modules for interacting with AI platform features like caching, routing, and optimization.
    *   `anthropic/`, `openai/`, `langchain/`: Integrations with third-party libraries.
    *   `evaluation/`: Functionality for evaluating LLM responses.
    *   `testing/`: Utilities for testing the SDK.
    *   `types/`: Pydantic models for data validation and serialization.
*   `tests/`: Unit and integration tests.
*   `examples/`: Example usage of the SDK.
*   `docs/`: Documentation for the SDK.
*   `scripts/`: Helper scripts for development and maintenance.

## 3. Building and Running

The project uses `make` to streamline development tasks. Here are the key commands:

*   **Install dependencies:**
    ```bash
    make install-dev
    ```
*   **Run tests:**
    ```bash
    make test
    ```
*   **Run tests with coverage:**
    ```bash
    make test-coverage
    ```
*   **Format code:**
    ```bash
    make format
    ```
*   **Lint code:**
    ```bash
    make lint
    ```
*   **Type-check code:**
    ```bash
    make type-check
    ```
*   **Build the project:**
    ```bash
    make build
    ```

## 4. Development Conventions

The project follows standard Python development conventions:

*   **Formatting:** `black` and `isort` are used for code formatting.
*   **Linting:** `flake8` is used for linting.
*   **Type Checking:** `mypy` is used for static type checking.
*   **Testing:** `pytest` is used for testing.
*   **Dependency Management:** `pip` and `setuptools` are used for dependency management.
*   **Continuous Integration:** GitHub Actions are used for CI/CD.

The project also has a `pre-commit` configuration to ensure that code quality checks are run before committing code.
