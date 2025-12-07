# Brokle Platform Python SDK Makefile

.PHONY: help install install-dev test test-verbose test-coverage lint format type-check clean build docs serve-docs

# Default target
help:
	@echo "Brokle Platform Python SDK"
	@echo ""
	@echo "Available commands:"
	@echo "  install       Install the package in development mode"
	@echo "  install-dev   Install development dependencies"
	@echo "  test          Run all tests"
	@echo "  test-verbose  Run tests with verbose output"
	@echo "  test-coverage Run tests with coverage report"
	@echo "  lint          Run linter (flake8)"
	@echo "  format        Format code with black and isort"
	@echo "  type-check    Run type checking with mypy"
	@echo "  clean         Clean build artifacts"
	@echo "  build         Build distribution packages"
	@echo "  docs          Generate documentation"
	@echo "  serve-docs    Serve documentation locally"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Testing
test:
	python -m pytest tests/ -v

test-verbose:
	python -m pytest tests/ -v -s

test-coverage:
	python -m pytest tests/ --cov=brokle --cov-report=html --cov-report=term-missing

test-specific:
	python -m pytest tests/$(TEST) -v

# Code quality
lint:
	flake8 brokle/ tests/

format:
	black brokle/ tests/ examples/
	isort brokle/ tests/ examples/

type-check:
	mypy brokle/

# Build and publish
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

# Documentation
docs:
	@echo "Documentation generation not yet implemented"

serve-docs:
	@echo "Documentation serving not yet implemented"

# Development workflow
dev-setup: install-dev
	pre-commit install

dev-test: format lint type-check test

dev-check: lint type-check test-coverage

# Quick checks
quick-test:
	python -m pytest tests/test_config.py -v

integration-test:
	python -m pytest tests/test_client.py tests/test_openai_client.py -v

# Security
security-check:
	bandit -r brokle/

# Dependencies
update-deps:
	pip-compile requirements.in
	pip-compile requirements-dev.in

# Environment
create-env:
	python -m venv venv
	@echo "Activate with: source venv/bin/activate"

# CI/CD helpers
ci-install:
	pip install -e ".[dev]"

ci-test:
	python -m pytest tests/ --cov=brokle --cov-report=xml

ci-check: lint type-check ci-test

# Release automation
release-patch:
	python scripts/release.py patch

release-minor:
	python scripts/release.py minor

release-major:
	python scripts/release.py major

release-patch-skip-tests:
	python scripts/release.py patch --skip-tests