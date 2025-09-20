# Repository Guidelines

## Project Structure & Module Organization
The SDK lives in `brokle/`, with `client.py`, `_client/`, and feature-specific folders (`ai_platform/`, `integrations/`, `_utils/`). Fixtures and test helpers reside in `brokle/testing/`. Pytest suites are under `tests/`, while `test_integration.py` and `test_manual.py` cover backend and manual runs. Contributor docs sit in `docs/`, examples in `examples/`, and automation scripts in `scripts/`. Prefer extending the `Makefile` when adding new workflows.

## Build, Test, and Development Commands
`make install-dev` sets up an editable environment with tooling. Use `make test` for the default pytest run and `make test-coverage` when you need coverage reports. `make format`, `make lint`, and `make type-check` invoke black+isort, flake8, and mypy; `make dev-check` chains quality checks with tests. `make integration-test` exercises client/back-end flows, and `python test_manual.py --interactive` walks through manual validation. Build artifacts with `make build` and only publish via `make publish-test` or `make publish` after maintainer approval.

## Coding Style & Naming Conventions
Code targets Python 3.8+, 4-space indentation, and Black’s 88-character limit. Modules and functions use `snake_case`, classes use `PascalCase`, and constants stay `UPPER_SNAKE`. Public APIs must be typed—`py.typed` ships downstream—so add annotations alongside new signatures. Keep docstrings concise, Google style, and focused on intent and error scenarios. Run `make format` before linting to keep import order and style consistent.

## Testing Guidelines
Pytest is the canonical runner; name new files `test_<feature>.py` and add fixtures under `brokle/testing/` when reuse is helpful. `make test-specific TEST=tests/test_feature.py` lets you target a single module. Manual and integration scripts expect a live backend (see `TESTING.md`); provide `--api-key` and `--backend` flags for authenticated checks. Aim for balanced coverage: exercise success, failure, and both sync/async code paths when applicable.

## Commit & Pull Request Guidelines
Commits follow `<type>: <description>` (for example `fix: guard environment validation error`). Keep each commit focused, and reference related tickets in the body when context helps. Pull requests should explain the change, list verification steps (`make dev-check`, integration output, screenshots), and call out documentation or config updates. Request review only after CI is green and examples or docs reflect the new behavior.

## Security & Operational Checks
Run `make security-check` (Bandit) whenever touching auth or transport code, and document accepted suppressions. Keep credentials out of the repo; rely on local `.env` files and environment variables instead.
