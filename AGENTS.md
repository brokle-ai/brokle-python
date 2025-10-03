# Repository Guidelines

## Project Structure & Module Organization
Core SDK code lives in `brokle/`, with the public client in `client.py`, shared helpers in `_client/`, and feature modules under `ai_platform/`, `integrations/`, and `_utils/`. Reusable fixtures and test scaffolding sit in `brokle/testing/`. Pytest suites reside in `tests/`, while `test_integration.py` and `test_manual.py` cover backend and manual verification. Contributor docs are stored in `docs/`, runnable examples in `examples/`, and automation scripts in `scripts/`. Prefer extending the root `Makefile` when adding new workflows.

## Build, Test, and Development Commands
Run `make install-dev` once to set up an editable environment with tooling. Use `make format`, `make lint`, and `make type-check` to apply Black/isort, Flake8, and mypy checks; `make dev-check` chains them with tests. Execute `make test` for the default pytest suite or `make test-specific TEST=tests/test_feature.py` to target a single module. Generate coverage with `make test-coverage`, integration checks with `make integration-test`, and build artifacts via `make build`.

## Coding Style & Naming Conventions
Target Python 3.8+ with 4-space indentation and Black's 88-character line length. Modules and functions follow `snake_case`, classes use `PascalCase`, and constants stay `UPPER_SNAKE`. Public APIs must be fully typed because `py.typed` ships downstream. Run `make format` before committing to keep imports ordered and formatting consistent.

## Testing Guidelines
Pytest is the canonical runner; name modules `test_<feature>.py`. Store reusable fixtures or factories under `brokle/testing/`. Exercise success and error paths, including both sync and async flows when present. For manual validation, invoke `python test_manual.py --interactive --api-key <key> --backend <url>` against a live backend.

## Commit & Pull Request Guidelines
Commit messages follow `<type>: <description>` (example: `fix: guard environment validation error`). Keep each commit focused, reference tickets when useful, and ensure docs or examples reflect new behavior. Before opening a PR, run `make dev-check` and any relevant integration commands, then include verification steps, linked issues, and screenshots or logs when they aid reviewers.

## Security & Configuration Tips
Run `make security-check` (Bandit) whenever touching auth or transport logic. Keep credentials in local `.env` files or environment variables; never commit secrets. Document any accepted suppressions and note backend requirements when sharing manual or integration instructions.
