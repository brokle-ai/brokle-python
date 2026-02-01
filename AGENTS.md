# Repository Guidelines

## Project Structure & Module Organization
- `brokle/` is the main SDK package. Core client/config lives in `brokle/_client.py`, `brokle/_base_client.py`, and `brokle/config.py`. Public patterns are in `brokle/wrappers/` (client wrappers), `brokle/decorators.py` (`@observe`), and `brokle/evaluate.py` (evaluation helpers).
- `tests/` holds pytest suites plus documentation (`tests/README.md`).
- `examples/` and `docs/` contain usage samples and guides.
- Build artifacts land in `dist/` and metadata in `brokle.egg-info/`.

## Build, Test, and Development Commands
Use the Makefile for repeatable workflows:
- `make install-dev` installs dev extras.
- `make test` runs the full pytest suite.
- `make test-coverage` runs tests with coverage reports.
- `make format` formats with Black + isort.
- `make lint` runs Flake8.
- `make type-check` runs mypy.
- `make dev-check` runs lint + type-check + coverage.
- `make build` builds distributions after cleaning.

## Coding Style & Naming Conventions
- Python with 4-space indentation; keep lines <= 88 characters (Black default).
- Formatting: Black + isort (`profile = black`).
- Linting: Flake8. Type checking: mypy with strict settings.
- Tests follow `test_*.py` files, `Test*` classes, and `test_*` functions.

## Testing Guidelines
- Framework: pytest (see `pytest.ini`). Markers: `unit`, `integration`, `slow`, `asyncio`.
- Naming: `test_<functionality>_<scenario>` where possible.
- Run a single file: `python -m pytest tests/test_streaming_wrappers.py -v`.

## Commit & Pull Request Guidelines
- Recent history uses conventional prefixes like `feat:`, `fix:`, `refactor(scope):`, `chore:`.
- Preferred format (from `CONTRIBUTING.md`): `<type>: <description>` with optional body.
- PRs should include a clear description, linked issues, testing notes, and highlight breaking changes. Add screenshots or examples when behavior changes.

## Configuration & Security Notes
- Local runs require `BROKLE_API_KEY`; optional `BROKLE_BASE_URL` defaults to `http://localhost:8080`.
- Mask sensitive data using the masking utilities in `brokle/utils/` before sending telemetry.
