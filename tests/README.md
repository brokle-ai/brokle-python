# Brokle Python SDK Test Suite

**Status**: ✅ All tests passing (73/73)
**Last Updated**: 2025-12-08

## Quick Start

```bash
# Run all tests
make test

# Run with verbose output
pytest tests/ -v

# Run with coverage
make test-coverage

# Run specific test file
pytest tests/test_streaming_wrappers.py -v
```

## Test Structure

```
tests/
├── conftest.py                      # Shared fixtures and configuration
├── test_input_output.py             # I/O capture and MIME type tests (9 tests)
├── test_integration.py              # Integration patterns and lifecycle (6 tests)
├── test_otel_resource.py            # OTEL resource management (16 tests)
├── test_serialization_edge_cases.py # Serialization edge cases (11 tests)
├── test_streaming_wrappers.py       # Streaming wrapper tests (31 tests)
├── CLEANUP_SUMMARY.md               # Cleanup history and decisions
├── IMPORT_MAPPING.md                # SDK API migration guide
└── MODULES_REMOVED.md               # List of removed modules
```

## Test Coverage

The test suite covers core SDK functionality:

- **Streaming** (31 tests) - Sync/async wrappers, error handling, resource cleanup
- **OTEL Integration** (16 tests) - Resource attributes, metrics forwarding, config
- **Serialization** (11 tests) - Data types, edge cases, circular references
- **I/O Capture** (9 tests) - Input/output capture, MIME types, nested spans
- **Integration** (6 tests) - Client patterns, lifecycle, configuration

## What's NOT Tested

These modules were removed during SDK refactoring:
- Old auth module (`brokle.auth`)
- Old exception hierarchy (`brokle.exceptions`)
- Old client resource API (`.chat`, `.embeddings`, `.models`)
- Old configuration helpers (`sanitize_environment_name`, etc.)
- Provider management, task manager, HTTP base layer

## Running Tests

### All Tests
```bash
pytest tests/
```

### Specific Test File
```bash
pytest tests/test_streaming_wrappers.py -v
```

### With Coverage Report
```bash
pytest tests/ --cov=brokle --cov-report=html
open htmlcov/index.html
```

### Watch Mode (requires pytest-watch)
```bash
ptw tests/
```

## Test Fixtures

See `conftest.py` for available fixtures:
- `test_config` - Test BrokleConfig instance
- `mock_http_client` - Mocked HTTP client
- `mock_telemetry_manager` - Mocked telemetry manager
- Auto-cleanup fixtures for client state and telemetry

## Contributing

When adding new tests:
1. Place in appropriate test file based on functionality
2. Use existing fixtures from `conftest.py`
3. Follow naming convention: `test_<functionality>_<scenario>`
4. Add docstring explaining what the test validates
5. Run `make test` to verify all tests still pass

## Cleanup History

This test suite was cleaned up on 2025-12-08:
- Removed 15 obsolete test files (~4,500 lines)
- Fixed 16 import errors from SDK refactoring
- Achieved 100% pass rate (73/73 tests)

See `CLEANUP_SUMMARY.md` for full details.
