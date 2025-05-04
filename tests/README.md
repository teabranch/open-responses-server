# OpenAI Responses Server Test Plan

This directory contains the test suite for the OpenAI Responses Server project.

> **Note:** For a comprehensive guide to running and writing tests, see the [Testing Guide](../docs/testing-guide.md) in the docs directory.

## Test Structure

The tests are organized into the following categories:

1. **Unit Tests**
   - `test_cli.py`: Tests for the CLI functionality
   - `test_server.py`: Tests for the server functionality, including API endpoints and conversion logic

2. **Integration Tests**
   - Server startup and configuration tests in `test_server.py` (TestServerIntegration class)
   - End-to-end tests in `test_e2e.py`

## Test Requirements

- Python 3.8+
- UV package manager (`pip install uv`)
- pytest and pytest-asyncio
- httpx

## Running Tests

You can run the tests using the provided `run_tests.sh` script in the project root:

```bash
./run_tests.sh
```

This script will:
1. Detect your Python version
2. Ensure UV is installed
3. Create a virtual environment using UV
4. Install the package and test dependencies
5. Run the tests

Alternatively, you can run tests manually:

```bash
# Install dependencies
uv pip install -e ".[dev]"
uv pip install pytest-asyncio httpx

# Run all tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_server.py -v

# Run a specific test
python -m pytest tests/test_server.py::TestServer::test_health_check -v
```

## Test Components

### CLI Tests

Tests the command-line interface functionality, including:
- Server startup
- Configuration management
- Command handling

### Server Tests

Tests the server functionality, including:
- API endpoints (/responses, /health, proxy endpoints)
- Request/response handling
- Format conversion between Responses API and chat.completions API

### Integration Tests

Tests the integration between components and the server's interaction with clients, including:
- Server startup and configuration
- End-to-end API flows
- Streaming response handling

## Test Fixtures

The test suite uses several fixtures to facilitate testing:

- `python_executable`: Detects the Python executable path
- `ensure_uv`: Ensures UV package manager is installed
- `temp_env_file`: Creates a temporary .env file for testing
- `server_process`: Starts the server as a separate process for testing
- `mock_httpx_client`: Mocks the httpx client for unit tests

## Adding New Tests

When adding new tests:

1. Follow the existing pattern in the appropriate test file
2. Use fixtures from `conftest.py` where appropriate
3. For API tests, create appropriate mock responses
4. For new components, consider creating dedicated test files

## Test Scope

The test suite covers:
- Server functionality (API endpoints, request/response handling)
- CLI operations
- Configuration management
- Error handling
- Streaming responses
- Protocol conversion between Responses API and chat.completions API

The tests use mock responses for external API calls to ensure tests run reliably without external dependencies. 