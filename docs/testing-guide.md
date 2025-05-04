# Testing Guide

This document provides an overview of the test suite for the OpenAI Responses Server, including how to run tests, what tests are included, and best practices for contributing new tests.

## Running Tests

The project includes a utility script `run_tests.sh` that configures and runs the test suite. This script:

1. Creates a virtual environment using `uv`
2. Installs the package and test dependencies
3. Runs all tests using pytest

To run all tests:

```bash
./run_tests.sh
```

To run specific tests:

```bash
# Activate the virtual environment first
source .venv/bin/activate

# Run a specific test file
python -m pytest tests/test_cli.py -v

# Run a specific test class
python -m pytest tests/test_cli.py::TestCLI -v

# Run a specific test method
python -m pytest tests/test_cli.py::TestCLI::test_start_server_imports -v
```

## Test Suite Structure

The test suite is organized into several files:

- `tests/test_cli.py`: Tests for CLI functionality
- `tests/test_server.py`: Tests for server API endpoints
- `tests/test_e2e.py`: End-to-end integration tests

### Fixtures and Utilities

Test fixtures are defined in `tests/conftest.py` and include:

- `python_executable`: Detects Python version and executable path
- `ensure_uv`: Verifies that the uv package manager is installed
- `temp_env_file`: Creates a temporary .env file for testing
- `server_process`: Starts the server as a separate process for testing
- `mock_httpx_client`: Mocks HTTP client responses for unit testing

## Writing Tests

When adding new functionality, please include corresponding tests. Follow these guidelines:

### Unit Tests

Unit tests should focus on testing individual functions or methods in isolation:

```python
def test_specific_function():
    # Arrange
    input_data = ...
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result == expected_result
```

### Mock External Dependencies

When testing code that depends on external systems, use mocks to isolate your tests:

```python
@patch('module.external_dependency')
def test_with_mock(mock_dependency):
    mock_dependency.return_value = expected_mock_value
    
    # Test with the mocked dependency
    result = function_using_dependency()
    
    assert result == expected_result
```

### Testing the CLI

CLI tests use `unittest.mock` to patch dependencies and avoid actual server startup:

```python
with patch('sys.argv', ['command_name', 'subcommand']):
    with patch('module.function_to_mock') as mock_function:
        main()  # Call the CLI entry point
        mock_function.assert_called_once()
```

### Testing Async Code

The project uses `pytest-asyncio` for testing asynchronous code. Mark your test functions with `@pytest.mark.asyncio`:

```python
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result == expected_result
```

## Troubleshooting Common Test Issues

### ImportError in Subprocess Fallback Test

If encountering an `ImportError` in `test_start_server_subprocess_fallback`, verify that the mocking is correctly set up. The test should mock imports by manipulating `sys.modules`:

```python
with patch.dict('sys.modules', {'uvicorn': None, 'openai_responses_server.server': None}):
    # The import will fail with ImportError
    with patch('module.subprocess') as mock_subprocess:
        # Test subprocess fallback
```

### Missing Dependencies

If tests fail with missing dependencies, ensure you've installed the test requirements:

```bash
uv pip install -e ".[dev]"
uv pip install pytest-asyncio httpx
```

## Test Coverage

To generate a test coverage report:

```bash
python -m pytest --cov=openai_responses_server tests/
```

For HTML coverage report:

```bash
python -m pytest --cov=openai_responses_server --cov-report=html tests/
```

The HTML report will be generated in the `htmlcov` directory. 