#!/bin/bash
# Script to run tests for open-responses-server using uv

set -e  # Exit on error

# Detect Python version and executable
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_EXECUTABLE=$(which python3)

echo "Detected Python $PYTHON_VERSION at $PYTHON_EXECUTABLE"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing now..."
    $PYTHON_EXECUTABLE -m pip install uv
fi

# Create a virtual environment using uv
echo "Creating a virtual environment with uv..."
uv venv .venv

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install the package and test dependencies
echo "Installing dependencies with uv..."
uv pip install -e ".[dev]"
uv pip install pytest-asyncio httpx

# Create log directory if it doesn't exist
mkdir -p log

# Run the main tests
echo "Running main tests..."
python -m pytest tests/ -v

# Run the verification for call_* test tools
echo "Verifying call_* scripts..."
chmod +x test_tools/verify_call_scripts.sh
bash test_tools/verify_call_scripts.sh

# Clean up
echo "All tests completed. Deactivating virtual environment..."
deactivate

echo "All tests completed successfully!" 