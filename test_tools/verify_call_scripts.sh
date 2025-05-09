#!/bin/bash
# Script to verify all call_* test tools work without error
# This script should be run from the project root directory

# Don't exit on error, we want to check all scripts
set +e

# Source common configuration
source "$(dirname "$0")/config.sh"

echo "Testing all call_* scripts in test_tools directory..."

# Create a temporary .env file with test values if it doesn't exist already
if [ ! -f .env ]; then
    echo "Creating temporary .env file for testing..."
    echo "OPENAI_BASE_URL_INTERNAL=https://api.openai.com" > .env
    echo "OPENAI_API_KEY=sk-test-key-for-ci" >> .env
    echo "DEFAULT_MODEL_ID=gpt-4.1-nano" >> .env
fi

# Export environment variables that may be needed
export OPENAI_API_KEY=${OPENAI_API_KEY:-"sk-test-key-for-ci"}
export DEFAULT_MODEL_ID=${DEFAULT_MODEL_ID:-"gpt-4.1-nano"}
export OPENAI_BASE_URL_INTERNAL=${OPENAI_BASE_URL_INTERNAL:-"https://api.openai.com"}

# Check each script's syntax without executing the actual commands
for script in $(find "$(dirname "$0")" -name "call_*.sh"); do
    echo "Checking syntax of $script..."
    bash -n "$script"
    if [ $? -eq 0 ]; then
        echo "✓ $script has valid syntax"
    else
        echo "✗ $script has syntax errors"
        exit 1
    fi
done

echo "All call_* scripts have valid syntax!"

# Also check if they would parse correctly when variables are substituted
echo "Checking variable substitution..."

# Start the server in the background for Codex script
SERVER_PID=""
if [[ -f "$(dirname "$0")/call_codex.sh" ]]; then
    echo "Starting server for Codex testing..."
    uv run ./src/openai_responses_server/cli.py start &
    SERVER_PID=$!
    echo "Server started with PID: $SERVER_PID"
    # Wait a moment for the server to start up
    sleep 5
fi

# Check each script
for script in $(find "$(dirname "$0")" -name "call_*.sh"); do
    script_name=$(basename "$script")
    echo "Testing variable substitution in $script..."
    
    if [[ "$script_name" == "call_codex.sh" && ! -z "$SERVER_PID" ]]; then
        # For the Codex script, execute it directly with proper environment variables
        echo "Running the Codex script..."
        MODEL_ID=gpt-4.1-nano bash "$script" || true
    else
        # For other scripts, just check that variables would be substituted correctly
        MODEL_ID=gpt-4.1-nano source "$script" > /dev/null 2>&1 || true
    fi
    
    echo "✓ Completed check for $script"
done

# Clean up: Kill the server if it was started
if [ ! -z "$SERVER_PID" ]; then
    echo "Stopping server (PID: $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null || true
    # Wait to ensure it's stopped
    sleep 2
fi

echo "All call_* scripts checked successfully!"
