#!/bin/bash

# Load configuration variables
source "$(dirname "$0")/config.sh"

# Function to start the MCP Simple Prompt server
start_mcp_server() {
    local transport=$1
    local port=$2

    echo "Starting MCP Simple Prompt server with $transport transport on port $port..."
    uv run mcp-simple-prompt --transport "$transport" --port "$port" &
    MCP_PID=$!
    sleep 2  # Give the server some time to start
}

# Function to stop the MCP Simple Prompt server
stop_mcp_server() {
    echo "Stopping MCP Simple Prompt server..."
    kill "$MCP_PID"
    wait "$MCP_PID" 2>/dev/null
}

# Test MCP in stdio mode
start_mcp_server "stdio" 8001
echo "Testing MCP in stdio mode..."

curl localhost:8080/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "'"${MODEL_ID}"'",
    "input": "Say 3 words, and 3 words only.",
    "max_output_tokens": 100,
    "temperature": 0.7,
    "stream": true
  }'
stop_mcp_server

# Test MCP in streamable SSE mode
start_mcp_server "sse" 8001
echo "Testing MCP in streamable SSE mode..."

curl localhost:8080/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "'"${MODEL_ID}"'",
    "input": "Say 3 words, and 3 words only.",
    "max_output_tokens": 100,
    "temperature": 0.7,
    "stream": true
  }'
stop_mcp_server

echo "All tests completed."