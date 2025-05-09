#!/bin/bash
# This script runs the random number MCP server

# Make sure we're in the right directory
cd "$(dirname "$0")/.." || exit 1

# Run the MCP server
python test_tools/random_number_mcp_server.py