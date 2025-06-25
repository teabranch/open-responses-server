#!/bin/bash
# Common configuration for test scripts
# This file should be sourced by other scripts

# Default model ID to use if DEFAULT_MODEL_ID environment variable is not set
DEFAULT_MODEL_FALLBACK="gpt-4o"
# For codex script which uses a different default model
DEFAULT_CODEX_MODEL_FALLBACK="gpt-4o"

if [ "$USE_CODEX" = "true" ]; then
    MODEL_ID=${DEFAULT_MODEL_ID:-"$DEFAULT_CODEX_MODEL_FALLBACK"}
else
    MODEL_ID=${DEFAULT_MODEL_ID:-"$DEFAULT_MODEL_FALLBACK"}
fi
