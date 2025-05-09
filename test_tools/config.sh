#!/bin/bash
# Common configuration for test scripts
# This file should be sourced by other scripts

# Default model ID to use if DEFAULT_MODEL_ID environment variable is not set
DEFAULT_MODEL_FALLBACK="meta-llama/llama-4-scout-17b-16e-instruct"
# For codex script which uses a different default model
DEFAULT_CODEX_MODEL_FALLBACK="meta-llama/llama-4-maverick-17b-128e-instruct"

MODEL_ID=${DEFAULT_MODEL_ID:-"$DEFAULT_MODEL_FALLBACK"}