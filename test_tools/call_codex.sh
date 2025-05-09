#!/bin/bash
# Use DEFAULT_MODEL_ID environment variable or fallback to a default
MODEL_ID=${DEFAULT_MODEL_ID:-"meta-llama/llama-4-maverick-17b-128e-instruct"}

nvm install 22.15.0
npm install -g @openai/codex

codex --provider Meta --model "${MODEL_ID}"