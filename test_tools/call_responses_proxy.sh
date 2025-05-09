#!/bin/bash
# Use DEFAULT_MODEL_ID environment variable or fallback to a default
MODEL_ID=${DEFAULT_MODEL_ID:-"meta-llama/llama-4-scout-17b-16e-instruct"}

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
