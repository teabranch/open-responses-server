#!/bin/bash

npm install -g @openai/codex

curl localhost:8080/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "meta-llama/llama-4-scout-17b-16e-instruct",
    "input": "Say 3 words, and 3 words only.",
    "max_output_tokens": 100,
    "temperature": 0.7,
    "stream": true
  }'
