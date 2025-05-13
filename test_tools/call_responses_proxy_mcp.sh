#!/bin/bash
# Source common configuration
source "$(dirname "$0")/config.sh"

curl localhost:8080/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "'"${MODEL_ID}"'",
    "input": "Navigate to https://randomwordgenerator.com/letter.php and tell me what you see.",
    "max_output_tokens": 100,
    "temperature": 0.7,
    "stream": true
  }'
