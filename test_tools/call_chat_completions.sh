#!/bin/bash
# Use DEFAULT_MODEL_ID environment variable or fallback to a default
MODEL_ID=${DEFAULT_MODEL_ID:-"meta-llama/llama-4-scout-17b-16e-instruct"}

curl ${OPENAI_BASE_URL_INTERNAL}/v1/chat/completions -s \
-H "Content-Type: application/json" \
-H "Authorization: Bearer ${OPENAI_API_KEY}" \
-d '{
"model": "'"${MODEL_ID}"'",
"messages": [{
    "role": "user",
    "content": "Say 3 words, and 3 words only."
}]
}'