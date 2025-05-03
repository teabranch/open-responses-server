#!/bin/bash
curl ${OPENAI_BASE_URL_INTERNAL}/v1/chat/completions -s \
-H "Content-Type: application/json" \
-H "Authorization: Bearer ${OPENAI_API_KEY}" \
-d '{
"model": "meta-llama/llama-4-scout-17b-16e-instruct",
"messages": [{
    "role": "user",
    "content": "Say 3 words, and 3 words only."
}]
}'