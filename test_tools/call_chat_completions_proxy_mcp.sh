#!/bin/bash
# Source common configuration
source "$(dirname "$0")/config.sh"

echo "Calling MCP proxy for chat completions with model: ${MODEL_ID}"
# save as variable then print it
response=$(curl localhost:8080/v1/chat/completions -s \
-H "Content-Type: application/json" \
-H "Authorization: Bearer ${OPENAI_API_KEY}" \
-d '{
"model": "gpt-4o",
"messages": [{
    "role": "user",
    "content": "Just call 'list_tables' action"
}]
}')
echo "$response"