#!/bin/bash
# Source common configuration
source "$(dirname "$0")/config.sh"

echo "Calling MCP proxy for chat completions with model: ${MODEL_ID}"
echo ""
echo "User: Just call 'list_tables' action"
echo "..."
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
# {"id":"chatcmpl-BmcEY89U4SIv18BI1LENuuSY6stq4","object":"chat.completion","created":1750926414,"model":"gpt-4o-2024-08-06","choices":[{"index":0,"message":{"role":"assistant","content":"It seems that there is a single table named `listings` in the database.","refusal":null,"annotations":[]},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":230,"completion_tokens":18,"total_tokens":248,"prompt_tokens_details":{"cached_tokens":0,"audio_tokens":0},"completion_tokens_details":{"reasoning_tokens":0,"audio_tokens":0,"accepted_prediction_tokens":0,"rejected_prediction_tokens":0}},"service_tier":"default","system_fingerprint":"fp_07871e2ad8"}
# I just want the text content of the response
response=$(echo "$response" | jq -r '.choices[0].message.content')
echo "AI Response: $response"