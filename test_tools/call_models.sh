#!/bin/bash
curl https://api.openai.com/v1/models -s \
-H "Content-Type: application/json" \
-H "Authorization: Bearer ${OPENAI_API_KEY}" | jq