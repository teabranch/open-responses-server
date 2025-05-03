#!/bin/bash
curl localhost:8080/models -s \
-H "Content-Type: application/json" \
-H "Authorization: Bearer ${OPENAI_API_KEY}" | jq