#!/bin/bash
# Source common configuration
source "$(dirname "$0")/config.sh"

nvm install 22.15.1
nvm use 22.15.1
npm install -g @openai/codex

codex --provider Meta --quiet --model "${MODEL_ID}" "I gave you sqlite access via tools. Confirm you see that tool. List all tables in the database. Create 'test_table' if none exist. When errors, let me know what the error is"