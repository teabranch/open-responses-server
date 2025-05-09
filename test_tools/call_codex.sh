#!/bin/bash
# Source common configuration
source "$(dirname "$0")/config.sh"

npm install -g @openai/codex

codex --provider Meta --quiet --model "${MODEL_ID}" "list all files"