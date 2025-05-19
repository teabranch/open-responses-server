#!/bin/bash
# Source common configuration
source "$(dirname "$0")/config.sh"

nvm install 22.15.1
nvm use 22.15.1
npm install -g @openai/codex

codex --provider Meta --quiet --model "${MODEL_ID}" "list all files"