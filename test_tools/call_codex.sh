#!/bin/bash
# Source common configuration
source "$(dirname "$0")/config.sh"

nvm install 22.15.0
npm install -g @openai/codex

codex --provider Meta --model "${MODEL_ID}"