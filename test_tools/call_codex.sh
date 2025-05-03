#!/bin/bash
nvm install 22.15.0
npm install -g @openai/codex

codex --provider Meta --model meta-llama/llama-4-maverick-17b-128e-instruct