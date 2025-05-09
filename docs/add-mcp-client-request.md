Read README, docs/add-mcp-client-support.md and src/openai-responses-server/server.py

Add support for MCP

It should initialize, load the tools, and run them via MCP when the AI Provider requested them.

Then send the result (When result ready) to the AI Provider and continue with the flow.