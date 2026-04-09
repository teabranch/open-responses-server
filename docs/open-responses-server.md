---
title: Architecture
nav_order: 1
---

## Open Responses Server — Architecture Overview

## What It Does

Open Responses Server is a FastAPI proxy that translates OpenAI's Responses API
into Chat Completions API calls, allowing any OpenAI-compatible backend (Ollama,
vLLM, LiteLLM, Groq, or OpenAI itself) to serve the Responses API. It also
provides a `/v1/chat/completions` endpoint with automatic MCP tool injection and
a tool-call execution loop, plus a generic proxy for all other endpoints.

## Module Map

| Module | Purpose |
| --- | --- |
| `api_controller.py` | FastAPI app, route definitions, CORS, startup/shutdown hooks, MCP tool injection for `/responses` |
| `responses_service.py` | Responses-to-ChatCompletions request conversion, SSE stream processing, conversation history |
| `chat_completions_service.py` | `/v1/chat/completions` handler with MCP tool injection and tool-call loop (streaming + non-streaming) |
| `common/config.py` | All configuration via environment variables, logging setup |
| `common/llm_client.py` | `LLMClient` singleton wrapping `httpx.AsyncClient` pointed at the backend |
| `common/mcp_manager.py` | `MCPManager` singleton: MCP server lifecycle, tool discovery/caching, execution, result serialization |
| `models/responses_models.py` | Pydantic models for all Responses API types and SSE streaming events |
| `server_entrypoint.py` | Uvicorn entry point (imports `app` from `api_controller`) |
| `cli.py` | `otc` CLI: `start`, `configure`, `help` commands |
| `version.py` | `__version__` string, read dynamically by setuptools |

## Request Routing

```text
Client
  │
  ├─ POST /responses
  │    → api_controller.create_response()
  │    → MCP tools injected into request
  │    → responses_service.convert_responses_to_chat_completions()
  │    → LLM backend POST /v1/chat/completions (streaming)
  │    → responses_service.process_chat_completions_stream()
  │    → SSE events in Responses API format back to client
  │
  ├─ POST /v1/chat/completions
  │    → api_controller.chat_completions()
  │    → chat_completions_service.handle_chat_completions()
  │    → MCP tools injected into request
  │    → Tool-call loop (up to MAX_TOOL_CALL_ITERATIONS)
  │    → Final response streamed or returned as JSON
  │
  ├─ GET /health → {"status": "ok", "adapter": "running"}
  ├─ GET /       → {"message": "Open Responses Server is running."}
  │
  └─ GET/POST /{path} (catch-all proxy)
       → Forwarded to LLM backend at /v1/{path}
       → Response proxied back (streaming or non-streaming)
```

## Configuration

All configuration is via environment variables, loaded from `.env` via
`python-dotenv`. Defined in `common/config.py`.

| Variable | Default | Description |
| --- | --- | --- |
| `OPENAI_BASE_URL_INTERNAL` | `http://localhost:8000` | Backend LLM API URL |
| `OPENAI_BASE_URL` | `http://localhost:8080` | This server's external URL |
| `OPENAI_API_KEY` | `dummy-key` | API key passed to backend |
| `API_ADAPTER_HOST` | `0.0.0.0` | Server bind address |
| `API_ADAPTER_PORT` | `8080` | Server port |
| `MCP_TOOL_REFRESH_INTERVAL` | `10` | Seconds between MCP tool cache refreshes |
| `MCP_SERVERS_CONFIG_PATH` | `src/open_responses_server/servers_config.json` | Path to MCP servers JSON config (use absolute path when pip-installed) |
| `MAX_CONVERSATION_HISTORY` | `100` | Max stored conversation entries |
| `MAX_TOOL_CALL_ITERATIONS` | `25` | Max tool-call loop iterations |
| `STREAM_TIMEOUT` | `120.0` | Backend read timeout in seconds; connect timeout stays at 30s |
| `HEARTBEAT_INTERVAL` | `15.0` | SSE heartbeat event interval in seconds |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `LOG_FILE_PATH` | `./log/api_adapter.log` | Path to log file |

### MCP Server Configuration

The JSON file at `MCP_SERVERS_CONFIG_PATH` defines MCP servers. Three transport
types are supported: `stdio` (default), `sse`, and `streamable-http`.

```json
{
  "mcpServers": {
    "stdio-server": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      "env": {"KEY": "value"}
    },
    "sse-server": {
      "type": "sse",
      "url": "http://example.com/sse",
      "headers": {"Authorization": "Bearer token"}
    },
    "http-server": {
      "type": "streamable-http",
      "url": "http://example.com/mcp",
      "headers": {"Authorization": "Bearer token"}
    }
  }
}
```

The `type` field defaults to `stdio` if omitted. Stdio servers use `command`,
`args`, and `env` fields. SSE and streamable-http servers use `url` and optional
`headers` fields.

## Startup / Shutdown Lifecycle

### Startup

1. **LLM Client** — `startup_llm_client()` creates the `httpx.AsyncClient`
   singleton pointed at `OPENAI_BASE_URL_INTERNAL`
2. **MCP Servers** — `mcp_manager.startup_mcp_servers()`:
   - Loads `servers_config.json`
   - Initializes each `MCPServer` via stdio (subprocess)
   - Populates tool cache with initial tool discovery
   - Builds `tool_name → server_name` mapping for fast lookup
   - Starts background refresh task (periodic tool re-discovery)

### Shutdown

1. **LLM Client** — closes `httpx.AsyncClient`
2. **MCP Servers** — cancels refresh task, calls `cleanup()` on each server

## Conversation History

In-memory dictionary keyed by `response_id`, storing the full message list for
multi-turn conversations. Loaded via `previous_response_id` in subsequent
requests. Trimmed when exceeding `MAX_CONVERSATION_HISTORY`.

See [Events & Tool Handling](events-and-tool-handling.md#conversation-history)
for details on save points and message validation.

## Further Reading

- [Events & Tool Handling](events-and-tool-handling.md) — SSE event types,
  emission sequences, tool lifecycle
- [API Flow Diagrams](responses_flow.md) — Mermaid sequence diagrams for both
  endpoints
- [Testing Guide](testing-guide.md) — Running tests and writing new ones
- [CLI Usage](cli-local.md) — `otc` commands
