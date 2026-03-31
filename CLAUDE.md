# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

Open Responses Server is a proxy that translates OpenAI's Responses API into Chat Completions API calls, allowing any OpenAI-compatible backend (Ollama, vLLM, LiteLLM, Groq, etc.) to serve the Responses API. This enables tools like OpenAI's Codex CLI to work with self-hosted or third-party LLMs.

## Build & Run

```bash
# Install from source (uses uv)
pip install uv
uv venv
uv pip install -e ".[dev]"

# Start the server
otc start
# Or directly:
uv run src/open_responses_server/cli.py start
```

## Testing

```bash
# Run all tests (creates venv, installs deps, runs pytest + verification scripts)
./run_tests.sh

# Run tests manually (after activating venv)
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_server.py -v

# Run a specific test
python -m pytest tests/test_cli.py::TestCLI::test_start_server_imports -v

# Coverage
python -m pytest --cov=open_responses_server tests/
```

## Linting

```bash
# The project uses flake8 + bandit + pylint
flake8 src/
pylint src/open_responses_server/
bandit -r src/
```

## Architecture

The server is a FastAPI application with two main API paths that both proxy to a backend LLM:

**Request flow:**
```
Client (Codex, etc.)
  |
  v
api_controller.py  -- FastAPI app with route definitions, CORS, startup/shutdown
  |
  +-- POST /responses  -->  responses_service.py
  |     Converts Responses API format to Chat Completions format,
  |     streams SSE events back in Responses API format
  |
  +-- POST /v1/chat/completions  -->  chat_completions_service.py
  |     Proxies with MCP tool injection and automatic tool-call loops
  |
  +-- GET/POST /{path}  -->  Generic proxy to backend (e.g. /v1/models)
```

**Key modules:**
- `api_controller.py` - Route definitions. `server_entrypoint.py` is the uvicorn entry point that imports from `api_controller`.
- `responses_service.py` - Converts Responses API requests to Chat Completions format (`convert_responses_to_chat_completions`), processes streaming Chat Completions responses back into Responses API SSE events (`process_chat_completions_stream`). Maintains in-memory `conversation_history` keyed by `previous_response_id`.
- `chat_completions_service.py` - Handles `/v1/chat/completions` with MCP tool injection. Implements a tool-call loop (up to `MAX_TOOL_CALL_ITERATIONS`) for both streaming and non-streaming modes.
- `common/mcp_manager.py` - `MCPManager` singleton manages MCP server lifecycle (stdio, sse, streamable-http transports), tool discovery/caching with periodic refresh, and tool execution. `MCPServer` wraps individual server sessions.
- `common/llm_client.py` - `LLMClient` singleton wrapping `httpx.AsyncClient`, pointed at `OPENAI_BASE_URL_INTERNAL`.
- `common/config.py` - All configuration via environment variables (loaded from `.env` via python-dotenv). Key vars: `OPENAI_BASE_URL_INTERNAL`, `OPENAI_API_KEY`, `MCP_SERVERS_CONFIG_PATH`, `MAX_TOOL_CALL_ITERATIONS`.
- `models/responses_models.py` - Pydantic models for Responses API request/response/streaming types.

**MCP tool flow:** On startup, MCP servers (defined in `servers_config.json` via `MCP_SERVERS_CONFIG_PATH`) are initialized. Their tools are cached and injected into both `/responses` and `/v1/chat/completions` requests. When the LLM returns tool calls for MCP tools, the server executes them and feeds results back in a loop.

## Configuration

All config is via environment variables (see `common/config.py`). The CLI command `otc configure` writes a `.env` file interactively. MCP servers are configured in a JSON file pointed to by `MCP_SERVERS_CONFIG_PATH` (default: `src/open_responses_server/servers_config.json`). Note: this default path assumes running from the repo root; when installed via pip, set it to an absolute path.

**Important:** The `/responses` endpoint only supports streaming (`stream=True`). Non-streaming requests return HTTP 501.

## Version & Releasing

Version lives in `src/open_responses_server/version.py` as `__version__` — the single source of truth. Both `pyproject.toml` (via setuptools dynamic) and `__init__.py` read from it.

**To release a new version:** bump `__version__` in `version.py`, merge to main. The `publish.yml` workflow handles everything automatically:
1. Detects `version.py` changed on main
2. Creates a GitHub Release (v{version})
3. Builds with `uv build` + `twine check`
4. Publishes to PyPI via trusted publishing (OIDC)

## CLI Entry Point

The `otc` command is defined in `pyproject.toml` pointing to `open_responses_server.cli:main`. Commands: `start`, `configure`, `help`. Also supports `--version` flag.

## PR Workflow

When a plan is ready and implementation is complete, create a PR and run the `/pr-review` skill. The PR review workflow should: wait 5 minutes for automated reviewers to run, then check every 2 minutes for completion before fetching and addressing review comments.
