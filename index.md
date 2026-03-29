---
title: Home
nav_order: 0
permalink: /
---

## Open Responses Server

A plug-and-play server that speaks OpenAI's Responses API — no matter which AI
backend you're running.

Ollama, vLLM, LiteLLM, Groq, or even OpenAI itself — this server bridges them
all to the OpenAI Responses API interface. It handles stateful chat, tool calls,
and MCP server integration behind a familiar API.

---

## Quick Start

### Install

```bash
pip install open-responses-server
```

Or from source:

```bash
pip install uv
uv venv
uv pip install -e ".[dev]"
```

### Configure

```bash
otc configure
```

Or set environment variables:

```bash
export OPENAI_BASE_URL_INTERNAL=http://localhost:11434  # Your LLM backend
export OPENAI_BASE_URL=http://localhost:8080             # This server
export OPENAI_API_KEY=sk-your-key
```

### Run

```bash
otc start
```

Verify:

```bash
curl http://localhost:8080/v1/models
```

---

## Documentation

| Page | Description |
| --- | --- |
| [Architecture](docs/open-responses-server.md) | Module map, request routing, configuration reference |
| [Events & Tools](docs/events-and-tool-handling.md) | SSE event types, emission sequences, tool lifecycle |
| [API Flow Diagrams](docs/responses_flow.md) | Mermaid sequence diagrams for both endpoints |
| [Testing Guide](docs/testing-guide.md) | Running tests, writing tests, coverage |
| [CLI Usage](docs/cli-local.md) | `otc` commands and options |
| [Extending](docs/extend-instructions.md) | Web search and RAG extension guide |
| [Security](docs/SECURITY.md) | Security scanning setup and policies |

---

## Key Features

- Drop-in replacement for OpenAI's Responses API
- Works with any OpenAI-compatible backend
- MCP server support for both Chat Completions and Responses APIs
- Supports OpenAI's Codex CLI and other Responses API clients
- Stateful multi-turn conversations via in-memory history
- Tool call execution loop with configurable iteration limits
