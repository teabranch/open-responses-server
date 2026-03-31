---
title: CLI Usage
nav_order: 5
---

## Overview

The `otc` command is the CLI entry point for Open Responses Server, defined in
`pyproject.toml` pointing to `open_responses_server.cli:main`.

## Commands

| Command | Description |
| --- | --- |
| `otc start` | Start the FastAPI server |
| `otc configure` | Interactive configuration wizard (saves to `.env`) |
| `otc help` | Display help information |
| `otc --version` | Show version information |

## Running after installation

```bash
# After pip install or uv pip install
otc start
otc configure
otc --version
```

## Running from source

```bash
# Using uv
uv run src/open_responses_server/cli.py start

# Or directly with Python (venv must be activated)
python src/open_responses_server/cli.py start
```

## Start command

Starts the FastAPI server via uvicorn. The server binds to the host and port
defined by `API_ADAPTER_HOST` and `API_ADAPTER_PORT` environment variables
(defaults: `0.0.0.0:8080`).

```bash
otc start
```

## Configure command

Interactive wizard that prompts for host, port, backend URL, external URL, and
API key. Saves the configuration to a `.env` file in the current directory,
merging with any existing values.

```bash
otc configure
```
