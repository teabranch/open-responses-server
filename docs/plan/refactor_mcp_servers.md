# Refactoring Plan: Modularizing API Server Logic

## 1. Introduction and Goals

This document outlines a plan to refactor the existing Python FastAPI server components (`chat_completions_server.py` and `server.py`) into a more modular and maintainable architecture.

**Current State:**
*   `chat_completions_server.py`: Primarily handles `/v1/chat/completions` with specific MCP (Model Context Protocol) tool integration, including a two-stage LLM call process for streaming responses with tool execution. It also contains a basic proxy.
*   `server.py`: Implements a custom `/responses` endpoint that translates to/from the chat completions API format, injects MCP tools, and includes a more generic proxy for other OpenAI-compatible endpoints.
*   Both files share functionalities like `httpx` client usage, MCP tool loading/injection, streaming response handling, and configuration via environment variables, leading to redundancy.

**Goals of Refactor:**
*   **Modularity:** Separate distinct functionalities into their own modules.
*   **Reusability:** Create common utilities for configuration, LLM API interactions, and MCP management.
*   **Clarity:** Improve code readability and understanding by isolating concerns.
*   **Maintainability:** Make it easier to update and extend specific parts of the server without affecting others.
*   **Single Entry Point:** Consolidate the FastAPI application into a central controller.

## 2. Proposed Architecture

The refactored architecture will consist of the following main components within the `src/open_responses_server/` directory:

```
src/open_responses_server/
├── __init__.py
├── api_controller.py          # Main FastAPI app, routing, shared dependencies
├── responses_service.py       # Business logic for the /responses endpoint
├── chat_completions_service.py # Business logic for /v1/chat/completions endpoint
├── common/
│   ├── __init__.py
│   ├── config.py              # Centralized configuration loading & logging setup
│   ├── llm_client.py          # Wrapper for httpx calls to the LLM API
│   └── mcp_manager.py         # MCP tool loading, caching, injection, and execution
├── models/
│   └── __init__.py            # Pydantic models (consolidate if needed)
└── server_entrypoint.py       # Script to run the uvicorn server
# ... (other existing files like cli.py, version.py)
```

## 3. Detailed Component Breakdown

### 3.1. `common/config.py`
*   **Responsibilities:**
    *   Load all environment variables (e.g., `OPENAI_BASE_URL_INTERNAL`, `OPENAI_API_KEY`, `API_ADAPTER_HOST`, `API_ADAPTER_PORT`, `MCP_TOOL_REFRESH_INTERVAL`, `MAX_CONVERSATION_HISTORY`).
    *   Define shared constants.
    *   Configure global logging settings (format, level, handlers).
*   **Source:** Consolidate from `chat_completions_server.py` and `server.py`.

### 3.2. `common/llm_client.py`
*   **Responsibilities:**
    *   Initialize and manage a shared `httpx.AsyncClient` instance, configured with `OPENAI_BASE_URL_INTERNAL` and `OPENAI_API_KEY`.
    *   Provide utility functions for making requests to the LLM API (e.g., `post_chat_completions_stream`, `post_chat_completions_non_stream`, generic `proxy_request`).
    *   Abstract away direct `httpx` calls, handling common headers and timeouts.
*   **Source:** Refactor `http_client` usage from both server files.

### 3.3. `common/mcp_manager.py`
*   **Responsibilities:**
    *   Manage MCP server instances and configurations (if `StdioServerParameters`, `ClientSession` logic is retained).
    *   Handle the `mcp_functions_cache`: loading, periodic refreshing (`MCP_TOOL_REFRESH_INTERVAL`).
    *   Provide `get_mcp_tools()`: Returns MCP tools formatted for API requests.
    *   `inject_mcp_tools(request_data: dict, existing_tools: Optional[List] = None) -> dict`: Merges MCP tools into request data, avoiding duplicates.
    *   `is_mcp_tool(tool_name: str) -> bool`: (Integrate from `is_mcp_tool.py`).
    *   `execute_mcp_tool(tool_name: str, tool_args: dict) -> Any`: Executes a specified MCP tool.
*   **Source:** Consolidate MCP-related logic from `chat_completions_server.py` (tool execution, cache refresh) and `server.py` (tool injection).

### 3.4. `chat_completions_service.py`
*   **Responsibilities:**
    *   Implement the core business logic for the `/v1/chat/completions` endpoint.
    *   Utilize `common.llm_client` for API calls and `common.mcp_manager` for tool handling.
    *   `handle_streaming_chat_completions_mcp(...)`: Manages the two-stage LLM call process for streaming responses when MCP tools are invoked. This will be a refined version of `_handle_chat_completions_streaming_mcp`.
    *   Handle non-streaming requests if they require specific MCP logic beyond simple proxying.
    *   Manage conversation history if it's specific to this endpoint's interactions.
*   **Source:** Primarily from `chat_completions_server.py`.

### 3.5. `responses_service.py`
*   **Responsibilities:**
    *   Implement the core business logic for the `/responses` endpoint.
    *   `convert_responses_to_chat_completions(request_data: dict) -> dict`: Translates `/responses` API format to chat completions format.
    *   Use `common.mcp_manager.inject_mcp_tools` to add tools to the converted request.
    *   Use `common.llm_client` to call the `/v1/chat/completions` endpoint (which might be routed to `chat_completions_service.py` or an external service).
    *   `process_chat_completions_stream(...)`: Convert the chat completion stream back to the `/responses` API stream format.
*   **Source:** Primarily from `server.py`.

### 3.6. `api_controller.py`
*   **Responsibilities:**
    *   Define the main FastAPI application (`app = FastAPI(...)`).
    *   Initialize and manage shared resources on startup/shutdown (e.g., `llm_client`, `mcp_manager` cache refresh).
    *   Configure CORS middleware.
    *   **Routing:**
        *   `@app.post("/v1/chat/completions")`: Delegates to `chat_completions_service.py`. It will first prepare the request by potentially injecting general tools using `mcp_manager.inject_mcp_tools` before passing it to the service.
        *   `@app.post("/responses")`: Delegates to `responses_service.py`.
        *   `@app.get("/health")`, `@app.get("/")`: Standard health and root endpoints.
        *   `@app.api_route("/{path_name:path}", ...)`: A generic proxy endpoint for all other paths, using `common.llm_client.proxy_request`. This should be robust, handling streaming and non-streaming, similar to the proxy in `server.py`.
*   **Source:** Acts as the new orchestrator, combining FastAPI app setup from both original files.

### 3.7. `server_entrypoint.py`
*   **Responsibilities:**
    *   Provide the `if __name__ == "__main__":` block to run the FastAPI application using `uvicorn`.
    *   Import `app` from `api_controller.py` and configuration (host, port) from `common.config.py`.
*   **Source:** Replaces the `if __name__ == "__main__":` blocks in the original server files.

### 3.8. `models/`
*   Consolidate Pydantic models used for request/response validation across the application.

## 4. Refactoring Steps (High-Level)

1.  **Setup Directory Structure:** Create the new directories (`common/`, `models/`) and empty Python files.
2.  **Implement `common/config.py`:** Centralize all environment variable loading and logging setup.
3.  **Implement `common/llm_client.py`:** Create the `httpx` client wrapper and abstract API call patterns.
4.  **Implement `common/mcp_manager.py`:** Consolidate all MCP tool management, caching, injection, and execution logic. Integrate `is_mcp_tool.py`.
5.  **Develop `chat_completions_service.py`:** Migrate and refactor logic from `chat_completions_server.py`, focusing on the `/v1/chat/completions` endpoint and its MCP interactions.
6.  **Develop `responses_service.py`:** Migrate and refactor logic from `server.py`, focusing on the `/responses` endpoint, its request/response translation, and MCP integration.
7.  **Build `api_controller.py`:**
    *   Set up the FastAPI app.
    *   Implement routing, delegating to the respective service modules.
    *   Implement the generic proxy endpoint.
    *   Manage application lifecycle events (startup/shutdown) for initializing/closing resources.
8.  **Create `server_entrypoint.py`:** Add the uvicorn server runner.
9.  **Pydantic Models:** Review and move all Pydantic models to the `models/` directory for better organization.
10. **Update Imports:** Adjust imports across all refactored files to reflect the new structure.
11. **Testing:** Thoroughly test all endpoints and functionalities, including streaming, MCP tool usage, and proxy behavior.
12. **Cleanup:** Remove or archive the old `chat_completions_server.py` and `server.py` files.

## 5. Key Considerations

*   **Error Handling:** Establish a consistent error handling and response strategy across services.
*   **Conversation History:** The `conversation_history` dictionary from `chat_completions_server.py` needs its usage clarified. If it's essential for maintaining context across multiple requests for a specific flow (e.g., within a chat session), its management should be integrated into the relevant service (likely `chat_completions_service.py`) or a dedicated state management utility if complexity warrants.
*   **Dependencies:** Ensure all necessary dependencies (like `python-dotenv`, `fastapi`, `uvicorn`, `httpx`, `mcp`) are correctly listed in `pyproject.toml` or `requirements.txt`.
*   **Async Operations:** Maintain proper use of `async` and `await` throughout the refactored codebase.
*   **Configuration of `OPENAI_BASE_URL` vs `OPENAI_BASE_URL_INTERNAL`**: Ensure clarity on which URL is used for which calls. The `llm_client.py` should primarily use `OPENAI_BASE_URL_INTERNAL` for its direct calls. The generic proxy might need to be flexible or also target this internal URL.

This refactoring aims to create a cleaner, more organized, and scalable codebase for the API server.
