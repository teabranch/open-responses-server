# Refactor Server Logic

## 1. Goal

The primary goal is to refactor the server by moving the correct and working logic from the monolithic `src/open_responses_server/server.py` file into the new modular structure comprised of `src/open_responses_server/api_controller.py`, `src/open_responses_server/responses_service.py`, and `src/open_responses_server/common/mcp_manager.py`. The `server.py` file will be deleted after the refactoring.

This will centralize the server's business logic within the service layer, making the codebase cleaner, easier to maintain, and simplifying the roles of the controller and entrypoint.

## 2. Plan

The refactoring will be performed in the following steps:

### Step 1: Consolidate Pydantic Models

The Pydantic models currently defined inside `server.py` will be compared with those in `src/open_responses_server/models/responses_models.py`. The `responses_models.py` file will be updated to be the single source of truth for all response-related models, ensuring it is complete and correct based on the working implementation in `server.py`. Any missing models or fields will be added.

### Step 2: Verify `mcp_manager`

The MCP handling logic in `server.py` (including the `MCPServer` class and helper functions like `execute_mcp_tool`) will be compared against the implementation in `src/open_responses_server/common/mcp_manager.py`. The `mcp_manager.py` module appears to be a well-refactored version, using a singleton pattern. We will confirm it contains all necessary functionality and make it the sole component for managing MCP servers and tools. The logic inside `responses_service.py` will be updated to use the `mcp_manager` instance exclusively.

### Step 3: Refactor `responses_service.py`

This is the core of the refactoring effort. The logic from `server.py` will be migrated to `responses_service.py`.

-   **`convert_responses_to_chat_completions`**: The more complete and correct implementation from `server.py` will replace the existing one in `responses_service.py`.
-   **`process_chat_completions_stream`**: The working implementation from `server.py` will replace the one in `responses_service.py`. This function will be adapted to use the `mcp_manager` singleton for all tool-related operations.
-   **Conversation History**: The in-memory conversation history mechanism from `server.py` will be moved into `responses_service.py`, making the service stateful as intended.
-   **Dependencies**: All dependencies (such as logging, models, and the MCP manager) will be updated to use the project's common modules (`common.config`, `models.responses_models`, etc.), ensuring consistency.

### Step 4: Update the API Controller

The `src/open_responses_server/api_controller.py` will be reviewed. The logic for injecting MCP tools into incoming requests will be simplified or moved into the `responses_service` to keep the controller thin and focused on handling HTTP requests and responses.

### Step 5: Cleanup

-   Delete the now-redundant `src/open_responses_server/server.py` file.
-   Confirm that the application entrypoint, `src/open_responses_server/server_entrypoint.py`, functions correctly with the refactored code.
-   Review project configuration files like `pyproject.toml` and any testing scripts to remove references to the old `server.py` file and ensure they align with the new structure. 