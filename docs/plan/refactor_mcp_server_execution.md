# MCP Server Refactoring Execution Plan

This document tracks the execution of the refactoring plan outlined in `refactor_mcp_servers.md`.

## Refactoring Tasks

- [x] **1. Setup Directory Structure:** Create the new directories (`common/`, `models/`) and empty Python files.
- [x] **2. Implement `common/config.py`:** Centralize all environment variable loading and logging setup.
- [x] **3. Implement `common/llm_client.py`:** Create the `httpx` client wrapper and abstract API call patterns.
- [x] **4. Implement `common/mcp_manager.py`:** Consolidate all MCP tool management, caching, injection, and execution logic.
- [x] **5. Develop `responses_service.py`:** Migrate and refactor logic from `server.py`, focusing on the `/responses` endpoint.
- [x] **6. Develop `chat_completions_service.py`:** Migrate and refactor logic for the `/v1/chat/completions` endpoint.
- [x] **7. Build `api_controller.py`:** Set up the FastAPI app, routing, and lifecycle events.
- [x] **8. Create `server_entrypoint.py`:** Add the uvicorn server runner.
- [x] **9. Pydantic Models:** Review and move all Pydantic models to the `models/` directory for better organization.
- [x] **10. Update Imports:** Adjust imports across all refactored files to reflect the new structure.
- [ ] **11. Testing:** Thoroughly test all endpoints and functionalities.
- [x] **12. Cleanup:** Remove or archive the old `server.py` file.

## Newly Discovered Tasks 