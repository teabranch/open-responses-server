## Plan: Enhance Chat Completions Endpoint with MCP Tool Use

**1. Goal:**
Integrate MCP (Model Context Protocol) tool use capabilities into the existing `/v1/chat/completions` proxy endpoint. This will allow clients using the standard OpenAI chat completions format to leverage tools provided by MCP servers, similar to how the `/responses` endpoint currently functions.

**2. Current Behavior:**
The `/v1/chat/completions` endpoint currently acts as a simple proxy. It forwards requests directly to the `OPENAI_BASE_URL_INTERNAL` without any modification or special handling for tools beyond what the underlying provider supports. It does not inject MCP tools or process MCP tool calls.

**3. Desired Behavior:**
The enhanced `/v1/chat/completions` endpoint should:
*   Inspect incoming chat completion requests for tool definitions and tool choices.
*   Inject available MCP tools into the request, augmenting any tools already defined by the client, similar to how the `/responses` endpoint handles this. Priority should be given to client-defined tools in case of naming conflicts.
*   If the LLM response indicates a tool call for an MCP tool:
    *   Execute the MCP tool.
    *   Format the MCP tool's output into a new message in the chat history.
    *   Potentially make a subsequent call to the LLM with the updated chat history (including the tool's output) to get a final user-facing response. This behavior needs to be carefully considered to align with user expectations for the chat completions API.
*   If the LLM response indicates a tool call for a non-MCP tool (client-defined), it should be returned to the client as per the standard OpenAI API behavior.
*   Support both streaming and non-streaming responses.
*   Maintain compatibility with the standard OpenAI chat completions API for requests that do not involve MCP tools.

**4. Key Changes Required:**

*   **File:** `src/open_responses_server/server.py`
*   **Function to Modify:** `proxy_endpoint` (or potentially a new dedicated function for `/v1/chat/completions` if the logic becomes too complex for the generic proxy).

*   **Logic for MCP Tool Injection:**
    *   Before forwarding the request to `OPENAI_BASE_URL_INTERNAL`, check if the request path is `/v1/chat/completions`.
    *   If so, parse the request body (JSON).
    *   Retrieve the `mcp_functions_cache`.
    *   Merge MCP tools with any tools already present in the `chat_request["tools"]` array.
        *   Ensure MCP tools are formatted correctly according to the OpenAI `tools` specification (e.g., `{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}`).
        *   Handle potential naming conflicts (e.g., client-defined tools take precedence).
    *   Update the request body with the augmented list of tools.

*   **Logic for Handling LLM Tool Call Responses (for MCP Tools):**
    *   **Non-Streaming:**
        *   After receiving the response from `OPENAI_BASE_URL_INTERNAL`, parse the JSON response.
        *   Check if the response contains `tool_calls` and if any of those calls are for MCP tools (use `is_mcp_tool` function).
        *   If an MCP tool call is found:
            *   Extract the tool name and arguments.
            *   Execute the MCP tool using `execute_mcp_tool`.
            *   Construct a new "tool" role message containing the output of the MCP tool.
            *   *Decision Point:*
                *   **Option A (Simpler Proxy):** Return the initial LLM response containing the MCP tool call directly to the client, letting the client handle the execution and follow-up. This is simpler but doesn't fully "hide" MCP.
                *   **Option B (Full Integration):** Make a *new* request to the LLM, appending the "assistant" message (with the tool call) and the new "tool" message (with the result) to the original `messages` array. The response from this second LLM call would then be returned to the client. This provides a more seamless integration but adds complexity and latency.
                *   *Initial implementation should likely lean towards Option A or make this configurable, clearly documenting the behavior.*
    *   **Streaming:**
        *   This is more complex. As chunks arrive from `OPENAI_BASE_URL_INTERNAL`:
            *   Accumulate `tool_calls` delta chunks.
            *   When a full MCP tool call is identified (e.g., by `finish_reason: "tool_calls"` or by parsing the accumulated arguments):
                *   *Decision Point (similar to non-streaming):*
                    *   **Option A (Stream back tool call):** Stream the tool call information back to the client as it's received. The client would then be responsible for executing the MCP tool and continuing the conversation.
                    *   **Option B (Execute and continue stream):** Pause streaming to the client, execute the MCP tool, construct the "tool" message, make a new streaming request to the LLM with the updated messages, and then resume streaming the new LLM response to the client. This is significantly more complex to manage within a streaming context.
                    *   *Initial implementation should likely lean towards Option A for streaming.*

*   **Helper Functions:**
    *   Leverage existing functions like `is_mcp_tool` and `execute_mcp_tool`.
    *   Potentially adapt parts of `convert_responses_to_chat_completions` for tool merging logic if applicable.
    *   Adapt parts of `process_chat_completions_stream` if Option B for streaming is chosen, to handle the interruption and continuation.

**5. Considerations:**

*   **Streaming Complexity:** Full integration (Option B) for streaming responses is challenging. It requires careful state management to pause the original stream, execute the tool, initiate a new stream to the LLM, and then proxy this new stream back to the client, all while appearing as a single continuous stream.
*   **Error Handling:** Robust error handling is needed for MCP tool execution failures. How should these errors be reported back to the chat completions client? (e.g., as a message in the chat, or an error in the response object).
*   **API Compatibility:** Changes must be made carefully to avoid breaking compatibility for clients not using tools or using only standard OpenAI tools. The endpoint should still behave as a standard OpenAI endpoint if no MCP-specific interactions occur.
*   **Configuration:** Consider adding configuration options to enable/disable this enhanced MCP processing for the chat completions endpoint, or to choose between different handling strategies (e.g., Option A vs. B for tool call results).
*   **Idempotency and State:** If the server makes a second call to the LLM (Option B), it needs to manage the conversation state correctly.
*   **`tool_choice` Parameter:** How should `tool_choice` interact with injected MCP tools? The logic should respect the user's `tool_choice` if it specifies a particular (MCP or non-MCP) tool.

**6. File to be created:**
`/docs/plan/chat_completions_mcp_enhancement.md` (This document)

This plan outlines the major areas of work. Detailed implementation choices, especially around handling tool call results (Option A vs. B), will need further refinement during development.
