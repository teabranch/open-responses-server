---
title: Events & Tools
nav_order: 2
---

## Event System and Tool Handling

The `/responses` endpoint translates Chat Completions SSE chunks into Responses
API SSE events. This document covers every event type, when each is emitted,
and how tool calls flow through the system.

## SSE Event Types Reference

All events are emitted as `data: {JSON}\n\n` lines over an SSE stream.

| Event Type | Pydantic Model | When Emitted | Payload |
| --- | --- | --- | --- |
| `response.created` | `ResponseCreated` | Always first | Full `ResponseModel` (empty output, status=in_progress) |
| `response.in_progress` | `ResponseInProgress` | After created; again after each new tool call detected | Full `ResponseModel` snapshot |
| `response.output_text.delta` | `OutputTextDelta` | Per text chunk from LLM | `item_id`, `output_index`, `delta` (text fragment) |
| `response.tool_calls.created` | `ToolCallsCreated` | Legacy `delta.function_call` path only: when a new tool call arrives with a function name (not emitted for modern `delta.tool_calls` — that path emits `response.in_progress` instead) | `item_id`, `output_index`, `tool_call` dict |
| `response.function_call_arguments.delta` | `ToolCallArgumentsDelta` | Per arguments JSON fragment | `item_id`, `output_index`, `delta` (JSON fragment) |
| `response.function_call_arguments.done` | `ToolCallArgumentsDone` | When finish_reason triggers finalization | `id`, `output_index`, complete `arguments` string |
| `response.completed` | `ResponseCompleted` | Terminal event for every response | Full `ResponseModel` with status=completed |

Models are defined in `src/open_responses_server/models/responses_models.py`.

## Event Emission Sequences

All streaming logic lives in `process_chat_completions_stream()` in
`responses_service.py`. Below are the exact event orderings for each scenario.

### Scenario 1: Plain text response (finish_reason="stop")

```text
response.created
response.in_progress
response.output_text.delta   (repeated per chunk)
response.completed
```

The accumulated text is placed in `response.output[0].content[0].text`.
Conversation history is saved keyed by the response ID.

### Scenario 2: Tool calls — modern format (finish_reason="tool_calls")

This is the most common tool path. The LLM streams tool call deltas, then
signals completion with `finish_reason="tool_calls"`.

```text
response.created
response.in_progress

── per tool call ──────────────────────────────────
response.in_progress                 (first delta with function name; output now contains the tool call)
response.function_call_arguments.delta  (repeated per argument chunk)
───────────────────────────────────────────────────

── on finish_reason="tool_calls" ──────────────────
response.function_call_arguments.done   (per tool, with complete arguments)

  If MCP tool:
    → server executes tool via mcp_manager.execute_mcp_tool()
    → function_call_output added to response output
    → response.output_text.delta emitted with serialized result

  If non-MCP tool:
    → function_call left in output with status="ready"
    → client is expected to execute and return result
───────────────────────────────────────────────────

response.completed
```

### Scenario 3: Legacy function_call (finish_reason="function_call")

Older-style single function call format. Same tool execution logic applies.

```text
response.created
response.in_progress
response.tool_calls.created
response.function_call_arguments.delta  (repeated)

── on finish_reason="function_call" ───────────────
  If MCP tool:
    → executes, function_call_output + output_text.delta with result
  If non-MCP tool:
    → function_call appended with status="ready"
───────────────────────────────────────────────────

response.completed
(streaming ends immediately via return)
```

### Scenario 4: [DONE] without prior completion

If the LLM sends `[DONE]` before any `finish_reason` triggered completion:

```text
response.created
response.in_progress
(text deltas if any)
response.completed   (emitted only if status != completed)
```

### Scenario 5: Error during streaming

```text
response.completed   (with response.error field populated)
```

Emitted if an exception occurs at any point during stream processing.

## MCP vs Non-MCP Tool Distinction

### How tools are classified

`mcp_manager.is_mcp_tool(tool_name)` checks against the cached
`mcp_functions_cache` list. In the streaming implementation, tools are
effectively classified at two moments:

1. **When a new tool call is first detected in the stream** — as soon as the
   tool name appears in the delta, the corresponding output item is appended
   to `response.output` and a `response.in_progress` snapshot is emitted.
   At this point the output item status is set:
   - MCP tool: `status="in_progress"`
   - Non-MCP tool: `status="ready"`
2. **When `finish_reason` triggers execution** — determines whether to execute
   server-side or forward to client.

### MCP tools (server-executed)

1. Executed immediately via `mcp_manager.execute_mcp_tool(tool_name, args)`
2. Result serialized via `serialize_tool_result()` — extracts `.text` from each
   content item in the MCP `ToolResult`, returns JSON array of strings
3. `function_call_output` item added to `response.output`
4. `response.output_text.delta` emitted with the serialized result
5. Tool response saved in conversation history for continuity

### Non-MCP tools (client-executed)

1. Left in `response.output` with `status="ready"` and `type="function_call"`
2. Client executes the tool externally
3. Client sends result back as a `function_call_output` input item in the next
   request (with `call_id` matching the original tool call)
4. No `output_text.delta` emitted for the tool result

## Tool Injection

MCP tools from the cache are merged into requests at both endpoints.
De-duplication ensures existing tool names take priority over MCP tools.

### /responses endpoint (api_controller.py)

Two injection points:

1. **Pre-conversion** — MCP tools added to `request_data["tools"]` in Responses
   API format before `convert_responses_to_chat_completions()` runs
2. **Post-conversion** — MCP tools added to `chat_request["tools"]` in Chat
   Completions format (`{"type": "function", "function": {...}}`)

The `tool_choice` parameter is stripped before sending to the LLM.

### /v1/chat/completions endpoint (chat_completions_service.py)

Single injection point in `handle_chat_completions()`:

1. `mcp_manager.get_mcp_tools()` returns cached tool definitions
2. Each tool not already in the request is appended as
   `{"type": "function", "function": tool_dict}`
3. Existing tool names take priority (de-duplication by name)

## The Chat Completions Tool Loop

The `/v1/chat/completions` endpoint has its own tool-call loop, separate from
the `/responses` streaming logic. It loops up to `MAX_TOOL_CALL_ITERATIONS`
(default: 25, configurable via env var).

### Non-streaming path (`_handle_non_streaming_request`)

```text
for each iteration (up to MAX_TOOL_CALL_ITERATIONS):
    POST /v1/chat/completions (non-streaming) to LLM
    if finish_reason == "tool_calls":
        for each tool_call:
            if MCP tool → execute, serialize result
            if non-MCP → error message ("not a registered MCP tool")
        append tool results as role="tool" messages
        continue loop
    else:
        return final response as JSON
```

### Streaming path (`_handle_streaming_request`)

**Key insight**: the streaming path uses non-streaming requests internally
during the tool-call loop. Only the final response is streamed to the client.

```text
for each iteration (up to MAX_TOOL_CALL_ITERATIONS):
    POST /v1/chat/completions (non-streaming) to LLM
    if finish_reason == "tool_calls":
        execute tools, append results, continue loop
    else:
        append assistant message to history
        POST /v1/chat/completions (streaming) with full message history
        proxy SSE stream directly to client
        return StreamingResponse
```

This means the client sees no streaming output during intermediate tool
executions — only the final response streams.

### Reasoning parameter handling

Both paths strip the `reasoning` parameter if all its values are null. This
prevents errors with backends that don't support reasoning.

## Conversation History

### Storage

In-memory dictionary in `responses_service.py`:

```python
conversation_history: Dict[str, List[Dict[str, Any]]] = {}
```

Keyed by `response_id`. Each value is the complete message list (system,
user, assistant, tool messages).

### Loading

At the start of `convert_responses_to_chat_completions()`, if
`previous_response_id` is present and exists in the dict, the stored messages
are loaded as the starting point.

### Saving

History is saved on every completion path in `process_chat_completions_stream`:

- **finish_reason="stop"** — appends assistant text message
- **finish_reason="function_call"** — appends assistant message with tool_calls
  and tool response for MCP tools
- **finish_reason="tool_calls"** — appends assistant message with all
  tool_calls + tool responses for each executed MCP tool
- **[DONE]** — appends assistant text message if output_text_content exists

### Trimming

When the dict exceeds `MAX_CONVERSATION_HISTORY` (default: 100), the oldest
entries are removed by sorting keys.

### Message validation

`validate_message_sequence()` runs before sending messages to the LLM:

- Removes duplicate tool messages (same `tool_call_id`)
- Removes orphaned tool messages (no preceding assistant with matching
  `tool_calls`)
- Logs validation results for debugging

This prevents API errors like "messages with role 'tool' must be a response to
a preceding message with 'tool_calls'."

### function_call_output input handling

When processing Responses API input with `function_call_output` items
(client returning tool results):

1. Looks for a matching assistant message with matching `call_id` in history
2. **If found**: adds a `role="tool"` message with the result
3. **If not found**: creates a synthetic assistant message with `tool_calls`
   containing the tool name and arguments, then adds the tool response.
   This handles resuming from external tool execution.

## Pydantic Models Reference

Defined in `src/open_responses_server/models/responses_models.py`.

### Core Models

| Model | Purpose |
| --- | --- |
| `ResponseModel` | The response envelope: id, status, model, output array, tools, usage, metadata |
| `ResponseCreateRequest` | Incoming request schema: model, input, tools, stream, temperature, etc. |
| `ToolFunction` | Tool function definition: name, description, parameters |
| `Tool` | Wrapper: type="function" + ToolFunction |
| `Message` | Message with id, type, role, content list |
| `ResponseItem` | Output item: id, type, role, content list |
| `OutputText` | Text content: type="output_text", text |
| `TextFormat` | Format spec: type="text" |

### Streaming Event Models

| Model | Event Type | Key Fields |
| --- | --- | --- |
| `ResponseCreated` | `response.created` | `response` (ResponseModel) |
| `ResponseInProgress` | `response.in_progress` | `response` (ResponseModel) |
| `ResponseCompleted` | `response.completed` | `response` (ResponseModel) |
| `OutputTextDelta` | `response.output_text.delta` | `item_id`, `output_index`, `delta` |
| `ToolCallsCreated` | `response.tool_calls.created` | `item_id`, `output_index`, `tool_call` |
| `ToolCallArgumentsDelta` | `response.function_call_arguments.delta` | `item_id`, `output_index`, `delta` |
| `ToolCallArgumentsDone` | `response.function_call_arguments.done` | `id`, `output_index`, `arguments` |
