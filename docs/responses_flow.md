---
title: API Flow Diagrams
nav_order: 3
---

## API Flow Diagrams

Sequence diagrams for the two main API endpoints. For detailed event type
documentation, see [Events & Tool Handling](events-and-tool-handling.md).

## Plain Text /responses Flow (No Tools)

The simplest path — user sends a prompt, LLM responds with text.

```mermaid
sequenceDiagram
    title /responses — Plain Text (No Tools)

    participant Client
    participant AC as api_controller
    participant RS as responses_service
    participant LLM

    Client->>+AC: POST /responses (stream=true, prompt)
    AC->>+RS: convert_responses_to_chat_completions(request)
    RS-->>-AC: chat_request

    AC->>+LLM: POST /v1/chat/completions (streaming)
    LLM-->>-AC: HTTP 200 (SSE stream begins)

    AC->>+RS: process_chat_completions_stream(response_stream)
    RS-->>Client: response.created (ResponseCreated)
    RS-->>Client: response.in_progress (ResponseInProgress)

    loop LLM streams text
        LLM-->>RS: chunk: delta.content
        RS-->>Client: response.output_text.delta (OutputTextDelta)
    end

    LLM-->>RS: chunk: finish_reason="stop"
    RS->>RS: Save conversation history
    RS-->>-Client: response.completed (ResponseCompleted)
```

## /responses Flow with MCP Tool Execution

Full flow when the LLM invokes a tool registered with the MCP manager.

```mermaid
sequenceDiagram
    title /responses — MCP Tool Execution

    participant Client
    participant AC as api_controller
    participant RS as responses_service
    participant LLM
    participant MCPM as mcp_manager
    participant MCPS as MCPServer (Subprocess)

    Client->>+AC: POST /responses (stream=true, prompt, tools)

    rect rgb(230, 240, 255)
        note over AC, RS: 1. Request Preparation
        AC->>+MCPM: get_mcp_tools()
        MCPM-->>-AC: Returns cached MCP tools
        AC->>AC: Injects MCP tools into request's 'tools' list

        AC->>+RS: convert_responses_to_chat_completions(request)
        RS-->>-AC: Returns chat_request (OpenAI format)
    end

    rect rgb(230, 255, 230)
        note over AC, LLM: 2. Call LLM API
        AC->>+LLM: POST /v1/chat/completions (streaming)
        LLM-->>-AC: HTTP 200 OK (SSE stream begins)
    end

    rect rgb(255, 255, 224)
        note over RS, Client: 3. Stream Processing & Tool Call Generation
        AC->>+RS: process_chat_completions_stream(response_stream)
        RS-->>Client: response.created (ResponseCreated)
        RS-->>Client: response.in_progress (ResponseInProgress)

        loop LLM streams back tool call
            LLM-->>RS: chunk: tool_calls delta (name)
            RS->>RS: Detects new tool call
            RS-->>Client: response.tool_calls.created (ToolCallsCreated)
            RS-->>Client: response.in_progress (ResponseInProgress)

            LLM-->>RS: chunk: tool_calls delta (arguments fragment)
            RS-->>Client: response.function_call_arguments.delta (ToolCallArgumentsDelta)
        end

        LLM-->>RS: chunk: finish_reason="tool_calls"
    end

    rect rgb(255, 230, 224)
        note over RS, MCPS: 4. MCP Tool Execution
        RS->>RS: Loop through completed tool calls
        RS->>+MCPM: is_mcp_tool(tool_name)?
        MCPM-->>-RS: returns true

        RS-->>Client: response.function_call_arguments.done (ToolCallArgumentsDone)

        RS->>+MCPM: execute_mcp_tool(tool_name, args)
        MCPM->>+MCPS: session.call_tool(tool_name, args)
        MCPS-->>-MCPM: tool result
        MCPM-->>-RS: tool result
    end

    rect rgb(224, 230, 255)
        note over RS, Client: 5. Emitting MCP Result
        RS->>RS: serialize_tool_result(result)
        RS->>RS: Add 'function_call_output' to response output
        RS-->>Client: response.output_text.delta (OutputTextDelta — serialized result)
    end

    rect rgb(240, 240, 240)
        note over RS, Client: 6. Finalizing Response
        RS->>RS: Save conversation history
        RS-->>-Client: response.completed (ResponseCompleted)
    end
```

## /responses Flow with Non-MCP Tool (Client-Executed)

When the LLM calls a tool that is not registered with MCP, the server returns
it to the client for execution.

```mermaid
sequenceDiagram
    title /responses — Non-MCP Tool (Client Executes)

    participant Client
    participant RS as responses_service

    RS-->>Client: response.tool_calls.created (status="ready")
    RS-->>Client: response.function_call_arguments.delta (repeated)
    RS-->>Client: response.function_call_arguments.done
    RS-->>Client: response.completed (output contains function_call with status="ready")

    Note over Client: Client executes tool externally

    Client->>RS: POST /responses (input includes function_call_output with call_id)
    Note over RS: Matches call_id to history, adds tool result, continues conversation
```

## /v1/chat/completions Flow with MCP Tool Loop

The chat completions endpoint has its own tool-call loop. Streaming mode uses
non-streaming requests internally during tool resolution, then streams only the
final response.

```mermaid
sequenceDiagram
    title /v1/chat/completions — MCP Tool Loop

    participant Client
    participant CCS as chat_completions_service
    participant LLM
    participant MCPM as mcp_manager
    participant MCPS as MCPServer (Subprocess)

    Client->>+CCS: POST /v1/chat/completions (messages, tools)

    rect rgb(230, 240, 255)
        note over CCS, MCPM: 1. MCP Tool Injection
        CCS->>MCPM: get_mcp_tools()
        MCPM-->>CCS: cached MCP tools
        CCS->>CCS: Merge MCP tools with existing tools (de-duplicate)
    end

    rect rgb(230, 255, 230)
        note over CCS, LLM: 2. Tool Call Loop (up to MAX_TOOL_CALL_ITERATIONS)
        loop Until no tool_calls or max iterations
            CCS->>+LLM: POST /v1/chat/completions (non-streaming)
            LLM-->>-CCS: Response with finish_reason

            alt finish_reason == "tool_calls"
                CCS->>+MCPM: execute_mcp_tool(name, args) per tool
                MCPM->>+MCPS: session.call_tool(name, args)
                MCPS-->>-MCPM: tool result
                MCPM-->>-CCS: serialized result
                CCS->>CCS: Append tool results to messages, continue loop
            else finish_reason != "tool_calls"
                CCS->>CCS: Break loop
            end
        end
    end

    alt Streaming mode
        rect rgb(255, 255, 224)
            note over CCS, Client: 3. Stream Final Response
            CCS->>+LLM: POST /v1/chat/completions (stream=true, full message history)
            LLM-->>-Client: SSE stream (proxied directly)
        end
    else Non-streaming mode
        rect rgb(255, 255, 224)
            note over CCS, Client: 3. Return Final Response
            CCS-->>-Client: JSON response
        end
    end
```
