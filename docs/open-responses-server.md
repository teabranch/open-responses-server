# Open Responses Server - Architecture Overview

## Introduction

The Open Responses Server is a proxy adapter that translates between different OpenAI-compatible API formats. Specifically, it provides translation between the Responses API format and the chat.completions API format. This enables applications designed for one API format to work with services that support a different format.

## High-Level Components

The server is built with FastAPI and includes these main components:

1. **API Adapter** - The core translation layer between API formats
2. **Request Handling** - Processing incoming requests to the `/responses` endpoint
3. **Response Streaming** - Converting streaming responses between formats
4. **Proxy Support** - Forwarding other requests directly to the backend LLM service

## Configuration

The server reads configuration from environment variables:
- `OPENAI_BASE_URL_INTERNAL` - The internal URL of the LLM API service (default: http://localhost:8000)
- `OPENAI_BASE_URL` - The external URL for clients (default: http://localhost:8080)
- `OPENAI_API_KEY` - API key for authentication (default: "dummy-key")
- `API_ADAPTER_HOST` - Host address to bind the server (default: 0.0.0.0)
- `API_ADAPTER_PORT` - Port to run the server on (default: 8080)

## Data Models

The server uses Pydantic models to define the structure of:
- Tool functions and tool definitions
- Messages and their content
- Response objects and events
- Request parameters

## Core Functionality

### Request Translation: Responses API → Chat Completions

The `convert_responses_to_chat_completions` function transforms requests from the Responses API format to the chat completions format:

1. It copies basic parameters like model name, temperature, and top_p
2. Converts instructions to system messages
3. Processes user messages from the input array
4. Transforms tool function definitions
5. Handles tool responses and maintains conversation context

### Response Translation: Chat Completions → Responses API

When handling streaming responses from the chat completions API, the server:

1. Creates an initial response object with a unique ID
2. Tracks the state of tool calls being built incrementally
3. Emits events like:
   - `response.created` - The initial response creation
   - `response.in_progress` - Updates during processing
   - `response.output_text.delta` - Text generation updates
   - `response.function_call_arguments.delta` - Tool call argument updates
   - `response.function_call_arguments.done` - Completed tool calls
   - `response.completed` - Final response completion

## Tool and Function Call Handling

The server has specialized handling for tool function calls:

### Incoming Tool Definitions

When a client submits a request with tools:
1. Each tool definition is converted to the chat completions format
2. The `function` property with name, description, and parameters is preserved
3. The `tool_choice` parameter is maintained when present

### Outgoing Function Calls

When the LLM responds with a function call:
1. The server tracks each tool call in a dictionary by index
2. As function name and arguments stream in, they're buffered and converted to the Responses format
3. Tool calls are emitted as events with unique IDs
4. Arguments are streamed as delta events to provide real-time feedback
5. When complete, a "done" event is emitted

For example, a single function call may generate:
1. A tool call creation event
2. Multiple argument delta events as the arguments stream in
3. A final "done" event when the arguments are complete

## Request Flow

1. Client sends a request to `/responses`
2. The request is validated and logged
3. Request is converted from Responses format to chat.completions format
4. The converted request is forwarded to the LLM API
5. If streaming, responses are processed chunk by chunk and converted back
6. Events are emitted in the Responses API format
7. The final response is sent back to the client

## Proxy Capabilities

For requests not targeting the `/responses` endpoint, the server acts as a transparent proxy:
1. It forwards the request to the actual LLM API
2. Preserves headers (except 'host')
3. Adds authentication if needed
4. Returns the response directly

## Error Handling

The server includes error handling at multiple levels:
1. JSON parsing errors during stream processing
2. API errors from the LLM service
3. General exceptions during request processing

## Key Functions

1. `convert_responses_to_chat_completions` - Translates request formats
2. `process_chat_completions_stream` - Processes streaming responses
3. `create_response` - Main endpoint handler for the `/responses` route
4. `proxy_endpoint` - Catch-all handler for other routes

## Future Development Considerations

When refactoring this server, consider:

1. **Separation of concerns** - Extract the conversion logic into separate modules
2. **Configuration management** - Move configuration to a dedicated module
3. **Error handling** - Enhance error reporting and recovery mechanisms
4. **Testing** - Add more comprehensive tests for edge cases
5. **Logging** - Optimize logging to focus on important information without excessive output

By understanding these components, you'll be well-prepared to refactor the server while maintaining its core functionality of translating between API formats.
