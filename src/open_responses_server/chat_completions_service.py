import json
from fastapi import Request
from fastapi.responses import StreamingResponse, Response, JSONResponse
from open_responses_server.common.llm_client import LLMClient
from open_responses_server.common.config import logger, OPENAI_BASE_URL_INTERNAL, OPENAI_API_KEY, MAX_TOOL_CALL_ITERATIONS
from open_responses_server.common.mcp_manager import mcp_manager

def serialize_tool_result(result):
    if hasattr(result, 'content') and isinstance(result.content, list):
        content_list = [content.text for content in result.content if hasattr(content, 'text')]
        tool_content = json.dumps(content_list)
    else:
        tool_content = json.dumps(result)
    return tool_content

async def _handle_non_streaming_request(client: LLMClient, request_data: dict):
    """Handles a non-streaming chat completions request with potential tool calls."""
    messages = list(request_data.get("messages", []))
    current_request_data = request_data.copy()
    
    for _ in range(MAX_TOOL_CALL_ITERATIONS):
        current_request_data["messages"] = messages
        current_request_data.pop("stream", None)

        try:
            response = await client.post(
                "/v1/chat/completions",
                json=current_request_data,
                timeout=120.0
            )
            response.raise_for_status()
            response_data = response.json()
            
            choice = response_data.get("choices", [])[0]
            message = choice.get("message", {})
            messages.append(message)

            if choice.get("finish_reason") == "tool_calls":
                tool_calls = message.get("tool_calls", [])
                tool_results_messages = []
                for tool_call in tool_calls:
                    function_call = tool_call.get("function", {})
                    tool_name = function_call.get("name")
                    tool_call_id = tool_call.get("id")

                    if mcp_manager.is_mcp_tool(tool_name):
                        try:
                            arguments = json.loads(function_call.get("arguments", "{}"))
                            result = await mcp_manager.execute_mcp_tool(tool_name, arguments)
                            logger.info(f"Executed tool {tool_name} with result: {result}")
                            tool_content = serialize_tool_result(result)
                        except Exception as e:
                            logger.error(f"Error executing tool {tool_name}: {e}")
                            tool_content = json.dumps({"error": f"Error executing tool: {e}"})
                    else:
                        logger.warning(f"Tool '{tool_name}' is not a registered MCP tool.")
                        tool_content = json.dumps({"error": f"Tool '{tool_name}' is not a registered MCP tool."})

                    tool_results_messages.append({
                        "tool_call_id": tool_call_id,
                        "role": "tool",
                        "content": tool_content,
                    })
                messages.extend(tool_results_messages)
                continue
            else:
                # Log the response for debugging
                logger.info(f"Final response data: {response_data}")
                
                # Return the response data directly - let FastAPI handle JSON serialization
                return response_data

        except Exception as e:
            logger.error(f"Error during non-streaming chat completion: {e}")
            return {"error": str(e)}
    
    return {"error": "Max tool call iterations reached"}


async def _handle_streaming_request(client: LLMClient, request_data: dict) -> StreamingResponse:
    """Handles a streaming chat completions request with potential tool calls."""
    messages = list(request_data.get("messages", []))
    non_stream_request_data = request_data.copy()
    non_stream_request_data["stream"] = False

    for _ in range(MAX_TOOL_CALL_ITERATIONS):
        try:
            # Make a non-streaming request first to check for tool calls
            response = await client.post("/v1/chat/completions", json={**non_stream_request_data, "messages": messages}, timeout=120.0)
            response.raise_for_status()
            response_data = response.json()
            
            choice = response_data["choices"][0]
            message = choice["message"]

            if choice.get("finish_reason") == "tool_calls":
                messages.append(message)
                tool_calls = message.get("tool_calls", [])
                tool_results_messages = []
                for tool_call in tool_calls:
                    function_call = tool_call.get("function", {})
                    tool_name = function_call.get("name")
                    tool_call_id = tool_call.get("id")

                    if mcp_manager.is_mcp_tool(tool_name):
                        try:
                            arguments = json.loads(function_call.get("arguments", "{}"))
                            result = await mcp_manager.execute_mcp_tool(tool_name, arguments)
                            logger.info(f"Executed tool {tool_name} with result: {result}")
                            #result is serlized as: "meta=None content=[TextContent(type='text', text="[{'name': 'listings'}]", annotations=None)] isError=False"
                            # so we need to convert it to json
                            tool_content = serialize_tool_result(result)
                        except Exception as e:
                            logger.error(f"Error executing tool {tool_name}: {e}")
                            tool_content = json.dumps({"error": f"Error executing tool: {e}"})
                    else:
                        logger.warning(f"Tool '{tool_name}' is not a registered MCP tool.")
                        tool_content = json.dumps({"error": f"Tool '{tool_name}' is not a registered MCP tool."})

                    tool_results_messages.append({
                        "tool_call_id": tool_call_id,
                        "role": "tool",
                        "content": tool_content,
                    })
                messages.extend(tool_results_messages)
                continue
            else:
                # No tool calls, so we can stream the final response
                messages.append(message)
                stream_request_data = request_data.copy()
                stream_request_data["messages"] = messages
                stream_request_data["stream"] = True

                async def stream_proxy():
                    try:
                        async with client.stream(
                            "POST",
                            "/v1/chat/completions",
                            json=stream_request_data,
                            timeout=120.0
                        ) as stream_response:
                            async for chunk in stream_response.aiter_bytes():
                                yield chunk
                    except Exception as e:
                        logger.error(f"Error during chat completions stream proxy: {e}")
                        yield f"data: {json.dumps({'error': str(e)})}\n\n".encode()

                return StreamingResponse(stream_proxy(), media_type="text/event-stream")

        except Exception as e:
            logger.error(f"Error during streaming chat completion: {e}")
            async def error_stream():
                yield f"data: {json.dumps({'error': str(e)})}\n\n".encode()
            return StreamingResponse(error_stream(), media_type="text/event-stream", status_code=500)
    
    async def final_error_stream():
        yield f"data: {json.dumps({'error': 'Max tool call iterations reached'})}\n\n".encode()
    return StreamingResponse(final_error_stream(), media_type="text/event-stream", status_code=500)


async def handle_chat_completions(request: Request):
    """
    Handles requests to the /v1/chat/completions endpoint.
    Injects MCP tools and proxies the request to the underlying LLM API.
    """
    client = await LLMClient.get_client()
    request_data = await request.json()

    # Inject MCP tools into the request
    mcp_tools = mcp_manager.get_mcp_tools()
    if mcp_tools:
        existing_tools = request_data.get("tools", [])
        existing_tool_names = {tool.get("function", {}).get("name") for tool in existing_tools}
        
        for tool in mcp_tools:
            if tool.get("name") not in existing_tool_names:
                existing_tools.append({"type": "function", "function": tool})
        
        request_data["tools"] = existing_tools
    
    logger.info(request_data["tools"])

    # Determine if the request is streaming
    is_stream = request_data.get("stream", False)

    if is_stream:
        return await _handle_streaming_request(client, request_data)
    else:
        return await _handle_non_streaming_request(client, request_data) 