import json
from fastapi import Request
from fastapi.responses import StreamingResponse, Response, JSONResponse
from open_responses_server.common.llm_client import LLMClient
from open_responses_server.common.config import logger, OPENAI_BASE_URL_INTERNAL, OPENAI_API_KEY, MAX_TOOL_CALL_ITERATIONS
from open_responses_server.common.mcp_manager import mcp_manager, serialize_tool_result

async def _handle_non_streaming_request(client: LLMClient, request_data: dict):
    """Handles a non-streaming chat completions request with potential tool calls."""
    messages = list(request_data.get("messages", []))
    current_request_data = request_data.copy()
    
    # Remove reasoning parameter if it has null values
    if "reasoning" in current_request_data:
        reasoning = current_request_data["reasoning"]
        if isinstance(reasoning, dict) and all(v is None for v in reasoning.values()):
            current_request_data.pop("reasoning", None)
            logger.info("[CHAT-COMPLETIONS-NON-STREAM] Removed reasoning parameter with null values")
    
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
                
                logger.info(f"[CHAT-COMPLETIONS-NON-STREAM] Processing {len(tool_calls)} tool calls")
                
                for tool_call in tool_calls:
                    function_call = tool_call.get("function", {})
                    tool_name = function_call.get("name")
                    tool_call_id = tool_call.get("id")

                    logger.info(f"[CHAT-COMPLETIONS-NON-STREAM] Processing tool call: {tool_name} (id: {tool_call_id})")

                    if mcp_manager.is_mcp_tool(tool_name):
                        logger.info(f"[CHAT-COMPLETIONS-NON-STREAM] Executing MCP tool: {tool_name}")
                        try:
                            arguments = json.loads(function_call.get("arguments", "{}"))
                            logger.debug(f"[CHAT-COMPLETIONS-NON-STREAM] Tool arguments: {arguments}")
                            result = await mcp_manager.execute_mcp_tool(tool_name, arguments)
                            logger.info(f"[CHAT-COMPLETIONS-NON-STREAM] ✓ Tool {tool_name} executed successfully")
                            logger.debug(f"[CHAT-COMPLETIONS-NON-STREAM] Tool result: {result}")
                            tool_content = serialize_tool_result(result)
                        except Exception as e:
                            logger.error(f"[CHAT-COMPLETIONS-NON-STREAM] ✗ Error executing tool {tool_name}: {e}")
                            tool_content = json.dumps({"error": f"Error executing tool: {e}"})
                    else:
                        logger.warning(f"[CHAT-COMPLETIONS-NON-STREAM] Tool '{tool_name}' is not a registered MCP tool.")
                        tool_content = json.dumps({"error": f"Tool '{tool_name}' is not a registered MCP tool."})

                    tool_results_messages.append({
                        "tool_call_id": tool_call_id,
                        "role": "tool",
                        "content": tool_content,
                    })
                    
                logger.info(f"[CHAT-COMPLETIONS-NON-STREAM] Added {len(tool_results_messages)} tool result messages")
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

    # Remove reasoning parameter if it has null values
    if "reasoning" in non_stream_request_data:
        reasoning = non_stream_request_data["reasoning"]
        if isinstance(reasoning, dict) and all(v is None for v in reasoning.values()):
            non_stream_request_data.pop("reasoning", None)
            logger.info("[CHAT-COMPLETIONS-STREAM] Removed reasoning parameter with null values from non-stream request")

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
                
                logger.info(f"[CHAT-COMPLETIONS-STREAM] Processing {len(tool_calls)} tool calls in streaming mode")
                
                for tool_call in tool_calls:
                    function_call = tool_call.get("function", {})
                    tool_name = function_call.get("name")
                    tool_call_id = tool_call.get("id")

                    logger.info(f"[CHAT-COMPLETIONS-STREAM] Processing tool call: {tool_name} (id: {tool_call_id})")

                    if mcp_manager.is_mcp_tool(tool_name):
                        logger.info(f"[CHAT-COMPLETIONS-STREAM] Executing MCP tool: {tool_name}")
                        try:
                            arguments = json.loads(function_call.get("arguments", "{}"))
                            logger.debug(f"[CHAT-COMPLETIONS-STREAM] Tool arguments: {arguments}")
                            result = await mcp_manager.execute_mcp_tool(tool_name, arguments)
                            logger.info(f"[CHAT-COMPLETIONS-STREAM] ✓ Tool {tool_name} executed successfully")
                            logger.debug(f"[CHAT-COMPLETIONS-STREAM] Tool result: {result}")
                            #result is serlized as: "meta=None content=[TextContent(type='text', text="[{'name': 'listings'}]", annotations=None)] isError=False"
                            # so we need to convert it to json
                            tool_content = serialize_tool_result(result)
                        except Exception as e:
                            logger.error(f"[CHAT-COMPLETIONS-STREAM] ✗ Error executing tool {tool_name}: {e}")
                            tool_content = json.dumps({"error": f"Error executing tool: {e}"})
                    else:
                        logger.warning(f"[CHAT-COMPLETIONS-STREAM] Tool '{tool_name}' is not a registered MCP tool.")
                        tool_content = json.dumps({"error": f"Tool '{tool_name}' is not a registered MCP tool."})

                    tool_results_messages.append({
                        "tool_call_id": tool_call_id,
                        "role": "tool",
                        "content": tool_content,
                    })
                    
                logger.info(f"[CHAT-COMPLETIONS-STREAM] Added {len(tool_results_messages)} tool result messages")
                messages.extend(tool_results_messages)
                continue
            else:
                # No tool calls, so we can stream the final response
                messages.append(message)
                stream_request_data = request_data.copy()
                stream_request_data["messages"] = messages
                stream_request_data["stream"] = True

                # Remove reasoning parameter if it has null values
                if "reasoning" in stream_request_data:
                    reasoning = stream_request_data["reasoning"]
                    if isinstance(reasoning, dict) and all(v is None for v in reasoning.values()):
                        stream_request_data.pop("reasoning", None)
                        logger.info("[CHAT-COMPLETIONS-STREAM] Removed reasoning parameter with null values from final stream request")

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

    logger.info("[CHAT-COMPLETIONS] Processing /v1/chat/completions request")
    
    # Inject MCP tools into the request
    mcp_tools = mcp_manager.get_mcp_tools()
    if mcp_tools:
        existing_tools = request_data.get("tools", [])
        existing_tool_names = {tool.get("function", {}).get("name") for tool in existing_tools}
        
        logger.info(f"[CHAT-COMPLETIONS] Found {len(mcp_tools)} MCP tools available")
        logger.info(f"[CHAT-COMPLETIONS] Request has {len(existing_tools)} existing tools: {list(existing_tool_names)}")
        
        added_tools = []
        for tool in mcp_tools:
            if tool.get("name") not in existing_tool_names:
                existing_tools.append({"type": "function", "function": tool})
                added_tools.append(tool.get("name"))
        
        request_data["tools"] = existing_tools
        
        logger.info(f"[CHAT-COMPLETIONS] Added {len(added_tools)} MCP tools: {added_tools}")
        logger.info(f"[CHAT-COMPLETIONS] Final tool count: {len(existing_tools)}")
    else:
        logger.info("[CHAT-COMPLETIONS] No MCP tools available to inject")
    
    logger.debug(f"[CHAT-COMPLETIONS] Final tools in request: {request_data.get('tools', [])}")

    # Determine if the request is streaming
    is_stream = request_data.get("stream", False)
    logger.info(f"[CHAT-COMPLETIONS] Request streaming mode: {is_stream}")

    if is_stream:
        return await _handle_streaming_request(client, request_data)
    else:
        return await _handle_non_streaming_request(client, request_data) 