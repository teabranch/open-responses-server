import json
from fastapi import Request
from fastapi.responses import StreamingResponse
from open_responses_server.common.llm_client import LLMClient
from open_responses_server.common.config import logger, OPENAI_BASE_URL_INTERNAL, OPENAI_API_KEY, logger
from open_responses_server.common.mcp_manager import mcp_manager

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
        # Placeholder for the two-stage MCP streaming logic from the original plan.
        # For now, we will proxy the streaming request directly.
        
        async def stream_proxy():
            try:
                async with client.stream(
                    "POST",
                    "/v1/chat/completions",
                    json=request_data,
                    timeout=120.0
                ) as response:
                    async for chunk in response.aiter_bytes():
                        yield chunk
            except Exception as e:
                logger.error(f"Error during chat completions stream proxy: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n".encode()

        return StreamingResponse(stream_proxy(), media_type="text/event-stream")

    else:
        # Handle non-streaming request
        try:
            logger.info("making request:")
            logger.info(request_data)
            # debug client
            response = await client.post(
                "/v1/chat/completions",
                json=request_data,
                timeout=120.0
            )
            logger.info("response:")
            logger.info(response.text)
            return response
        except Exception as e:
            logger.error(f"Error during chat completions non-stream proxy: {e}")
            return {"error": str(e)}, 500 