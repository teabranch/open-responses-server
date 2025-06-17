import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from open_responses_server.common.config import logger
from open_responses_server.common.llm_client import startup_llm_client, shutdown_llm_client, LLMClient
from open_responses_server.common.mcp_manager import mcp_manager
from open_responses_server.responses_service import convert_responses_to_chat_completions, process_chat_completions_stream
from open_responses_server.chat_completions_service import handle_chat_completions

app = FastAPI(
    title="Open Responses Server",
    description="A proxy server that converts between different OpenAI-compatible API formats.",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Application startup event handler."""
    await startup_llm_client()
    await mcp_manager.startup_mcp_servers()
    logger.info("API Controller startup complete.")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event handler."""
    await shutdown_llm_client()
    await mcp_manager.shutdown_mcp_servers()
    logger.info("API Controller shutdown complete.")

@app.post("/responses")
async def create_response(request: Request):
    """
    Endpoint for the custom /responses API.
    Converts the request, calls the chat completions endpoint, and streams the converted response.
    """
    try:
        request_data = await request.json()
        
        # Log basic request information
        logger.info(f"Received request: model={request_data.get('model')}, stream={request_data.get('stream')}")
        
        # Log input content for better visibility
        if "input" in request_data and request_data["input"]:
            logger.info("==== REQUEST CONTENT ====")
            for i, item in enumerate(request_data["input"]):
                if isinstance(item, dict):
                    if item.get("type") == "message" and item.get("role") == "user":
                        if "content" in item and isinstance(item["content"], list):
                            for index, content_item in enumerate(item["content"]):
                                if isinstance(content_item, dict):
                                    # Handle nested content structure like {"type": "input_text", "text": "actual message"}
                                    if content_item.get("type") == "input_text" and "text" in content_item:
                                        user_text = content_item.get("text", "")
                                        logger.info(f"USER INPUT: {user_text}")
                                    elif content_item.get("type") == "text" and "text" in content_item:
                                        user_text = content_item.get("text", "")
                                        logger.info(f"USER INPUT: {user_text}")
                                    # Handle other content types
                                    elif "type" in content_item:
                                        logger.info(f"USER INPUT ({content_item.get('type')}): {str(content_item)[:100]}...")
                                elif isinstance(content_item, str):
                                    logger.info(f"USER INPUT: {content_item}")
                    elif item.get("type") == "function_call_output":
                        logger.info(f"FUNCTION RESULT: call_id={item.get('call_id')}, output={str(item.get('output', ''))[:100]}...")
                elif isinstance(item, str):
                    logger.info(f"USER INPUT: {item}")
            logger.info("=======================")

        # Inject MCP tools into the request before conversion
        mcp_tools = mcp_manager.get_mcp_tools()
        if mcp_tools:
            # Start with user-provided tools, or an empty list
            final_tools = request_data.get("tools", [])
            
            # Get the names of the tools already in the list
            final_tool_names = {
                tool.get("function", {}).get("name") if tool.get("function") else tool.get("name")
                for tool in final_tools
                if (tool.get("function") and tool.get("function").get("name")) or tool.get("name")
            }
            
            # Add only the new MCP tools that don't conflict
            for tool in mcp_tools:
                if tool.get("name") not in final_tool_names:
                    final_tools.append({"type": "function", "function": tool})
            
            request_data["tools"] = final_tools
            logger.info(f"Injected {len(mcp_tools)} MCP tools into request")

        chat_request = convert_responses_to_chat_completions(request_data)
        
        client = await LLMClient.get_client()
        
        async def stream_response():
            try:
                async with client.stream("POST", "/v1/chat/completions", json=chat_request, timeout=120.0) as response:
                    if response.status_code != 200:
                        error_content = await response.aread()
                        logger.error(f"Error from LLM API: {error_content.decode()}")
                        yield f"data: {json.dumps({'error': 'LLM API Error'})}\n\n"
                        return
                    
                    async for event in process_chat_completions_stream(response, chat_request):
                        yield event
            except Exception as e:
                logger.error(f"Error in /responses stream: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return StreamingResponse(stream_response(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Error in create_response endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    Endpoint for /v1/chat/completions, delegating to the service.
    """
    logger.info("Handling chat completions")
    response = await handle_chat_completions(request)
    logger.info("Chat completions handled")
    if isinstance(response, StreamingResponse):
        return response
    elif isinstance(response, Response):
        return response
    return response


@app.get("/health")
async def health_check():
    return {"status": "ok", "adapter": "running"}

@app.get("/")
async def root():
    return {"message": "Open Responses Server is running."}

@app.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH", "TRACE"])
async def proxy_endpoint(request: Request, path_name: str):
    """
    A generic proxy for any other endpoints, forwarding them to the LLM backend.
    """
    client = await LLMClient.get_client()
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() != 'host'}

    try:
        url = f"{client.base_url}/v1/{path_name}"
        
        # Handle streaming for the proxy
        is_stream = False
        if body:
            try:
                is_stream = json.loads(body).get("stream", False)
            except json.JSONDecodeError:
                pass

        if is_stream:
            async def stream_proxy():
                async with client.stream(request.method, url, headers=headers, content=body, timeout=120.0) as response:
                    async for chunk in response.aiter_bytes():
                        yield chunk
            return StreamingResponse(stream_proxy(), media_type=response.headers.get("content-type"))
        else:
            response = await client.request(request.method, url, headers=headers, content=body, timeout=120.0)
            return Response(content=response.content, status_code=response.status_code, headers=response.headers)
            
    except Exception as e:
        logger.error(f"Error in proxy endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error proxying request: {str(e)}") 