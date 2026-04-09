import json
import asyncio
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from open_responses_server.common.config import logger, HEARTBEAT_INTERVAL, STREAM_TIMEOUT
from open_responses_server.common.llm_client import (
    startup_llm_client,
    shutdown_llm_client,
    LLMClient,
)
from open_responses_server.common.mcp_manager import mcp_manager
from open_responses_server.responses_service import convert_responses_to_chat_completions, process_chat_completions_stream
from open_responses_server.chat_completions_service import handle_chat_completions

_STREAM_DONE = object()
_STREAM_ERROR = object()
_HEARTBEAT_EVENT = 'event: response.heartbeat\ndata: {"type":"response.heartbeat"}\n\n'


async def _stream_with_keepalive(async_iter_factory, interval, keepalive_event=_HEARTBEAT_EVENT):
    """Yield keepalive comments while a producer waits on upstream stream activity.

    Starts the client-facing SSE stream immediately and runs the upstream work
    in a background task, so heartbeats continue even before the backend stream
    context manager yields control. The heartbeat is a real SSE data event so
    clients like Codex reset idle timers after parsing it.
    """
    if not interval or interval <= 0:
        interval = 1.0

    queue = asyncio.Queue()

    async def producer():
        try:
            async for item in async_iter_factory():
                await queue.put(item)
        except Exception as exc:
            await queue.put((_STREAM_ERROR, exc))
        finally:
            await queue.put(_STREAM_DONE)

    task = asyncio.create_task(producer())
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=interval)
            except asyncio.TimeoutError:
                logger.debug("[STREAM-HEARTBEAT] Sending SSE keepalive")
                yield keepalive_event
                continue

            if item is _STREAM_DONE:
                break

            if isinstance(item, tuple) and len(item) == 2 and item[0] is _STREAM_ERROR:
                raise item[1]

            yield item
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


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


# API endpoints
@app.post("/responses")
async def create_response(request: Request):
    """
    Create a response in Responses API format, translating to/from chat.completions API.
    """
    try:
        logger.info("Received request to /responses")
        request_data = await request.json()
        
        # Log basic request information
        logger.info(f"Received request: model={request_data.get('model')}, stream={request_data.get('stream')}")
        
        # Log input content for better visibility
        if "input" in request_data and request_data["input"]:
            logger.info("==== REQUEST CONTENT ====")
            #     "input": [{"role": "user", "content": [{"type": "input_text", "text": "save a file with \"demo2\" text called \"demo2.md\""}], "type": "message"}],
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
        
        # Inject cached MCP tools into request_data before conversion so conversion sees them
        if mcp_manager.mcp_functions_cache:
            # Get existing tools from request_data or initialize empty list
            existing_tools = request_data.get("tools", [])
            
            # Create tools format for MCP functions
            mcp_tools = [
                {"type": "function", "name": f["name"], "description": f.get("description"), "parameters": f.get("parameters", {})}
                for f in mcp_manager.mcp_functions_cache
            ]
            
            # Get the names of existing tools to avoid duplicates
            existing_tool_names = set(tool["name"] for tool in existing_tools if "name" in tool)
            
            # Only add MCP tools that don't conflict with existing tools
            filtered_mcp_tools = [
                tool for tool in mcp_tools 
                if tool["name"] not in existing_tool_names
            ]
            
            # Append filtered MCP tools to existing tools, keeping existing tools first (priority)
            request_data["tools"] = existing_tools + filtered_mcp_tools
            
            logger.info(f"[TOOL-INJECT] /responses: {len(existing_tools)} existing tools, {len(mcp_manager.mcp_functions_cache)} MCP tools available")
            logger.info(f"[TOOL-INJECT] /responses: existing tool names: {list(existing_tool_names)}")
            logger.info(f"[TOOL-INJECT] /responses: available MCP tools: {[t['name'] for t in mcp_tools]}")
            logger.info(f"[TOOL-INJECT] /responses: filtered {len(filtered_mcp_tools)} MCP tools to inject: {[t['name'] for t in filtered_mcp_tools]}")
            logger.info(f"[TOOL-INJECT] /responses: final tool count: {len(request_data['tools'])}")
        else:
            logger.info("[TOOL-INJECT] /responses: no MCP tools available in cache")
        
        # Convert request to chat.completions format
        chat_request = convert_responses_to_chat_completions(request_data)
        
        # Inject cached MCP tool definitions
        if mcp_manager.mcp_functions_cache:
            # Keep any existing functions and merge with MCP functions
            existing_functions = chat_request.get("functions", [])
            
            # Convert to the "tools" format which is more broadly supported
            if "tools" not in chat_request:
                chat_request["tools"] = []
                
            # Get existing tool names to avoid duplicates and ensure priority
            existing_tool_names = set()
            for tool in chat_request["tools"]:
                if isinstance(tool, dict) and "function" in tool and "name" in tool["function"]:
                    existing_tool_names.add(tool["function"]["name"])
                elif isinstance(tool, dict) and "name" in tool:
                    existing_tool_names.add(tool["name"])
            
            # First convert existing functions to tools format
            for func in existing_functions:
                if func.get("name") not in existing_tool_names:
                    chat_request["tools"].append({
                        "type": "function",
                        "function": func
                    })
                    existing_tool_names.add(func.get("name", ""))
            
            # Then add MCP functions that don't conflict with existing tools
            mcp_tools_added = []
            for func in mcp_manager.mcp_functions_cache:
                if func.get("name") not in existing_tool_names:
                    chat_request["tools"].append({
                        "type": "function",
                        "function": func
                    })
                    mcp_tools_added.append(func.get("name"))
            
            # Remove the functions key as we've converted to tools format
            chat_request.pop("functions", None)
            
            logger.info(f"[TOOL-CONVERT] /responses: converted {len(existing_functions)} existing functions to tools format")
            logger.info(f"[TOOL-CONVERT] /responses: added {len(mcp_tools_added)} MCP tools: {mcp_tools_added}")
            logger.info(f"[TOOL-CONVERT] /responses: final chat_request tools count: {len(chat_request.get('tools', []))}")
        else:
            logger.info("[TOOL-CONVERT] /responses: no MCP functions cached, sending without MCP tools")
        
        # Remove tool_choice when no functions/tools are provided
        if not chat_request.get("functions") and not chat_request.get("tools"):
            chat_request.pop("tool_choice", None)
        # End MCP injection
        # Remove unsupported tool_choice parameter before sending
        chat_request.pop("tool_choice", None)

        # Check for streaming mode
        stream = request_data.get("stream", False)
        
        if stream:
            logger.info("Handling streaming response")
            # Handle streaming response
            async def stream_response():
                try:
                    async def llm_event_stream():
                        # Fetch available MCP tools and format as functions for chat.completions
                        mcp_functions = []
                        for server in mcp_manager.mcp_servers:
                            try:
                                for t in await server.list_tools():
                                    mcp_functions.append({
                                        "name": t["name"],
                                        "description": t.get("description"),
                                        "parameters": t.get("parameters", {}),
                                    })
                            except Exception as e:
                                logger.warning(f"Error listing tools from {server.name}: {e}")
                        # Only include functions if we have them
                        if mcp_functions:
                            # Convert to the "tools" format which is more broadly supported
                            existing_tools = chat_request.get("tools", [])
                            existing_functions = chat_request.get("functions", [])
                            
                            # Convert any existing functions to tools format
                            for func in existing_functions:
                                existing_tools.append({
                                    "type": "function",
                                    "function": func
                                })
                            
                            # Get the names of existing tools to avoid duplicates
                            existing_tool_names = set()
                            for tool in existing_tools:
                                if isinstance(tool, dict) and "function" in tool and "name" in tool["function"]:
                                    existing_tool_names.add(tool["function"]["name"])
                                elif isinstance(tool, dict) and "name" in tool:
                                    existing_tool_names.add(tool["name"])
                            
                            # Only add MCP functions that don't conflict with existing tools
                            for func in mcp_functions:
                                if func["name"] not in existing_tool_names:
                                    existing_tools.append({
                                        "type": "function",
                                        "function": func
                                    })
                                
                            # Set the tools and remove functions
                            chat_request["tools"] = existing_tools
                            chat_request.pop("functions", None)
                            
                            logger.info(f"Converted {len(existing_functions)} existing functions and {len(mcp_functions)} MCP functions to tools format")
                        elif "functions" in chat_request:
                            # Convert any existing functions to tools format
                            existing_tools = chat_request.get("tools", [])
                            existing_functions = chat_request.get("functions", [])
                            
                            if existing_functions:
                                # Convert functions to tools format
                                for func in existing_functions:
                                    existing_tools.append({
                                        "type": "function",
                                        "function": func
                                    })
                                    
                                chat_request["tools"] = existing_tools
                                logger.info(f"Converted {len(existing_functions)} existing functions to tools format")
                            
                            # Remove the functions key regardless
                            chat_request.pop("functions", None)
                            
                            if not chat_request.get("tools"):
                                # If we don't have any tools either, remove that key
                                chat_request.pop("tools", None)
                                logger.info("No tools or functions available, sending without them")
                        # Log the initial Chat Completions request payload
                        logger.info(f"Sending Chat Completions request: {json.dumps(chat_request)}")
                        client = await LLMClient.get_client()
                        async with client.stream(
                            "POST",
                            "/v1/chat/completions",
                            json=chat_request,
                            timeout=STREAM_TIMEOUT
                        ) as response:
                            logger.info(f"Stream request status: {response.status_code}")
                            
                            if response.status_code != 200:
                                error_content = await response.aread()
                                logger.error(f"Error from LLM API: {error_content}")
                                yield f"data: {json.dumps({'type': 'error', 'error': {'message': f'Error from LLM API: {response.status_code}'}})}\n\n"
                                return
                            
                            async for event in process_chat_completions_stream(response, chat_request):
                                yield event

                    async for event in _stream_with_keepalive(
                        llm_event_stream,
                        HEARTBEAT_INTERVAL
                    ):
                        yield event
                except Exception as e:
                    logger.error(f"Error in stream_response: {str(e)}")
                    yield f"data: {json.dumps({'type': 'error', 'error': {'message': str(e)}})}\n\n"
            
            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream"
            )
        
        else:
            logger.info("Non-streaming response unsupported")
            raise HTTPException(
                status_code=501,
                detail="Non-streaming responses are not supported on /responses. Set stream=True."
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in create_response: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


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
                async with client.stream(
                    request.method,
                    url,
                    headers=headers,
                    content=body,
                    timeout=STREAM_TIMEOUT
                ) as response:
                    async for chunk in response.aiter_bytes():
                        yield chunk
            return StreamingResponse(stream_proxy(), media_type=request.headers.get('accept', 'application/json'))
        else:
            response = await client.request(
                request.method,
                url,
                headers=headers,
                content=body,
                timeout=STREAM_TIMEOUT
            )
            return Response(content=response.content, status_code=response.status_code, headers=response.headers)
            
    except Exception as e:
        logger.error(f"Error in proxy endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error proxying request: {str(e)}") 
