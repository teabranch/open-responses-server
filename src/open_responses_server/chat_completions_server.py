#!/usr/bin/env python3

"""
OpenAI Responses Server - A proxy server that converts between different OpenAI-compatible API formats.

This module provides a FastAPI server that acts as an adapter between different OpenAI API formats,
specifically translating between the Responses API format and the chat.completions API format.
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
import uuid
from fastapi import FastAPI, Request, Response, BackgroundTasks, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import httpx
from pydantic import BaseModel, Field
import time
import traceback # Added
from dotenv import load_dotenv
from contextlib import AsyncExitStack

# MCP support imports
from pathlib import Path
import shutil
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

__version__ = "0.1.0"
__author__ = "TeaBranch"
__license__ = "MIT"

# Load environment variables
load_dotenv()

# Configure logging with more focused format
logging.basicConfig(
    level=logging.INFO,  # Keep at INFO level for important logs only
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    # Both console and logfile at ./log folder

    handlers=[
        logging.FileHandler("./log/api_adapter.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("api_adapter")

# Configuration from environment variables
OPENAI_BASE_URL_INTERNAL = os.environ.get("OPENAI_BASE_URL_INTERNAL", "http://localhost:8000")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:8080")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "dummy-key")
API_ADAPTER_HOST = os.environ.get("API_ADAPTER_HOST", "0.0.0.0")
API_ADAPTER_PORT = int(os.environ.get("API_ADAPTER_PORT", "8080"))

logger.info(f"Configuration: OPENAI_BASE_URL_INTERNAL={OPENAI_BASE_URL_INTERNAL}, API_PORT={API_ADAPTER_PORT}")

app = FastAPI(title="API Adapter", description="Adapter for Responses API to chat.completions API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTTP client for making requests to the LLM API
http_client = httpx.AsyncClient(
    base_url=OPENAI_BASE_URL_INTERNAL,  # Fixed: using the actual variable
    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
    timeout=httpx.Timeout(120.0)  # Increased timeout
)

# Global list to hold initialized MCP servers
mcp_servers: List["MCPServer"] = []
# Cache for MCP functions, updated periodically
mcp_functions_cache: List[Dict[str, Any]] = []
# Refresh interval for updating MCP tools (in seconds)
MCP_TOOL_REFRESH_INTERVAL = int(os.environ.get("MCP_TOOL_REFRESH_INTERVAL", "10"))
# Global dictionary to store conversation history by response ID
conversation_history: Dict[str, List[Dict[str, Any]]] = {}
# Maximum number of conversations to keep in memory
MAX_CONVERSATION_HISTORY = int(os.environ.get("MAX_CONVERSATION_HISTORY", "100"))

# Helper to perform a standard proxy request
async def _default_proxy(request: Request, path_name: str, request_body_bytes: bytes):
    url = httpx.URL(path=f"/{path_name.lstrip('/')}", query=request.url.query.encode(\"utf-8\"))
    headers = dict(request.headers)
    headers.pop("host", None) # Let httpx handle the host header
    headers.pop("content-length", None) # Let httpx handle content-length

    logger.info(f"Default proxying {request.method} to {url}")

    rp_req = http_client.build_request(
        request.method, url, headers=headers, content=request_body_bytes, timeout=None
    )
    rp_resp = await http_client.send(rp_req, stream=True)

    return StreamingResponse(
        rp_resp.aiter_raw(),
        status_code=rp_resp.status_code,
        headers=dict(rp_resp.headers),
        media_type=rp_resp.headers.get("content-type"),
    )

async def _handle_chat_completions_streaming_mcp(
    original_request_data: dict,
    request_headers: dict
):
    """
    Handles streaming chat completions with MCP tool integration (Option B).
    Makes an initial LLM call, then if MCP tools are called, executes them,
    and makes a second LLM call with the tool results.
    """
    logger.info("Starting MCP-enhanced streaming for /v1/chat/completions")
    
    # 1. First LLM Call
    # ------------------
    request_data_llm1 = original_request_data.copy()
    # Ensure stream is true for the first call
    request_data_llm1["stream"] = True

    # Prepare headers for the internal request
    internal_headers = {k: v for k, v in request_headers.items() if k.lower() not in ["host", "content-length", "accept-encoding"]}
    internal_headers["Authorization"] = f"Bearer {OPENAI_API_KEY}" # Ensure API key is set

    logger.info(f"LLM1 (stream) request to /chat/completions: {json.dumps(request_data_llm1, indent=2)}")

    llm1_response_chunks = []
    accumulated_tool_calls = {}
    assistant_content_llm1 = ""
    llm1_had_tool_calls = False
    llm1_model_name = original_request_data.get("model", "unknown_model")


    try:
        async with http_client.stream(
            "POST",
            "/chat/completions",
            json=request_data_llm1,
            headers=internal_headers,
            timeout=None 
        ) as rp_resp_llm1:
            rp_resp_llm1.raise_for_status() # Raise HTTPStatusError for bad responses (4xx or 5xx)
            async for chunk_bytes in rp_resp_llm1.aiter_bytes():
                chunk_str = chunk_bytes.decode(\'utf-8\')
                # Yield non-tool-call related chunks immediately
                # We need to parse to see if it's a tool call or content
                
                for line in chunk_str.splitlines():
                    if not line.strip():
                        continue
                    if line.startswith("data: "):
                        line_data_str = line[6:]
                        if line_data_str == "[DONE]":
                            logger.info("LLM1 stream part 1 [DONE]")
                            # If [DONE] is received, it means LLM1 finished.
                            # If it had tool calls, they should have been processed.
                            # If not, this is the end of the stream.
                            if not llm1_had_tool_calls:
                                yield f"data: {line_data_str}\\n\\n"
                            break # Break from processing lines in this chunk
                        
                        try:
                            chunk_json = json.loads(line_data_str)
                            if chunk_json.get("model"):
                                llm1_model_name = chunk_json["model"]

                            choice = chunk_json.get("choices", [{}])[0]
                            delta = choice.get("delta", {})
                            finish_reason = choice.get("finish_reason")

                            # Accumulate content
                            if "content" in delta and delta["content"] is not None:
                                assistant_content_llm1 += delta["content"]
                                # Yield content chunks directly
                                yield f"{line}\\n\\n"
                                continue # Next line in chunk

                            # Accumulate tool calls
                            if "tool_calls" in delta and delta["tool_calls"]:
                                llm1_had_tool_calls = True
                                for tc_delta in delta["tool_calls"]:
                                    idx = tc_delta["index"]
                                    if idx not in accumulated_tool_calls:
                                        accumulated_tool_calls[idx] = {
                                            "id": tc_delta.get("id"),
                                            "type": tc_delta.get("type", "function"),
                                            "function": {"name": "", "arguments": ""}
                                        }
                                    if tc_delta.get("id"): # ID comes first
                                        accumulated_tool_calls[idx]["id"] = tc_delta["id"]
                                    if "function" in tc_delta:
                                        if "name" in tc_delta["function"] and tc_delta["function"]["name"]:
                                            accumulated_tool_calls[idx]["function"]["name"] = tc_delta["function"]["name"]
                                        if "arguments" in tc_delta["function"] and tc_delta["function"]["arguments"]:
                                            accumulated_tool_calls[idx]["function"]["arguments"] += tc_delta["function"]["arguments"]
                                # Yield the tool_call chunk itself as it's part of LLM1's response
                                yield f"{line}\\n\\n"
                                continue # Next line in chunk
                            
                            # If it's not content and not a tool_call delta, but has a finish_reason
                            if finish_reason:
                                if finish_reason == "tool_calls":
                                    llm1_had_tool_calls = True # Mark that tool calls were intended
                                    logger.info(f"LLM1 finish_reason: tool_calls. Accumulated: {accumulated_tool_calls}")
                                    # Yield the chunk that signals tool_calls finish_reason
                                    yield f"{line}\\n\\n"
                                    # Now we break from LLM1 stream to process tools
                                    break # Break from processing lines in this chunk
                                else: # e.g. "stop"
                                    logger.info(f"LLM1 finish_reason: {finish_reason}. No MCP tools to call, or LLM1 decided to stop.")
                                    yield f"{line}\\n\\n" # Yield the final chunk from LLM1
                                    # If it's a "stop" and there were no tool calls, we are done.
                                    if not llm1_had_tool_calls:
                                        return # End the generator

                            # If it's some other kind of chunk from LLM1, yield it
                            if not llm1_had_tool_calls: # Only if we are not expecting tool calls
                                yield f"{line}\\n\\n"

                        except json.JSONDecodeError:
                            logger.warning(f"Could not decode JSON from LLM1 stream: {line_data_str}")
                            yield f"{line}\\n\\n" # Yield as is
                
                if llm1_had_tool_calls and choice.get("finish_reason") == "tool_calls": # Check again if we broke due to tool_calls
                    break # Break from rp_resp_llm1.aiter_bytes()

            # After iterating through LLM1 stream
            if not llm1_had_tool_calls:
                logger.info("LLM1 stream finished without any tool calls to process.")
                # If [DONE] was not in the last chunk, ensure it's sent
                if not line.endswith("[DONE]"): # crude check, might need refinement
                     yield "data: [DONE]\\n\\n"
                return # End of streaming if no tool calls

    except httpx.HTTPStatusError as e:
        logger.error(f"LLM1 HTTP error: {e.response.status_code} - {e.response.text}")
        yield f"data: {json.dumps({'error': {'message': e.response.text, 'type': 'llm_request_failed', 'code': e.response.status_code}})}\\n\\n"
        yield "data: [DONE]\\n\\n"
        return
    except Exception as e:
        logger.error(f"Error during LLM1 streaming: {traceback.format_exc()}")
        yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'stream_error'}})}\\n\\n"
        yield "data: [DONE]\\n\\n"
        return

    logger.info(f"LLM1 processing complete. Accumulated tool calls: {json.dumps(accumulated_tool_calls, indent=2)}")

    # 2. Execute MCP Tools if any were called by LLM1
    # -----------------------------------------------
    if not llm1_had_tool_calls or not accumulated_tool_calls:
        logger.info("No MCP tool calls from LLM1, or tool calls were not for MCP. Stream ends.")
        # Ensure DONE is sent if not already
        # This path should ideally be covered by LLM1's [DONE] if it finished without tool_calls.
        # However, if it finished with tool_calls but none were MCP, we might need to send DONE.
        yield "data: [DONE]\\n\\n"
        return

    executed_mcp_tool_results = []
    messages_for_llm2 = original_request_data.get("messages", []).copy()

    # Add LLM1's response (assistant message with tool_calls) to messages for LLM2
    assistant_message_llm1 = {"role": "assistant", "content": assistant_content_llm1 if assistant_content_llm1 else None}
    assistant_message_llm1_tool_calls = []
    for idx, tc_data in accumulated_tool_calls.items():
        if tc_data.get("id") and tc_data.get("function",{}).get("name"): # Ensure valid tool call structure
             assistant_message_llm1_tool_calls.append({
                "id": tc_data["id"],
                "type": "function",
                "function": tc_data["function"]
            })
    if assistant_message_llm1_tool_calls:
        assistant_message_llm1["tool_calls"] = assistant_message_llm1_tool_calls
    
    messages_for_llm2.append(assistant_message_llm1)

    mcp_tools_were_called_and_executed = False
    for idx, tool_call_data in accumulated_tool_calls.items():
        tool_name = tool_call_data["function"]["name"]
        tool_id = tool_call_data["id"]
        
        if is_mcp_tool(tool_name):
            mcp_tools_were_called_and_executed = True
            logger.info(f"Executing MCP tool: {tool_name} with ID: {tool_id} and args: {tool_call_data['function']['arguments']}")
            try:
                tool_args_str = tool_call_data["function"]["arguments"]
                tool_args = json.loads(tool_args_str if tool_args_str else "{}")
                tool_result = await execute_mcp_tool(tool_name, tool_args)
                logger.info(f"MCP tool {tool_name} (ID: {tool_id}) result: {tool_result}")
                
                # Add tool result message for LLM2
                messages_for_llm2.append({
                    "tool_call_id": tool_id,
                    "role": "tool",
                    "name": tool_name,
                    "content": json.dumps(tool_result) if not isinstance(tool_result, str) else tool_result
                })
            except json.JSONDecodeError as e:
                logger.error(f"JSONDecodeError for MCP tool {tool_name} (ID: {tool_id}) arguments: {tool_call_data['function']['arguments']}. Error: {e}")
                messages_for_llm2.append({
                    "tool_call_id": tool_id, "role": "tool", "name": tool_name,
                    "content": json.dumps({"error": "Invalid arguments format", "details": str(e)})
                })
            except Exception as e:
                logger.error(f"Error executing MCP tool {tool_name} (ID: {tool_id}): {traceback.format_exc()}")
                messages_for_llm2.append({
                    "tool_call_id": tool_id, "role": "tool", "name": tool_name,
                    "content": json.dumps({"error": "Tool execution failed", "details": str(e)})
                })
        # else: Non-MCP tools are already part of LLM1's streamed response and client will handle them.
        # No special server-side execution for non-MCP tools in Option B for chat/completions.

    if not mcp_tools_were_called_and_executed:
        logger.info("LLM1 indicated tool_calls, but none were MCP tools. Client will handle. Stream ends.")
        # The [DONE] for LLM1 should have been sent. If not, this ensures it.
        yield "data: [DONE]\\n\\n"
        return

    # 3. Second LLM Call with MCP tool results
    # -----------------------------------------
    logger.info("Proceeding to LLM2 call with MCP tool results.")
    request_data_llm2 = original_request_data.copy()
    request_data_llm2["messages"] = messages_for_llm2
    request_data_llm2["stream"] = True # Ensure LLM2 is also streaming
    # Remove tools and tool_choice for the second call, as tools have been "used"
    request_data_llm2.pop("tools", None)
    request_data_llm2.pop("tool_choice", None)

    logger.info(f"LLM2 (stream) request to /chat/completions: {json.dumps(request_data_llm2, indent=2)}")
    
    try:
        async with http_client.stream(
            "POST",
            "/chat/completions",
            json=request_data_llm2,
            headers=internal_headers, # Reuse headers from LLM1
            timeout=None
        ) as rp_resp_llm2:
            rp_resp_llm2.raise_for_status()
            async for chunk_bytes in rp_resp_llm2.aiter_bytes():
                yield chunk_bytes # Stream LLM2 response directly to client
            # LLM2's stream will end with its own "data: [DONE]\\n\\n"
            logger.info("LLM2 stream finished.")
    except httpx.HTTPStatusError as e:
        logger.error(f"LLM2 HTTP error: {e.response.status_code} - {e.response.text}")
        yield f"data: {json.dumps({'error': {'message': e.response.text, 'type': 'llm2_request_failed', 'code': e.response.status_code}})}\\n\\n"
        yield "data: [DONE]\\n\\n"
    except Exception as e:
        logger.error(f"Error during LLM2 streaming: {traceback.format_exc()}")
        yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'llm2_stream_error'}})}\\n\\n"
        yield "data: [DONE]\\n\\n"


# Catch-all route to proxy any other requests to the AI provider
@app.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH", "TRACE"])
async def proxy_endpoint(request: Request, path_name: str):
    request_body_bytes = await request.body()
    
    # Special handling for /v1/chat/completions
    # Note: path_name might be "v1/chat/completions" or "chat/completions" depending on how the server is run/proxied
    is_chat_completions = "chat/completions" in path_name.lower() and request.method == "POST"

    if is_chat_completions:
        logger.info(f"Processing {request.method} for /v1/chat/completions with MCP enhancements.")
        try:
            original_request_data = json.loads(request_body_bytes.decode(\'utf-8\') if request_body_bytes else "{}")
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON body for /v1/chat/completions")
            return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

        # --- MCP Tool Injection ---
        request_tools = original_request_data.get("tools", [])
        if not isinstance(request_tools, list): request_tools = []
        
        # Create a dictionary of existing tool names for quick lookup
        existing_tool_names = {tool.get("function", {}).get("name") for tool in request_tools if tool.get("type") == "function"}

        merged_tools = list(request_tools) # Start with client tools

        for mcp_tool in mcp_functions_cache:
            if mcp_tool.get("name") not in existing_tool_names:
                merged_tools.append({
                    "type": "function",
                    "function": {
                        "name": mcp_tool["name"],
                        "description": mcp_tool.get("description", ""),
                        "parameters": mcp_tool.get("parameters", {})
                    }
                })
        
        if merged_tools:
            original_request_data["tools"] = merged_tools
            logger.info(f"Augmented tools for /v1/chat/completions. Total tools: {len(merged_tools)}")
        # --- End MCP Tool Injection ---

        is_streaming_request = original_request_data.get("stream", False)

        if is_streaming_request:
            logger.info("Handling STREAMING /v1/chat/completions with MCP Option B")
            # Prepare headers for internal requests, ensuring correct auth
            internal_request_headers = dict(request.headers)
            internal_request_headers.pop("host", None)
            internal_request_headers.pop("content-length", None) # httpx will set this
            internal_request_headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"

            return StreamingResponse(
                _handle_chat_completions_streaming_mcp(original_request_data, internal_request_headers),
                media_type="text/event-stream"
            )
        else: # Non-streaming
            logger.info("Handling NON-STREAMING /v1/chat/completions with MCP Option B")
            
            # 1. First LLM Call
            # ------------------
            request_data_llm1 = original_request_data.copy()
            request_data_llm1["stream"] = False # Ensure stream is false for the first non-streaming call

            # Prepare headers for the internal request
            internal_headers = {k: v for k, v in request.headers.items() if k.lower() not in ["host", "content-length", "accept-encoding"]}
            internal_headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"

            logger.info(f"LLM1 (non-stream) request to /chat/completions: {json.dumps(request_data_llm1, indent=2)}")
            
            try:
                rp_resp_llm1 = await http_client.post(
                    "/chat/completions", 
                    json=request_data_llm1,
                    headers=internal_headers,
                    timeout=None
                )
                rp_resp_llm1.raise_for_status()
                llm1_response_data = rp_resp_llm1.json()
                logger.info(f"LLM1 (non-stream) response: {json.dumps(llm1_response_data, indent=2)}")

            except httpx.HTTPStatusError as e:
                logger.error(f"LLM1 (non-stream) HTTP error: {e.response.status_code} - {e.response.text}")
                return JSONResponse(status_code=e.response.status_code, content=e.response.json())
            except Exception as e:
                logger.error(f"Error during LLM1 (non-stream) call: {traceback.format_exc()}")
                return JSONResponse(status_code=500, content={"error": "LLM1 request failed", "details": str(e)})

            # Check for tool calls in LLM1 response
            llm1_choice = llm1_response_data.get("choices", [{}])[0]
            llm1_message = llm1_choice.get("message", {})
            llm1_tool_calls = llm1_message.get("tool_calls", [])

            if not llm1_tool_calls:
                logger.info("LLM1 (non-stream) response has no tool calls. Returning response directly.")
                return JSONResponse(content=llm1_response_data, headers=dict(rp_resp_llm1.headers))

            # 2. Execute MCP Tools if any were called by LLM1
            # -----------------------------------------------
            messages_for_llm2 = original_request_data.get("messages", []).copy()
            messages_for_llm2.append(llm1_message) # Add assistant's message from LLM1

            mcp_tools_were_called_and_executed = False
            for tool_call in llm1_tool_calls:
                tool_name = tool_call.get("function", {}).get("name")
                tool_id = tool_call.get("id")

                if is_mcp_tool(tool_name):
                    mcp_tools_were_called_and_executed = true
                    logger.info(f"Executing MCP tool (non-stream): {tool_name} with ID: {tool_id}")
                    try:
                        tool_args_str = tool_call.get("function", {}).get("arguments", "{}")
                        tool_args = json.loads(tool_args_str)
                        tool_result = await execute_mcp_tool(tool_name, tool_args)
                        logger.info(f"MCP tool {tool_name} (ID: {tool_id}) result: {tool_result}")
                        
                        messages_for_llm2.append({
                            "tool_call_id": tool_id,
                            "role": "tool",
                            "name": tool_name,
                            "content": json.dumps(tool_result) if not isinstance(tool_result, str) else tool_result
                        })
                    except json.JSONDecodeError as e:
                        logger.error(f"JSONDecodeError for MCP tool {tool_name} (ID: {tool_id}) arguments: {tool_args_str}. Error: {e}")
                        messages_for_llm2.append({
                            "tool_call_id": tool_id, "role": "tool", "name": tool_name,
                            "content": json.dumps({"error": "Invalid arguments format", "details": str(e)})
                        })
                    except Exception as e:
                        logger.error(f"Error executing MCP tool {tool_name} (ID: {tool_id}): {traceback.format_exc()}")
                        messages_for_llm2.append({
                            "tool_call_id": tool_id, "role": "tool", "name": tool_name,
                            "content": json.dumps({"error": "Tool execution failed", "details": str(e)})
                        })
                # else: Non-MCP tools are part of LLM1's response. Client handles them.

            if not mcp_tools_were_called_and_executed:
                logger.info("LLM1 (non-stream) had tool_calls, but none were MCP. Returning LLM1 response.")
                return JSONResponse(content=llm1_response_data, headers=dict(rp_resp_llm1.headers))

            # 3. Second LLM Call with MCP tool results
            # -----------------------------------------
            logger.info("Proceeding to LLM2 (non-stream) call with MCP tool results.")
            request_data_llm2 = original_request_data.copy()
            request_data_llm2["messages"] = messages_for_llm2
            request_data_llm2["stream"] = False # Ensure LLM2 is non-streaming
            request_data_llm2.pop("tools", None)
            request_data_llm2.pop("tool_choice", None)

            logger.info(f"LLM2 (non-stream) request to /chat/completions: {json.dumps(request_data_llm2, indent=2)}")
            try:
                rp_resp_llm2 = await http_client.post(
                    "/chat/completions",
                    json=request_data_llm2,
                    headers=internal_headers, # Reuse headers
                    timeout=None
                )
                rp_resp_llm2.raise_for_status()
                llm2_response_data = rp_resp_llm2.json()
                logger.info(f"LLM2 (non-stream) response: {json.dumps(llm2_response_data, indent=2)}")
                return JSONResponse(content=llm2_response_data, headers=dict(rp_resp_llm2.headers))
            
            except httpx.HTTPStatusError as e:
                logger.error(f"LLM2 (non-stream) HTTP error: {e.response.status_code} - {e.response.text}")
                return JSONResponse(status_code=e.response.status_code, content=e.response.json())
            except Exception as e:
                logger.error(f"Error during LLM2 (non-stream) call: {traceback.format_exc()}")
                return JSONResponse(status_code=500, content={"error": "LLM2 request failed", "details": str(e)})

    # If not a special chat completions request, use the default proxy
    return await _default_proxy(request, path_name, request_body_bytes)


if __name__ == "__main__":
    logger.info(f"Starting API Adapter server on {API_ADAPTER_HOST}:{API_ADAPTER_PORT}")
    logger.info(f"Using OpenAI Base URL (internal): {OPENAI_BASE_URL_INTERNAL}")
    logger.info(f"Using OpenAI Base URL: {OPENAI_BASE_URL}")
    
    uvicorn.run("server:app", host=API_ADAPTER_HOST, port=API_ADAPTER_PORT, reload=True)