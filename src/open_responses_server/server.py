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
import traceback
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

async def _refresh_mcp_functions() -> None:
    """Fetch tools from all MCP servers and update the cache."""
    global mcp_functions_cache
    new_cache: List[Dict[str, Any]] = []
    for server in mcp_servers:
        try:
            tools = await server.list_tools()
            tool_entries = []
            for t in tools:
                entry = {"name": t["name"], "description": t.get("description"), "parameters": t.get("parameters", {})}
                tool_entries.append(entry)
            #logger.info(f"Refreshed tools from {server.name}: {[tool['name'] for tool in tool_entries]}")
            new_cache.extend(tool_entries)
        except Exception as e:
            logger.warning(f"Error refreshing tools from {server.name}: {e}")
    mcp_functions_cache = new_cache

async def _mcp_refresh_loop() -> None:
    """Background task: periodically refresh MCP tool cache."""
    while True:
        await _refresh_mcp_functions()
        await asyncio.sleep(MCP_TOOL_REFRESH_INTERVAL)

class MCPServer:
    """Wrapper for an MCP server session and tool execution."""
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.exit_stack = AsyncExitStack()
        self.session: ClientSession | None = None
        self._cleanup_lock = asyncio.Lock()

    async def initialize(self) -> None:
        command = shutil.which(self.config.get("command")) if self.config.get("command") != "npx" else shutil.which("npx")
        if not command:
            raise ValueError(f"Invalid command for MCP server {self.name}")
        params = StdioServerParameters(
            command=command,
            args=self.config.get("args", []),
            env={**os.environ, **self.config.get("env", {})} if self.config.get("env") else None,
        )
        transport = await self.exit_stack.enter_async_context(stdio_client(params))
        read, write = transport
        session = await self.exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self.session = session

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools with metadata from the server."""
        if not self.session:
            raise RuntimeError(f"MCP server {self.name} not initialized")
        resp = await self.session.list_tools()
        tools: List[Dict[str, Any]] = []
        for item in resp:
            if isinstance(item, tuple) and item[0] == "tools":
                for tool in item[1]:
                    # Gather name, description, and parameters schema
                    tools.append({
                        "name": tool.name,
                        "description": getattr(tool, "description", None),
                        "parameters": getattr(tool, "inputSchema", {}),
                    })
        return tools

    async def execute_tool(self, tool_name: str, arguments: dict) -> Any:
        if not self.session:
            raise RuntimeError(f"MCP server {self.name} not initialized")
        return await self.session.call_tool(tool_name, arguments)

    async def cleanup(self) -> None:
        async with self._cleanup_lock:
            await self.exit_stack.aclose()
            self.session = None

async def execute_mcp_tool(tool_name: str, arguments: dict) -> Any:
    """Finds the appropriate MCP server hosting the tool and executes it."""
    for server in mcp_servers:
        try:
            tools = await server.list_tools()
            for tool in tools:
                if tool.get("name") == tool_name:
                    logger.info(f"Found tool {tool_name} on server {server.name}")
                    return await server.execute_tool(tool_name, arguments)
        except Exception:
            logger.warning(f"Error checking tools on server {server.name}: {traceback.format_exc()}")
            continue
    raise RuntimeError(f"Tool '{tool_name}' not found on any MCP server")
    
def is_mcp_tool(tool_name: str) -> bool:
    """
    Determines if the given tool name belongs to an MCP tool.
    
    Args:
        tool_name: The name of the tool to check
        
    Returns:
        bool: True if it's an MCP tool, False otherwise
    """
    for func in mcp_functions_cache:
        if func.get("name") == tool_name:
            return True
    return False

@app.on_event("startup")
async def startup_mcp_servers():
    """Initialize all MCP servers defined in servers_config.json on startup."""
    config_path = Path(__file__).parent / "servers_config.json"
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        for name, srv in cfg.get("mcpServers", {}).items():
            server = MCPServer(name, srv)
            try:
                await server.initialize()
                mcp_servers.append(server)
                logger.info(f"Initialized MCP server: {name}")
                # Log initial tool list per server
                try:
                    tools = await server.list_tools()
                    logger.info(f"Initial tools for {name}: {[t['name'] for t in tools]}")
                except Exception as e:
                    logger.warning(f"Could not list tools for {name} on startup: {e}")
            except Exception as e:
                logger.error(f"Error initializing MCP server {name}: {e}")
    except FileNotFoundError:
        logger.warning("servers_config.json not found. No MCP servers will be available.")
    # Start background refresh of MCP tools
    asyncio.create_task(_mcp_refresh_loop())

@app.on_event("shutdown")
async def shutdown_mcp_servers():
    """Clean up all MCP servers on shutdown."""
    for server in mcp_servers:
        try:
            await server.cleanup()
            logger.info(f"Cleaned up MCP server: {server.name}")
        except Exception as e:
            logger.error(f"Error cleaning up MCP server {server.name}: {e}")

# Pydantic models for requests and responses
class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict] = None

class Tool(BaseModel):
    type: str = "function"
    function: ToolFunction

class OutputText(BaseModel):
    type: str = "output_text"
    text: str

class Message(BaseModel):
    id: Optional[str] = None
    type: str = "message"
    role: str
    content: List[Any]

class TextFormat(BaseModel):
    type: str = "text"

class ResponseItem(BaseModel):
    id: str
    type: str
    role: str
    content: List[Any]

class ResponseModel(BaseModel):
    id: str
    object: str = "response"
    created_at: int
    status: str = "in_progress"
    error: Optional[Any] = None
    incomplete_details: Optional[Any] = None
    instructions: Optional[str] = None
    max_output_tokens: Optional[int] = None
    model: str
    output: List[Any] = []
    parallel_tool_calls: bool = True
    previous_response_id: Optional[str] = None
    reasoning: Dict = Field(default_factory=lambda: {"effort": None, "summary": None})
    store: bool = False
    temperature: float = 1.0
    text: Dict = Field(default_factory=lambda: {"format": {"type": "text"}})
    tool_choice: str = "auto"
    tools: List[Tool] = []
    top_p: float = 1.0
    truncation: str = "disabled"
    usage: Optional[Dict] = None
    user: Optional[str] = None
    metadata: Dict = Field(default_factory=dict)

class ResponseCreateRequest(BaseModel):
    model: str
    input: Optional[List[Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[str] = "auto"
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_output_tokens: Optional[int] = None
    user: Optional[str] = None
    metadata: Optional[Dict] = None

class ToolCallArgumentsDelta(BaseModel):
    type: str = "response.function_call_arguments.delta"
    item_id: str
    output_index: int
    delta: str

class ToolCallArgumentsDone(BaseModel):
    type: str = "response.function_call_arguments.done"
    id: str
    output_index: int
    arguments: str

class ToolCallsCreated(BaseModel):
    type: str = "response.tool_calls.created"
    item_id: str
    output_index: int
    tool_call: Dict

class OutputTextDelta(BaseModel):
    type: str = "response.output_text.delta"
    item_id: str
    output_index: int
    delta: str

class ResponseCreated(BaseModel):
    type: str = "response.created"
    response: ResponseModel

class ResponseInProgress(BaseModel):
    type: str = "response.in_progress"
    response: ResponseModel

class ResponseCompleted(BaseModel):
    type: str = "response.completed"
    response: ResponseModel

# Helper functions
def current_timestamp() -> int:
    return int(time.time())

def convert_responses_to_chat_completions(request_data: dict) -> dict:
    """
    Convert a request in Responses API format to chat.completions API format.
    """
    logger = logging.getLogger("api_adapter_conversion")
    # Log only essential info - model, tools count, if instructions present
    logger.info(f"Request: model={request_data.get('model')}, " +
                f"tools={len(request_data.get('tools', []))}, " +
                f"has_instructions={'instructions' in request_data}")
    logger.info(f"Tools in request={request_data.get('tools', [])}")
    
    chat_request = {
        "model": request_data.get("model"),
        "temperature": request_data.get("temperature", 1.0),
        "top_p": request_data.get("top_p", 1.0),
        "stream": request_data.get("stream", False),
    }

    # Convert any max_output_tokens to max_tokens
    if "max_output_tokens" in request_data:
        chat_request["max_tokens"] = request_data["max_output_tokens"]

    # Convert input to messages
    messages = []
    
    # Check for previous_response_id and load conversation history if available
    previous_response_id = request_data.get("previous_response_id")
    if previous_response_id and previous_response_id in conversation_history:
        logger.info(f"Loading conversation history from previous_response_id: {previous_response_id}")
        messages = conversation_history[previous_response_id].copy()
        logger.info(f"Loaded {len(messages)} messages from conversation history")
    
    # Check for system message first if we have instructions
    if "instructions" in request_data:
        # If we're loading from history, check if we already have a system message
        has_system_message = any(msg.get("role") == "system" for msg in messages)
        if not has_system_message:
            messages.append({"role": "system", "content": request_data["instructions"]})
            logger.info(f"Added system message from instructions")
        else:
            # Replace existing system message with the new instructions
            for msg in messages:
                if msg.get("role") == "system":
                    msg["content"] = request_data["instructions"]
                    logger.info(f"Updated existing system message with new instructions")
                    break
    
    # Check for previous tool responses in the input
    if "input" in request_data and request_data["input"]:
        user_message = {"role": "user", "content": ""}
        logger.info(f"Processing input messages {request_data['input']}")
        for i, item in enumerate(request_data["input"]):
            if isinstance(item, dict):
                if item.get("type") == "message" and item.get("role") == "user":
                    # Add user message
                    content = ""
                    if "content" in item:
                        for j, content_item in enumerate(item["content"]):
                            if isinstance(content_item, dict) and content_item.get("type") == "input_text":
                                content += content_item.get("text", "")
                            elif isinstance(content_item, dict) and content_item.get("type") == "text":
                                content += content_item.get("text", "")
                            elif isinstance(content_item, str):
                                content += content_item
                    user_message = {"role": "user", "content": content}
                    messages.append(user_message)
                    # Log user message content for context
                    logger.info(f"User message: {content[:100]}...")
                    
                elif item.get("type") == "function_call_output":
                    # Add tool output - log tool usage
                    logger.info(f"Tool response: call_id={item.get('call_id')}, output={item.get('output', '')[:50]}...")
                    
                    # Check if we have a corresponding assistant message with a tool call first
                    call_id = item.get("call_id")
                    has_matching_tool_call = False
                    
                    # Look for a matching tool call in the existing messages
                    for msg in messages:
                        if msg.get("role") == "assistant" and "tool_calls" in msg:
                            for tool_call in msg["tool_calls"]:
                                if tool_call.get("id") == call_id:
                                    has_matching_tool_call = True
                                    break
                    
                    if has_matching_tool_call:
                        # Only add the tool response if we found a matching tool call
                        tool_message = {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": item.get("output", "")
                        }
                        messages.append(tool_message)
                    else:
                        # If no matching tool call, we need to add an assistant message with the tool call first
                        # as this could be from a previous conversation
                        tool_name = item.get("name", "unknown_tool")
                        
                        # Create an assistant message with a tool call
                        assistant_message = {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": item.get("arguments", "{}")
                                }
                            }]
                        }
                        messages.append(assistant_message)
                        
                        # Then add the tool response
                        tool_message = {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": item.get("output", "")
                        }
                        messages.append(tool_message)
                        logger.info(f"Added assistant message with tool call and corresponding tool response for {tool_name}")
                elif item.get("type") == "message" and item.get("role") == "assistant":
                    # Handle assistant messages from previous conversations
                    content = ""
                    if "content" in item and isinstance(item["content"], list):
                        for content_item in item["content"]:
                            if isinstance(content_item, dict) and content_item.get("type") == "output_text":
                                content += content_item.get("text", "")
                    
                    if content:
                        messages.append({"role": "assistant", "content": content})
                        logger.info(f"Added assistant message: {content[:100]}...")
            elif isinstance(item, str):
                # Simple string input
                messages.append({"role": "user", "content": item})
                logger.info(f"User message (string): {item[:100]}...")
    
    # If we only have a system message or no messages at all, add an empty user message
    if not messages or (len(messages) == 1 and messages[0]["role"] == "system"):
        messages.append({"role": "user", "content": ""})
    
    chat_request["messages"] = messages

    # Convert tools - log each tool being processed
    if "tools" in request_data and request_data["tools"]:
        chat_request["tools"] = []
        
        for i, tool in enumerate(request_data["tools"]):
            try:
                logger.info(f"Trying to convert tool {i}: {tool}")
                if not isinstance(tool, dict) or "type" not in tool or tool.get("type") != "function":
                    continue
                    
                function_obj = tool
                if not isinstance(function_obj, dict) or "name" not in function_obj:
                    continue
                
                function_data = {
                    "name": function_obj["name"],
                }
                
                # Log tool information
                logger.info(f"Converting Tool {i}: {function_data['name']}")
                
                if "description" in function_obj:
                    function_data["description"] = function_obj["description"]
                    
                if "parameters" in function_obj:
                    function_data["parameters"] = function_obj["parameters"]
                
                chat_request["tools"].append({
                    "type": "function",
                    "function": function_data
                })
            except Exception as e:
                logger.error(f"Error processing tool {i}: {str(e)}")
    
    # Handle tool_choice
    if "tool_choice" in request_data:
        chat_request["tool_choice"] = request_data["tool_choice"]
    
    # Add optional parameters if they exist
    for key in ["user", "metadata"]:
        if key in request_data and request_data[key] is not None:
            chat_request[key] = request_data[key]
    
    logger.info(f"Converted to chat completions: {len(messages)} messages, {len(chat_request.get('tools', []))} tools")
    return chat_request

async def process_chat_completions_stream(response, chat_request=None):
    """
    Process the streaming response from chat.completions API.
    Tracks the state of tool calls to properly convert them to Responses API events.
    
    Args:
        response: The streaming response from chat.completions API
        chat_request: (Optional) The chat request that was sent to the API for history storage
    """
    logger = logging.getLogger("api_adapter_stream")
    tool_calls = {}  # Store tool calls being built
    response_id = f"resp_{uuid.uuid4().hex}"
    tool_call_counter = 0
    message_id = f"msg_{uuid.uuid4().hex}"
    output_text_content = ""  # Track the full text content for logging
    logger.info(f"Processing streaming response from chat.completions API response_id {response_id}; message_id {message_id}")
    
    # Create and yield the initial response.created event
    response_obj = ResponseModel(
        id=response_id,
        created_at=current_timestamp(),
        model="", # Will be filled from the first chunk
        output=[]
    )
    
    created_event = ResponseCreated(
        type="response.created",
        response=response_obj
    )
    logger.info(f"Emitting {created_event}")
    yield f"data: {json.dumps(created_event.dict())}\n\n"
    
    # Also emit the in_progress event
    in_progress_event = ResponseInProgress(
        type="response.in_progress",
        response=response_obj
    )
    
    logger.info(f"Emitting {in_progress_event}")
    yield f"data: {json.dumps(in_progress_event.dict())}\n\n"
    
    chunk_counter = 0
    try:
        async for chunk in response.aiter_lines():
            chunk_counter += 1
            if not chunk.strip():
                continue
            #logger.info(chunk)
                
            # Handle [DONE] message
            if chunk.strip() == "data: [DONE]" or chunk.strip() == "[DONE]":
                logger.info(f"Received [DONE] message after {chunk_counter} chunks (status: {response_obj.status})")
                
                # If we haven't already completed the response, do it now
                if response_obj.status != "completed":
                    # If no output, add empty message
                    if not response_obj.output:
                        response_obj.output.append({
                            "id": message_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": f"{output_text_content}\n\n" or "Done"}]
                        })
                    
                    response_obj.status = "completed"
                    completed_event = ResponseCompleted(
                        type="response.completed",
                        response=response_obj
                    )
                    
                    # Save conversation history for DONE events if we have chat_request
                    if chat_request and output_text_content:
                        # Get the existing messages from the request
                        messages = chat_request.get("messages", [])
                        
                        # Add the assistant response to the conversation history
                        messages.append({
                            "role": "assistant",
                            "content": output_text_content
                        })
                        
                        # Store in conversation history
                        conversation_history[response_id] = messages
                        logger.info(f"Saved conversation history for response_id {response_id} with {len(messages)} messages")
                        
                        # Trim conversation history if it grows too large
                        if len(conversation_history) > MAX_CONVERSATION_HISTORY:
                            # Remove oldest conversations
                            excess = len(conversation_history) - MAX_CONVERSATION_HISTORY
                            oldest_keys = sorted(conversation_history.keys())[:excess]
                            for key in oldest_keys:
                                del conversation_history[key]
                            logger.info(f"Trimmed {excess} oldest conversations from history")
                    
                    logger.info(f"Emitting completed event after [DONE]: {completed_event}")
                    yield f"data: {json.dumps(completed_event.dict())}\n\n"
                continue

            # Skip prefix if present
            if chunk.startswith("data: "):
                chunk = chunk[6:]
                
            try:
                data = json.loads(chunk)
                logger.info(f"data: {data}")
                # Extract model name from the first chunk if available
                if "model" in data and response_obj.model == "":
                    response_obj.model = data["model"]
                
                # Check for delta choices
                if "choices" in data and data["choices"]:
                    choice = data["choices"][0]
                    
                    # Process delta
                    if "delta" in choice:
                        delta = choice["delta"]

                        # Handle OpenAI 'function_call' style deltas
                        if "function_call" in delta:
                            func = delta["function_call"]
                            index = 0
                            # Initialize tool call entry if first fragment
                            if index not in tool_calls:
                                tool_calls[index] = {
                                    "id": f"call_{uuid.uuid4().hex}",
                                    "function": {"name": func.get("name", ""), "arguments": ""},
                                    "output_index": 0
                                }
                                # Emit created event for function call
                                created_evt = ToolCallsCreated(
                                    type="response.tool_calls.created",
                                    item_id=tool_calls[index]["id"],
                                    output_index=0,
                                    tool_call={"id": tool_calls[index]["id"], "name": tool_calls[index]["function"]["name"], "arguments": ""}
                                )
                                yield f"data: {json.dumps(created_evt.dict())}\n\n"
                            # Append argument fragment if present
                            if "arguments" in func and func.get("arguments") is not None:
                                fragment = func.get("arguments")
                                tool_calls[index]["function"]["arguments"] += fragment
                                # Emit arguments delta
                                delta_evt = ToolCallArgumentsDelta(
                                    type="response.function_call_arguments.delta",
                                    item_id=tool_calls[index]["id"],
                                    output_index=0,
                                    delta=fragment
                                )
                                yield f"data: {json.dumps(delta_evt.dict())}\n\n"
                            continue  # skip other tool call handling

                        # Handle legacy 'tool_calls' schema
                        if "tool_calls" in delta and delta["tool_calls"]:
                            for tool_delta in delta["tool_calls"]:
                                index = tool_delta.get("index", 0)
                                
                                # Initialize tool call if not exists
                                if index not in tool_calls:
                                    tool_calls[index] = {
                                        "id": tool_delta.get("id", f"call_{uuid.uuid4().hex}"),
                                        "type": tool_delta.get("type", "function"),
                                        "function": {
                                            "name": tool_delta.get("function", {}).get("name", ""),
                                            "arguments": tool_delta.get("function", {}).get("arguments", ""),
                                        },
                                        "item_id": f"tool_call_{uuid.uuid4().hex}",
                                        "output_index": tool_call_counter
                                    }
                                    
                                    # If we got a tool name, emit the created event
                                    if "function" in tool_delta and "name" in tool_delta["function"]:
                                        tool_call = tool_calls[index]
                                        tool_call["function"]["name"] = tool_delta["function"]["name"]
                                        # Log tool call creation
                                        logger.info(f"Tool call created: {tool_call['function']['name']}")
                                        
                                        # Check if this is an MCP tool or a user-defined tool
                                        is_mcp = is_mcp_tool(tool_call["function"]["name"])
                                        tool_status = "in_progress" if is_mcp else "ready"
                                        
                                        # Add the tool call to the response output in Responses API format
                                        response_obj.output.append({
                                            "arguments": tool_call["function"]["arguments"],
                                            "call_id": tool_call["id"],
                                            "name": tool_call["function"]["name"],
                                            "type": "function_call",
                                            "id": tool_call["id"],
                                            "status": tool_status
                                        })
                                        
                                        # Emit the in_progress event
                                        in_progress_event = ResponseInProgress(
                                            type="response.in_progress",
                                            response=response_obj
                                        )
                                        
                                        logger.info(f"Emitting {in_progress_event}")
                                        yield f"data: {json.dumps(in_progress_event.dict())}\n\n"


                                        # created_event = ToolCallsCreated(
                                        #     type="response.tool_calls.created",
                                        #     item_id=tool_call["item_id"],
                                        #     output_index=tool_call["output_index"],
                                        #     tool_call={
                                        #         "id": tool_call["id"],
                                        #         "type": tool_call["type"],
                                        #         "function": {
                                        #             "name": tool_call["function"]["name"],
                                        #             "arguments": ""
                                        #         }
                                        #     }
                                        # )
                                        
                                        # yield f"data: {json.dumps(created_event.dict())}\n\n"
                                        tool_call_counter += 1
                                
                                # Process function arguments if present
                                if "function" in tool_delta and "arguments" in tool_delta["function"]:
                                    arg_fragment = tool_delta["function"]["arguments"]
                                    tool_calls[index]["function"]["arguments"] += arg_fragment
                                    
                                    # Emit delta event
                                    args_event = ToolCallArgumentsDelta(
                                        type="response.function_call_arguments.delta",
                                        item_id=tool_calls[index]["item_id"],
                                        output_index=tool_calls[index]["output_index"],
                                        delta=arg_fragment
                                    )
                                    
                                    yield f"data: {json.dumps(args_event.dict())}\n\n"
                        
                        # Handle content (text)
                        elif "content" in delta and delta["content"] is not None:
                            content_delta = delta["content"]
                            output_text_content += content_delta
                            
                            # Create a new message if it doesn't exist
                            if not response_obj.output:
                                response_obj.output.append({
                                    "id": message_id,
                                    "type": "message",
                                    "role": "assistant",
                                    "content": [{"type": "output_text", "text": output_text_content or "(No update)"}]  # Updated to use [{}] instead of []
                                })
                            
                            # Emit text delta event
                            text_event = OutputTextDelta(
                                type="response.output_text.delta",
                                item_id=message_id,
                                output_index=0,
                                delta=content_delta
                            )
                            
                            yield f"data: {json.dumps(text_event.dict())}\n\n"
                    
                    if "finish_reason" in choice and choice["finish_reason"] is not None:
                        logger.info(f"Received finish_reason: {choice['finish_reason']}")
                        
                        # If the finish reason indicates a function call, execute the tool via MCP
                        if choice["finish_reason"] == "function_call":
                            logger.info("Processing tool call")
                            for index, tool_call in tool_calls.items():
                                tool_name = tool_call["function"]["name"]
                                # Parse the arguments JSON
                                try:
                                    args = json.loads(tool_call["function"]["arguments"])
                                except Exception:
                                    args = {}
                                    
                                # Check if this is an MCP tool or a non-MCP tool
                                if is_mcp_tool(tool_name):
                                    logger.info(f"Executing MCP tool: {tool_name}")
                                    # Execute MCP tool
                                    try:
                                        result = await execute_mcp_tool(tool_name, args)
                                    except Exception as e:
                                        result = {"error": str(e)}
                                    logger.info(f"MCP tool result for {tool_name}: {result}")
                                    
                                    # Append as function_call_output
                                    response_obj.output.append({
                                        "id": tool_call["id"],
                                        "type": "function_call_output",
                                        "call_id": tool_call["id"],
                                        "output": result
                                    })
                                    
                                    # Convert result to JSON, with fallback to string if needed
                                    try:
                                        text = json.dumps(result)
                                    except TypeError:
                                        text = json.dumps(str(result))
                                        
                                    text_event = OutputTextDelta(
                                        type="response.output_text.delta",
                                        item_id=tool_call["id"],
                                        output_index=0,
                                        delta=text
                                    )
                                    yield f"data: {json.dumps(text_event.dict())}\n\n"
                                else:
                                    # For non-MCP tools, send the function call back to the client in Responses API format
                                    logger.info(f"Forwarding non-MCP tool call to client: {tool_name}")
                                    
                                    # Include the function call in the response
                                    response_obj.output.append({
                                        "id": tool_call["id"],
                                        "type": "function_call",
                                        "name": tool_name,
                                        "arguments": tool_call["function"]["arguments"],
                                        "call_id": tool_call["id"],
                                        "status": "ready"
                                    })
                                    
                                # After tool handling, complete the response
                                response_obj.status = "completed"
                                completed_event = ResponseCompleted(
                                    type="response.completed",
                                    response=response_obj
                                )
                                
                                # Save conversation history if we have chat_request available
                                if chat_request:
                                    # Get the existing messages from the request
                                    messages = chat_request.get("messages", [])
                                    
                                    # Add the assistant response with tool calls
                                    assistant_message = {
                                        "role": "assistant",
                                        "content": None,  # Content is null for tool calls
                                        "tool_calls": [{
                                            "id": tool_call["id"],
                                            "type": "function",
                                            "function": {
                                                "name": tool_call["function"]["name"],
                                                "arguments": tool_call["function"]["arguments"]
                                            }
                                        }]
                                    }
                                    messages.append(assistant_message)
                                    
                                    # Add the tool response for immediate tools
                                    if is_mcp:
                                        # For MCP tools, also add the tool response
                                        tool_message = {
                                            "role": "tool",
                                            "tool_call_id": tool_call["id"],
                                            "content": json.dumps(result)
                                        }
                                        messages.append(tool_message)
                                    
                                    # Store in conversation history
                                    conversation_history[response_id] = messages
                                    logger.info(f"Saved conversation history for response_id {response_id} with {len(messages)} messages")
                                    
                                    # Trim conversation history if it grows too large
                                    if len(conversation_history) > MAX_CONVERSATION_HISTORY:
                                        # Remove oldest conversations
                                        excess = len(conversation_history) - MAX_CONVERSATION_HISTORY
                                        oldest_keys = sorted(conversation_history.keys())[:excess]
                                        for key in oldest_keys:
                                            del conversation_history[key]
                                        logger.info(f"Trimmed {excess} oldest conversations from history")
                                
                                logger.info(f"Emitting completed event after function_call: {completed_event}")
                                yield f"data: {json.dumps(completed_event.dict())}\n\n"
                                return  # End streaming after function result
                        # If the finish reason is "tool_calls", emit the arguments.done events
                        if choice["finish_reason"] == "tool_calls":
                            for index, tool_call in tool_calls.items():
                                # Log the complete tool call arguments
                                logger.info(f"Tool call completed: {tool_call['function']['name']} with arguments: {tool_call['function']['arguments']}")
                                
                                # Check if this is an MCP tool or a user-defined tool
                                is_mcp = is_mcp_tool(tool_call["function"]["name"])
                                
                                # For non-MCP tools, we leave them in the "ready" state for the client to handle
                                if is_mcp:
                                    done_event = ToolCallArgumentsDone(
                                        type="response.function_call_arguments.done",
                                        id=tool_call["item_id"],
                                        output_index=tool_call["output_index"],
                                        arguments=tool_call["function"]["arguments"]
                                    )
                                    logger.info(f"Emitting {done_event}")
                                    yield f"data: {json.dumps(done_event.dict())}\n\n"
                                
                                # Update response object based on tool type
                                if not is_mcp:
                                    # For non-MCP tools, keep status "ready" for client handling
                                    # Find any existing entry for this tool call and update args
                                    found = False
                                    for output_item in response_obj.output:
                                        if output_item.get("id") == tool_call["id"] and output_item.get("type") == "function_call":
                                            output_item["arguments"] = tool_call["function"]["arguments"]
                                            found = True
                                            break
                                    
                                    # If not found, add it
                                    if not found:
                                        response_obj.output.append({
                                            "id": tool_call["id"],
                                            "type": "function_call",
                                            "name": tool_call["function"]["name"],
                                            "arguments": tool_call["function"]["arguments"],
                                            "call_id": tool_call["id"],
                                            "status": "ready"
                                        })
                                else:
                                    # For MCP tools, add as tool_call
                                    response_obj.output.append({
                                        "id": tool_call["item_id"],
                                        "type": "tool_call",
                                        "function": {
                                            "name": tool_call["function"]["name"],
                                            "arguments": tool_call["function"]["arguments"]
                                        }
                                    })
                        
                        # If the finish reason is "stop", emit the completed event
                        if choice["finish_reason"] == "stop":
                            logger.info("Received stop finish reason")
                            # If we have any text content, add it to the output
                            if not response_obj.output:
                                response_obj.output.append({
                                    "id": message_id,
                                    "type": "message",
                                    "role": "assistant",
                                    "content": [{"type": "output_text", "text": f"{output_text_content}\n\n" or "Done"}]
                                })
                            
                            # Log complete output text
                            logger.info(f"Response completed with text: {output_text_content[:100]}...\n\n")
                                
                            response_obj.status = "completed"
                            response_obj.output= [{
                                "id": message_id,
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": output_text_content or "(No update)"}]
                            }]
                            completed_event = ResponseCompleted(
                                type="response.completed",
                                response=response_obj
                            )
                            
                            # Save conversation history if we have chat_request available
                            if chat_request:
                                # Get the existing messages from the request
                                messages = chat_request.get("messages", [])
                                
                                # Add the assistant response to the conversation history
                                messages.append({
                                    "role": "assistant",
                                    "content": output_text_content or "(No update)"
                                })
                                
                                # Store in conversation history
                                conversation_history[response_id] = messages
                                logger.info(f"Saved conversation history for response_id {response_id} with {len(messages)} messages")
                                
                                # Trim conversation history if it grows too large
                                if len(conversation_history) > MAX_CONVERSATION_HISTORY:
                                    # Remove oldest conversations
                                    excess = len(conversation_history) - MAX_CONVERSATION_HISTORY
                                    oldest_keys = sorted(conversation_history.keys())[:excess]
                                    for key in oldest_keys:
                                        del conversation_history[key]
                                    logger.info(f"Trimmed {excess} oldest conversations from history")
                            
                            logger.info(f"Emitting completed event after stop: {completed_event}")
                            yield f"data: {json.dumps(completed_event.dict())}\n\n"
                
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from chunk: {chunk}")
                continue
    
    except Exception as e:
        logger.error(f"Error processing streaming response: {str(e)}")
        # Emit a completion event if we haven't already
        if response_obj.status != "completed":
            response_obj.status = "completed"
            response_obj.error = {"message": str(e)}
            
            completed_event = ResponseCompleted(
                type="response.completed",
                response=response_obj
            )
            
            yield f"data: {json.dumps(completed_event.dict())}\n\n"

# API endpoints
@app.post("/responses")
async def create_response(request: Request):
    """
    Create a response in Responses API format, translating to/from chat.completions API.
    """
    try:
        logger = logging.getLogger("api_adapter_responses")
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
        if mcp_functions_cache:
            # Get existing tools from request_data or initialize empty list
            existing_tools = request_data.get("tools", [])
            
            # Create tools format for MCP functions
            mcp_tools = [
                {"type": "function", "name": f["name"], "description": f.get("description"), "parameters": f.get("parameters", {})}
                for f in mcp_functions_cache
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
            
            logger.info(f"Appended {len(filtered_mcp_tools)} MCP tools to {len(existing_tools)} existing tools in request_data")
        # Convert request to chat.completions format
        chat_request = convert_responses_to_chat_completions(request_data)
        # Inject cached MCP tool definitions
        if mcp_functions_cache:
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
            for func in mcp_functions_cache:
                if func.get("name") not in existing_tool_names:
                    chat_request["tools"].append({
                        "type": "function",
                        "function": func
                    })
            
            # Remove the functions key as we've converted to tools format
            chat_request.pop("functions", None)
            
            logger.info(f"Converted existing functions and MCP functions to tools format")
        # else:
        #     chat_request.pop("tools", None)
        #     chat_request.pop("functions", None)
        #     logger.info("No MCP functions cached, sending without functions")
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
                    # Fetch available MCP tools and format as functions for chat.completions
                    mcp_functions = []
                    for server in mcp_servers:
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
                    async with http_client.stream(
                        "POST",
                        "/v1/chat/completions",
                        json=chat_request,
                        timeout=120.0
                    ) as response:
                        logger.info(f"Stream request status: {response.status_code}")
                        
                        if response.status_code != 200:
                            error_content = await response.aread()
                            logger.error(f"Error from LLM API: {error_content}")
                            yield f"data: {json.dumps({'type': 'error', 'error': {'message': f'Error from LLM API: {response.status_code}'}})}\n\n"
                            return
                        
                        async for event in process_chat_completions_stream(response, chat_request):
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
            
    except Exception as e:
        logger.error(f"Error in create_response: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok", "adapter": "running"}

@app.get("/")
async def root():
    return {"message": "API Adapter is running. Use /responses endpoint to interact with the API."}

# Catch-all route to proxy any other requests to the AI provider
@app.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH", "TRACE"])
async def proxy_endpoint(request: Request, path_name: str):
    """
    Proxy any requests not handled by other routes directly to the AI provider without changes.
    This ensures compatibility with applications that expect the full OpenAI API.
    """
    try:
        logger = logging.getLogger("api_adapter_proxy")
        logger.info(f"Proxying request to {path_name}")

        # Get the request body if available
        body = await request.body()
        # Log the raw payload being proxied
        logger.info(f"Proxy request payload for {path_name}: {body.decode('utf-8', errors='ignore')}")
        # Get headers but exclude host
        headers = {k.lower(): v for k, v in request.headers.items() if k.lower() != 'host'}
        
        # Make sure we have authorization header
        if 'authorization' not in headers and OPENAI_API_KEY:
            headers['authorization'] = f'Bearer {OPENAI_API_KEY}'
            
        logger.info(f"Proxying request to {path_name}")
        
        # Determine if this is a streaming request
        is_stream = False
        if body:
            try:
                data = json.loads(body)
                is_stream = data.get('stream', False)
            except:
                pass
                
        # Forward the request to the AI provider
        if is_stream:
            # Handle streaming response
            async def stream_proxy():
                try:
                    # Create a client for this specific request
                    async with httpx.AsyncClient(timeout=60.0) as client:
                        async with client.stream(
                            request.method,
                            f"{OPENAI_BASE_URL_INTERNAL}/v1/{path_name}",  # Add /v1 here
                            headers=headers,
                            content=body,
                            timeout=60.0
                        ) as response:
                            if response.status_code != 200:
                                # If there's an error, just return the error response
                                error_content = await response.aread()
                                yield error_content
                                return
                                
                            # Stream the response back
                            async for chunk in response.aiter_bytes():
                                yield chunk
                except Exception as e:
                    logger.error(f"Error in proxy_endpoint streaming: {str(e)}")
                    yield f"Error: {str(e)}".encode('utf-8')
            
            return StreamingResponse(
                stream_proxy(),
                media_type=request.headers.get('accept', 'application/json'),
                status_code=200
            )
        else:
            # Handle regular response
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.request(
                    request.method,
                    f"{OPENAI_BASE_URL_INTERNAL}/v1/{path_name}",  # Add /v1 here
                    headers=headers,
                    content=body,
                    timeout=60.0
                )
                
                # Create the response with the same status code and headers from the proxied response
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=dict(response.headers)
                )
                
    except Exception as e:
        logger.error(f"Error in proxy_endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error proxying request: {str(e)}"
        )

if __name__ == "__main__":
    logger.info(f"Starting API Adapter server on {API_ADAPTER_HOST}:{API_ADAPTER_PORT}")
    logger.info(f"Using OpenAI Base URL (internal): {OPENAI_BASE_URL_INTERNAL}")
    logger.info(f"Using OpenAI Base URL: {OPENAI_BASE_URL}")
    
    uvicorn.run("server:app", host=API_ADAPTER_HOST, port=API_ADAPTER_PORT, reload=True)