#!/usr/bin/env python3

"""
Tools Manager for the OpenAI Responses Server

This module handles the tools loading, fetching, management, and execution for the OpenAI Responses Server.
It provides functionality for converting between different tool formats and processing tool calls
in streaming responses.

This module is responsible for:
1. Converting tools between different API formats
2. Processing tool calls in streaming responses
3. Generating events for tool call updates
4. Tracking and managing tool call state
5. Processing tool responses in input data
"""

import os
import json
import logging
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field
import time
import httpx
from .version import __version__

# Configure logging
logger = logging.getLogger("tools_manager")

# Pydantic models for tools
class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict] = None

class Tool(BaseModel):
    type: str = "function"
    function: ToolFunction

class ToolCallArgumentsDelta(BaseModel):
    """Model for tool call argument delta events"""
    type: str = "response.function_call_arguments.delta"
    item_id: str
    output_index: int
    delta: str

class ToolCallArgumentsDone(BaseModel):
    """Model for tool call argument completion events"""
    type: str = "response.function_call_arguments.done"
    id: str
    output_index: int
    arguments: str

class ToolCallsCreated(BaseModel):
    """Model for tool call creation events"""
    type: str = "response.tool_calls.created"
    item_id: str
    output_index: int
    tool_call: Dict

# Helper functions for tools processing
def convert_tools_to_chat_completions_format(tools: List[Dict]) -> List[Dict]:
    """
    Convert tools from Responses API format to chat.completions API format.
    
    Args:
        tools: List of tool definitions in Responses API format
        
    Returns:
        List of tool definitions in chat.completions API format
    """
    logger.info(f"Converting {len(tools)} tools to chat completions format")
    
    chat_tools = []
    for i, tool in enumerate(tools):
        try:
            if not isinstance(tool, dict) or "type" not in tool or tool.get("type") != "function":
                continue
                
            function_obj = tool.get("function", {})
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
            
            chat_tools.append({
                "type": "function",
                "function": function_data
            })
        except Exception as e:
            logger.error(f"Error processing tool {i}: {str(e)}")
            
    return chat_tools

def extract_tools_from_request(request_data: Dict) -> List[Dict]:
    """
    Extract tools from the request data.
    
    Args:
        request_data: The request data in Responses API format
        
    Returns:
        List of tool definitions
    """
    tools = request_data.get("tools", [])
    logger.info(f"Extracting {len(tools)} tools from request")
    return tools

def process_tool_calls_in_delta(
    delta: Dict, 
    tool_calls: Dict, 
    tool_call_counter: int,
    response_obj: Any,
) -> Tuple[Dict, int, List[Dict]]:
    """
    Process tool calls in a delta chunk from the chat.completions API.
    
    Args:
        delta: The delta chunk containing tool_calls
        tool_calls: Dictionary to track tool calls being built
        tool_call_counter: Counter for tool call indices
        response_obj: The response object being built
        
    Returns:
        Updated tool_calls dictionary, tool_call_counter, and events to emit
    """
    events_to_emit = []
    
    if "tool_calls" not in delta or not delta["tool_calls"]:
        return tool_calls, tool_call_counter, events_to_emit
    
    logger.info(f"Processing tool calls in delta: {delta}")
    
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
                
                response_obj.output.append({
                    "arguments": tool_call["function"]["arguments"],
                    "call_id": tool_call["id"],
                    "name": tool_call["function"]["name"],
                    "type": "function_call",
                    "id": tool_call["id"],
                    "status": "in_progress"
                })
                
                # Create tool call created event
                created_event = ToolCallsCreated(
                    type="response.tool_calls.created",
                    item_id=tool_call["item_id"],
                    output_index=tool_call["output_index"],
                    tool_call={
                        "id": tool_call["id"],
                        "type": tool_call["type"],
                        "function": {
                            "name": tool_call["function"]["name"],
                            "arguments": ""
                        }
                    }
                )
                events_to_emit.append(created_event)
                tool_call_counter += 1
        
        # Process function arguments if present
        if "function" in tool_delta and "arguments" in tool_delta["function"]:
            arg_fragment = tool_delta["function"]["arguments"]
            tool_calls[index]["function"]["arguments"] += arg_fragment
            
            # Create delta event
            args_event = ToolCallArgumentsDelta(
                type="response.function_call_arguments.delta",
                item_id=tool_calls[index]["item_id"],
                output_index=tool_calls[index]["output_index"],
                delta=arg_fragment
            )
            events_to_emit.append(args_event)
            
    return tool_calls, tool_call_counter, events_to_emit

def process_tool_calls_completion(tool_calls: Dict, choice: Dict) -> List[Dict]:
    """
    Process the completion of tool calls.
    
    Args:
        tool_calls: Dictionary tracking tool calls being built
        choice: The choice object containing finish_reason
        
    Returns:
        List of events to emit
    """
    events_to_emit = []
    
    if "finish_reason" not in choice or choice["finish_reason"] != "tool_calls":
        return events_to_emit
    
    # If the finish reason is "tool_calls", emit the arguments.done events
    logger.info(f"Tool calls completed. Processing {len(tool_calls)} tool calls.")
    
    for index, tool_call in tool_calls.items():
        # Log the complete tool call arguments
        logger.info(f"Tool call completed: {tool_call['function']['name']} with arguments: {tool_call['function']['arguments']}")
        
        done_event = ToolCallArgumentsDone(
            type="response.function_call_arguments.done",
            id=tool_call["item_id"],
            output_index=tool_call["output_index"],
            arguments=tool_call["function"]["arguments"]
        )
        events_to_emit.append(done_event)
    
    return events_to_emit

def update_response_with_tool_calls(response_obj: Any, tool_calls: Dict) -> Any:
    """
    Update the response object with completed tool calls.
    
    Args:
        response_obj: The response object to update
        tool_calls: Dictionary tracking tool calls
        
    Returns:
        Updated response object
    """
    for tool_call in tool_calls.values():
        response_obj.output.append({
            "id": tool_call["item_id"],
            "type": "tool_call",
            "function": {
                "name": tool_call["function"]["name"],
                "arguments": tool_call["function"]["arguments"]
            }
        })
    
    return response_obj

def process_tool_responses_in_input(input_data: List[Any]) -> List[Dict]:
    """
    Process tool responses from the input data and convert them to chat.completions format.
    
    Args:
        input_data: List of input items which may include tool responses
        
    Returns:
        List of messages in chat.completions format
    """
    messages = []
    logger = logging.getLogger("tools_manager")
    
    for i, item in enumerate(input_data):
        if isinstance(item, dict):
            if item.get("type") == "message" and item.get("role") == "user":
                # Add user message
                content = ""
                if "content" in item:
                    for j, content_item in enumerate(item["content"]):
                        if isinstance(content_item, dict) and content_item.get("type") == "input_text":
                            content = content_item.get("text", "")
                user_message = {"role": "user", "content": content}
                messages.append(user_message)
                logger.info(f"User message: {content[:100]}...")
                
            elif item.get("type") == "function_call_output":
                # Add tool output
                logger.info(f"Tool response: call_id={item.get('call_id')}, output={item.get('output', '')[:50]}...")
                tool_message = {
                    "role": "tool",
                    "tool_call_id": item.get("call_id"),
                    "content": item.get("output", "")
                }
                messages.append(tool_message)
        elif isinstance(item, str):
            # Simple string input
            messages.append({"role": "user", "content": item})
            logger.info(f"User message (string): {item[:100]}...")
    
    return messages
