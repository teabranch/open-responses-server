import json
import uuid
import time
from typing import Dict, List, Any

from open_responses_server.common.config import logger, MAX_CONVERSATION_HISTORY
from open_responses_server.common.mcp_manager import mcp_manager
from open_responses_server.models.responses_models import (
    ResponseModel, ResponseCreated, ResponseInProgress, ResponseCompleted,
    ToolCallsCreated, ToolCallArgumentsDelta, ToolCallArgumentsDone, OutputTextDelta
)

# Global dictionary to store conversation history by response ID
conversation_history: Dict[str, List[Dict[str, Any]]] = {}

def current_timestamp() -> int:
    return int(time.time())

def convert_responses_to_chat_completions(request_data: dict) -> dict:
    """
    Convert a request in Responses API format to chat.completions API format.
    """
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
                                        is_mcp = mcp_manager.is_mcp_tool(tool_call["function"]["name"])
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
                                    "content": [{"type": "output_text", "text": output_text_content or "(No update)"}]
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
                                if mcp_manager.is_mcp_tool(tool_name):
                                    logger.info(f"Executing MCP tool: {tool_name}")
                                    # Execute MCP tool
                                    try:
                                        result = await mcp_manager.execute_mcp_tool(tool_name, args)
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
                                    if mcp_manager.is_mcp_tool(tool_name):
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
                                is_mcp = mcp_manager.is_mcp_tool(tool_call["function"]["name"])
                                
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