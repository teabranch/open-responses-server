import json
import uuid
import time
from collections import OrderedDict
from typing import Dict, List, Any

from open_responses_server.common.config import logger, MAX_CONVERSATION_HISTORY
from open_responses_server.common.mcp_manager import mcp_manager, serialize_tool_result
from open_responses_server.models.responses_models import (
    ResponseModel, ResponseCreated, ResponseInProgress, ResponseCompleted,
    ToolCallsCreated, ToolCallArgumentsDelta, ToolCallArgumentsDone, OutputTextDelta,
    OutputItemAdded, OutputItemDone, OutputTextDone
)

# Global dictionary to store conversation history by response ID
conversation_history: Dict[str, List[Dict[str, Any]]] = {}

# Cache reasoning_content (CoT) keyed by tool call_id for passback.
# Keep a bounded insertion-ordered cache so recent tool-call chains can feed
# reasoning back into the next request without unbounded growth.
reasoning_content_cache: OrderedDict[str, str] = OrderedDict()

def current_timestamp() -> int:
    return int(time.time())


def _stringify_tool_output(output: Any) -> str:
    """Normalize tool output into the string payload chat.completions expects."""
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    try:
        return serialize_tool_result(output)
    except TypeError:
        return str(output)


def _cache_reasoning_content(call_id: str, reasoning_content: str, max_entries: int = 200) -> None:
    """Store reasoning content by call_id and evict oldest entries when bounded."""
    if not call_id or not reasoning_content:
        return

    reasoning_content_cache[call_id] = reasoning_content
    reasoning_content_cache.move_to_end(call_id)

    while len(reasoning_content_cache) > max_entries:
        reasoning_content_cache.popitem(last=False)

def validate_message_sequence(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validate and fix the message sequence to ensure tool messages have preceding assistant messages with tool_calls.
    Also deduplicates tool messages with the same tool_call_id.
    This prevents OpenAI API errors like "messages with role 'tool' must be a response to a preceeding message with 'tool_calls'."
    """
    validated_messages = []
    orphaned_tool_messages = []
    seen_tool_call_ids = set()
    
    for i, message in enumerate(messages):
        if message.get("role") == "tool":
            tool_call_id = message.get("tool_call_id")
            
            # Check for duplicate tool call IDs
            if tool_call_id in seen_tool_call_ids:
                logger.warning(f"Duplicate tool message at position {i}: call_id={tool_call_id} - skipping duplicate")
                continue
            
            # Check if the previous message is an assistant with tool_calls
            has_preceding_tool_call = False
            
            # Look backward from current position for matching tool call
            for j in range(len(validated_messages) - 1, -1, -1):
                prev_msg = validated_messages[j]
                if prev_msg.get("role") == "assistant" and "tool_calls" in prev_msg:
                    for tool_call in prev_msg["tool_calls"]:
                        if tool_call.get("id") == tool_call_id:
                            has_preceding_tool_call = True
                            break
                    if has_preceding_tool_call:
                        break
                # Stop looking if we hit another assistant message without tool_calls
                elif prev_msg.get("role") == "assistant":
                    break
            
            if has_preceding_tool_call:
                validated_messages.append(message)
                seen_tool_call_ids.add(tool_call_id)
                logger.debug(f"Valid tool message at position {i}: call_id={tool_call_id}")
            else:
                logger.warning(f"Orphaned tool message at position {i}: call_id={tool_call_id} - no preceding assistant with matching tool_call")
                orphaned_tool_messages.append(message)
        else:
            validated_messages.append(message)
    
    if orphaned_tool_messages:
        logger.warning(f"Removed {len(orphaned_tool_messages)} orphaned tool messages to prevent API validation error")
        
    logger.info(f"Message validation: {len(messages)} -> {len(validated_messages)} messages")
    return validated_messages

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
        logger.info(f"Processing {len(request_data['input'])} input items")
        for i, item in enumerate(request_data["input"]):
            if isinstance(item, dict):
                item_type = item.get("type")
                item_role = item.get("role")

                # Handle message items
                if item_type == "message":
                    if item_role == "user":
                        content = ""
                        if "content" in item:
                            for content_item in item["content"]:
                                if isinstance(content_item, dict) and content_item.get("type") in ("input_text", "text"):
                                    content += content_item.get("text", "")
                                elif isinstance(content_item, str):
                                    content += content_item
                        messages.append({"role": "user", "content": content})
                        logger.info(f"User message: {content[:100]}...")

                    elif item_role == "developer":
                        # Developer messages → system role in chat completions
                        content = ""
                        if "content" in item:
                            for content_item in item["content"]:
                                if isinstance(content_item, dict) and content_item.get("type") in ("input_text", "text"):
                                    content += content_item.get("text", "")
                                elif isinstance(content_item, str):
                                    content += content_item
                        if content:
                            # Check if system message already exists
                            has_system = any(msg.get("role") == "system" for msg in messages)
                            if has_system:
                                # Append to existing system message
                                for msg in messages:
                                    if msg.get("role") == "system":
                                        msg["content"] += "\n" + content
                                        break
                            else:
                                messages.append({"role": "system", "content": content})
                            logger.info(f"Developer message (as system): {content[:100]}...")

                    elif item_role == "assistant":
                        content = ""
                        if "content" in item and isinstance(item["content"], list):
                            for content_item in item["content"]:
                                if isinstance(content_item, dict) and content_item.get("type") == "output_text":
                                    content += content_item.get("text", "")
                        if content:
                            messages.append({"role": "assistant", "content": content})
                            logger.info(f"Assistant message: {content[:100]}...")

                # Handle function_call items (assistant's tool calls sent back by client)
                elif item_type == "function_call":
                    call_id = item.get("call_id", item.get("id", f"call_{uuid.uuid4().hex}"))
                    tool_name = item.get("name", "")
                    arguments = item.get("arguments", "{}")

                    # Look up cached reasoning_content for CoT passback
                    cached_reasoning = reasoning_content_cache.get(call_id, "")
                    if cached_reasoning:
                        logger.info(f"[INPUT] function_call: name={tool_name} call_id={call_id} +reasoning={len(cached_reasoning)} chars")
                    else:
                        logger.info(f"[INPUT] function_call: name={tool_name} call_id={call_id}")

                    # Group consecutive function_calls into one assistant message
                    # Check if the last message is an assistant with tool_calls
                    if messages and messages[-1].get("role") == "assistant" and "tool_calls" in messages[-1]:
                        messages[-1]["tool_calls"].append({
                            "id": call_id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": arguments}
                        })
                        # Merge reasoning: use longest (first call's reasoning covers all)
                        if cached_reasoning and not messages[-1].get("reasoning_content"):
                            messages[-1]["reasoning_content"] = cached_reasoning
                    else:
                        assistant_msg = {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": call_id,
                                "type": "function",
                                "function": {"name": tool_name, "arguments": arguments}
                            }]
                        }
                        if cached_reasoning:
                            assistant_msg["reasoning_content"] = cached_reasoning
                        messages.append(assistant_msg)

                # Handle function_call_output items (tool results)
                elif item_type == "function_call_output":
                    call_id = item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}"
                    output = _stringify_tool_output(item.get("output", ""))
                    logger.info(f"[INPUT] function_call_output: call_id={call_id} output_len={len(str(output))}")

                    # Check if we have a corresponding assistant message with a matching tool call
                    has_matching_tool_call = False
                    for msg in messages:
                        if msg.get("role") == "assistant" and "tool_calls" in msg:
                            for tool_call in msg["tool_calls"]:
                                if tool_call.get("id") == call_id:
                                    has_matching_tool_call = True
                                    break

                    if has_matching_tool_call:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": output
                        })
                        logger.info(f"[INPUT] Added tool response for call_id={call_id}")
                    else:
                        # Fallback: create synthetic assistant + tool message
                        tool_name = item.get("name", "unknown_tool")
                        if tool_name and tool_name != "unknown_tool":
                            messages.append({
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
                            })
                            messages.append({
                                "role": "tool",
                                "tool_call_id": call_id,
                                "content": output
                            })
                            logger.info(f"[INPUT] Created synthetic assistant+tool for {tool_name} call_id={call_id}")
                        else:
                            logger.warning(f"[INPUT] Skipping orphaned function_call_output: call_id={call_id}, no tool name")

            elif isinstance(item, str):
                messages.append({"role": "user", "content": item})
                logger.info(f"User message (string): {item[:100]}...")
    
    # If we only have a system message or no messages at all, add an empty user message
    if not messages or (len(messages) == 1 and messages[0]["role"] == "system"):
        messages.append({"role": "user", "content": ""})
    
    chat_request["messages"] = messages

    # Convert tools - log each tool being processed
    if "tools" in request_data and request_data["tools"]:
        chat_request["tools"] = []
        
        logger.info(f"[TOOL-CONVERSION] Processing {len(request_data['tools'])} tools from request_data")
        
        for i, tool in enumerate(request_data["tools"]):
            try:
                logger.debug(f"[TOOL-CONVERSION] Processing tool {i}: {tool}")
                if not isinstance(tool, dict) or "type" not in tool or tool.get("type") != "function":
                    logger.warning(f"[TOOL-CONVERSION] Skipping tool {i}: not a function type or invalid format")
                    continue
                    
                function_obj = tool
                if not isinstance(function_obj, dict) or "name" not in function_obj:
                    logger.warning(f"[TOOL-CONVERSION] Skipping tool {i}: missing name or invalid function object")
                    continue
                
                function_data = {
                    "name": function_obj["name"],
                }
                
                # Log tool information
                logger.info(f"[TOOL-CONVERSION] Converting Tool {i}: {function_data['name']}")
                
                if "description" in function_obj:
                    function_data["description"] = function_obj["description"]
                    
                if "parameters" in function_obj:
                    function_data["parameters"] = function_obj["parameters"]
                
                chat_request["tools"].append({
                    "type": "function",
                    "function": function_data
                })
                
                # Check if this is an MCP tool
                is_mcp = any(func.get("name") == function_data["name"] for func in mcp_manager.mcp_functions_cache)
                logger.info(f"[TOOL-CONVERSION] Tool '{function_data['name']}': is_mcp={is_mcp}")
                
            except Exception as e:
                logger.error(f"[TOOL-CONVERSION] Error processing tool {i}: {str(e)}")
        
        logger.info(f"[TOOL-CONVERSION] Successfully converted {len(chat_request['tools'])} tools to chat_request format")
    else:
        logger.info("[TOOL-CONVERSION] No tools found in request_data")
    
    # Handle tool_choice
    if "tool_choice" in request_data:
        chat_request["tool_choice"] = request_data["tool_choice"]
    
    # Handle reasoning parameter - only include if it has actual values (not just None/null)
    if "reasoning" in request_data:
        reasoning = request_data["reasoning"]
        # Only include reasoning if it has actual non-null values
        if isinstance(reasoning, dict) and any(v is not None for v in reasoning.values()):
            chat_request["reasoning"] = reasoning
            logger.info(f"[TOOL-CONVERSION] Including reasoning parameter: {reasoning}")
        else:
            logger.info("[TOOL-CONVERSION] Skipping reasoning parameter (all values are null)")
    
    # Add optional parameters if they exist
    for key in ["user", "metadata"]:
        if key in request_data and request_data[key] is not None:
            chat_request[key] = request_data[key]
    
    # Final cleanup - ensure reasoning is completely removed if it has null values
    if "reasoning" in chat_request:
        reasoning = chat_request["reasoning"]
        if isinstance(reasoning, dict) and all(v is None for v in reasoning.values()):
            chat_request.pop("reasoning", None)
            logger.info("[TOOL-CONVERSION] Removed reasoning parameter with all null values from chat_request")
    
    # Log final chat_request for debugging
    logger.info(f"[TOOL-CONVERSION] Final chat_request keys: {list(chat_request.keys())}")
    if "reasoning" in chat_request:
        logger.warning(f"[TOOL-CONVERSION] WARNING: reasoning parameter still present: {chat_request['reasoning']}")
    
    # Validate message sequence before sending to API
    validated_messages = validate_message_sequence(messages)
    chat_request["messages"] = validated_messages
    
    logger.info(f"Converted to chat completions: {len(validated_messages)} messages, {len(chat_request.get('tools', []))} tools")
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
    reasoning_content = ""  # Accumulate reasoning/CoT from model for passback
    request_start_time = time.time()
    last_chunk_time = request_start_time
    logger.info(f"[STREAM-START] response_id={response_id} message_id={message_id}")
    
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

    def ensure_tool_call_added(tool_call: Dict[str, Any]) -> str | None:
        """Emit output_item.added once per tool call after its name becomes available."""
        tool_name = tool_call["function"]["name"]
        if tool_call.get("added_emitted") or not tool_name:
            return None

        logger.info(f"Tool call created: {tool_name}")
        is_mcp = mcp_manager.is_mcp_tool(tool_name)
        logger.info(f"[TOOL-CALL-CREATED] Tool '{tool_name}': is_mcp={is_mcp}, status=in_progress")

        fc_item = {
            "arguments": "",
            "call_id": tool_call["id"],
            "name": tool_name,
            "type": "function_call",
            "id": tool_call["id"],
            "status": "in_progress"
        }
        response_obj.output.append(fc_item)
        tool_call["added_emitted"] = True

        item_added_event = OutputItemAdded(
            output_index=tool_call["output_index"],
            item=fc_item
        )
        logger.info(f"Emitting output_item.added for '{tool_name}'")
        return f"data: {json.dumps(item_added_event.dict())}\n\n"

    try:
        async for chunk in response.aiter_lines():
            chunk_counter += 1
            now = time.time()
            chunk_gap = now - last_chunk_time
            last_chunk_time = now
            if chunk_gap > 2.0:
                logger.info(
                    f"[STREAM-TIMING] response_id={response_id} "
                    f"chunk_gap={chunk_gap:.1f}s chunk={chunk_counter}"
                )
            if not chunk.strip():
                continue
                
            # Handle [DONE] message
            if chunk.strip() == "data: [DONE]" or chunk.strip() == "[DONE]":
                total_time = time.time() - request_start_time
                logger.info(
                    f"[STREAM-DONE] response_id={response_id} "
                    f"chunks={chunk_counter} total_time={total_time:.1f}s "
                    f"status={response_obj.status}"
                )
                
                # If we haven't already completed the response, do it now
                if response_obj.status != "completed":
                    final_text = output_text_content or ""

                    # Emit text closing events if we had text content
                    if final_text:
                        yield f"data: {json.dumps({'type': 'response.output_text.done', 'item_id': message_id, 'output_index': 0, 'content_index': 0, 'text': final_text})}\n\n"
                        yield f"data: {json.dumps({'type': 'response.content_part.done', 'item_id': message_id, 'output_index': 0, 'content_index': 0, 'part': {'type': 'output_text', 'text': final_text, 'annotations': []}})}\n\n"

                    final_msg_item = {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": final_text, "annotations": []}]
                    }

                    # Emit output_item.done if we have text
                    if final_text:
                        yield f"data: {json.dumps({'type': 'response.output_item.done', 'output_index': 0, 'item': final_msg_item})}\n\n"

                    response_obj.output = [final_msg_item] if final_text else response_obj.output
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
                                    tool_call_id = tool_delta.get("id", f"call_{uuid.uuid4().hex}")
                                    tool_calls[index] = {
                                        "id": tool_call_id,
                                        "type": tool_delta.get("type", "function"),
                                        "function": {
                                            "name": tool_delta.get("function", {}).get("name", ""),
                                            "arguments": "",
                                        },
                                        "output_index": tool_call_counter,
                                        "added_emitted": False,
                                    }
                                    tool_call_counter += 1

                                tool_call = tool_calls[index]

                                if "function" in tool_delta and "name" in tool_delta["function"]:
                                    tool_call["function"]["name"] = tool_delta["function"]["name"]
                                    item_added_payload = ensure_tool_call_added(tool_call)
                                    if item_added_payload:
                                        yield item_added_payload
                                
                                # Process function arguments if present
                                if "function" in tool_delta and "arguments" in tool_delta["function"]:
                                    arg_fragment = tool_delta["function"]["arguments"]
                                    tool_calls[index]["function"]["arguments"] += arg_fragment

                                    # Emit delta event
                                    args_event = ToolCallArgumentsDelta(
                                        type="response.function_call_arguments.delta",
                                        item_id=tool_calls[index]["id"],
                                        output_index=tool_calls[index]["output_index"],
                                        delta=arg_fragment
                                    )

                                    yield f"data: {json.dumps(args_event.dict())}\n\n"
                        
                        # Handle content (text)
                        elif "content" in delta and delta["content"] is not None:
                            content_delta = delta["content"]
                            output_text_content += content_delta

                            # On first text chunk, emit output_item.added + content_part.added
                            if not response_obj.output or not any(
                                o.get("type") == "message" for o in response_obj.output
                            ):
                                msg_item = {
                                    "id": message_id,
                                    "type": "message",
                                    "role": "assistant",
                                    "status": "in_progress",
                                    "content": []
                                }
                                response_obj.output.append(msg_item)
                                # output_item.added
                                yield f"data: {json.dumps({'type': 'response.output_item.added', 'output_index': 0, 'item': msg_item})}\n\n"
                                # content_part.added
                                yield f"data: {json.dumps({'type': 'response.content_part.added', 'item_id': message_id, 'output_index': 0, 'content_index': 0, 'part': {'type': 'output_text', 'text': '', 'annotations': []}})}\n\n"

                            # Emit text delta event
                            text_event = OutputTextDelta(
                                type="response.output_text.delta",
                                item_id=message_id,
                                output_index=0,
                                content_index=0,
                                delta=content_delta
                            )
                            yield f"data: {json.dumps(text_event.dict())}\n\n"

                        # Accumulate reasoning_content (CoT) from model for passback
                        if "reasoning_content" in delta and delta["reasoning_content"] is not None:
                            reasoning_content += delta["reasoning_content"]

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
                                    
                                logger.info(f"[TOOL-EXECUTE] Processing tool '{tool_name}' with args: {args}")
                                
                                # Check if this is an MCP tool or a non-MCP tool
                                if mcp_manager.is_mcp_tool(tool_name):
                                    logger.info(f"[TOOL-EXECUTE] Executing MCP tool: {tool_name}")
                                    # Execute MCP tool
                                    try:
                                        result = await mcp_manager.execute_mcp_tool(tool_name, args)
                                        logger.info(f"[TOOL-EXECUTE] ✓ MCP tool '{tool_name}' executed successfully")
                                        logger.debug(f"[TOOL-EXECUTE] MCP tool '{tool_name}' result: {result}")
                                    except Exception as e:
                                        result = {"error": str(e)}
                                        logger.error(f"[TOOL-EXECUTE] ✗ MCP tool '{tool_name}' failed: {e}")
                                    
                                    # Append as function_call_output
                                    response_obj.output.append({
                                        "id": tool_call["id"],
                                        "type": "function_call_output",
                                        "call_id": tool_call["id"],
                                        "output": serialize_tool_result(result)
                                    })
                                    
                                    # Convert result to JSON, with fallback to string if needed
                                    try:
                                        text = serialize_tool_result(result)
                                    except TypeError:
                                        text = serialize_tool_result(str(result))
                                        
                                    text_event = OutputTextDelta(
                                        type="response.output_text.delta",
                                        item_id=tool_call["id"],
                                        output_index=0,
                                        content_index=0,
                                        delta=text
                                    )
                                    yield f"data: {json.dumps(text_event.dict())}\n\n"
                                else:
                                    # For non-MCP tools, send the function call back to the client in Responses API format
                                    logger.info(f"[TOOL-EXECUTE] Forwarding non-MCP tool call to client: {tool_name}")

                                    # Include the function call in the response
                                    response_obj.output.append({
                                        "id": tool_call["id"],
                                        "type": "function_call",
                                        "name": tool_name,
                                        "arguments": tool_call["function"]["arguments"],
                                        "call_id": tool_call["id"],
                                        "status": "completed"
                                    })
                                
                                # Cache reasoning for CoT passback
                                if reasoning_content:
                                    _cache_reasoning_content(tool_call["id"], reasoning_content)
                                    logger.info(f"[COT-PASSBACK] Cached reasoning ({len(reasoning_content)} chars) for call_id={tool_call['id']}")

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
                                    # Preserve reasoning_content for CoT passback on tool call turns
                                    if reasoning_content:
                                        assistant_message["reasoning_content"] = reasoning_content
                                        logger.info(f"[COT-PASSBACK] Stored {len(reasoning_content)} chars of reasoning_content in history")
                                    messages.append(assistant_message)

                                    # Add the tool response for immediate tools
                                    if mcp_manager.is_mcp_tool(tool_name):
                                        # For MCP tools, also add the tool response
                                        tool_message = {
                                            "role": "tool",
                                            "tool_call_id": tool_call["id"],
                                            "content": serialize_tool_result(result)
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
                            logger.info(f"[TOOL-CALLS-FINISH] Processing {len(tool_calls)} tool calls")
                            for index, tool_call in tool_calls.items():
                                # Log the complete tool call arguments
                                logger.info(f"[TOOL-CALLS-FINISH] Tool call completed: {tool_call['function']['name']} with arguments: {tool_call['function']['arguments']}")
                                
                                # Check if this is an MCP tool or a user-defined tool
                                is_mcp = mcp_manager.is_mcp_tool(tool_call["function"]["name"])
                                
                                logger.info(f"[TOOL-CALLS-FINISH] Tool '{tool_call['function']['name']}': is_mcp={is_mcp}")
                                
                                # Emit the arguments.done event (same for MCP and non-MCP)
                                done_event = ToolCallArgumentsDone(
                                    type="response.function_call_arguments.done",
                                    item_id=tool_call["id"],
                                    output_index=tool_call["output_index"],
                                    arguments=tool_call["function"]["arguments"]
                                )
                                logger.info(f"Emitting arguments.done for '{tool_call['function']['name']}'")
                                yield f"data: {json.dumps(done_event.dict())}\n\n"

                                # Update the function_call item in output: set final arguments and status
                                for output_item in response_obj.output:
                                    if output_item.get("id") == tool_call["id"] and output_item.get("type") == "function_call":
                                        output_item["arguments"] = tool_call["function"]["arguments"]
                                        output_item["status"] = "completed"
                                        break

                                # Emit response.output_item.done with completed status
                                done_fc_item = {
                                    "arguments": tool_call["function"]["arguments"],
                                    "call_id": tool_call["id"],
                                    "name": tool_call["function"]["name"],
                                    "type": "function_call",
                                    "id": tool_call["id"],
                                    "status": "completed"
                                }
                                item_done_event = OutputItemDone(
                                    output_index=tool_call["output_index"],
                                    item=done_fc_item
                                )
                                logger.info(f"Emitting output_item.done for '{tool_call['function']['name']}'")
                                yield f"data: {json.dumps(item_done_event.dict())}\n\n"

                                # For MCP tools, execute them immediately
                                if is_mcp:
                                    logger.info(f"[TOOL-CALLS-FINISH] Executing MCP tool '{tool_call['function']['name']}'")

                                    try:
                                        args = json.loads(tool_call["function"]["arguments"])
                                    except Exception:
                                        args = {}

                                    try:
                                        result = await mcp_manager.execute_mcp_tool(tool_call["function"]["name"], args)
                                        logger.info(f"[TOOL-CALLS-FINISH] MCP tool '{tool_call['function']['name']}' executed successfully")
                                    except Exception as e:
                                        result = {"error": str(e)}
                                        logger.error(f"[TOOL-CALLS-FINISH] MCP tool '{tool_call['function']['name']}' failed: {e}")

                                    response_obj.output.append({
                                        "id": tool_call["id"],
                                        "type": "function_call_output",
                                        "call_id": tool_call["id"],
                                        "output": serialize_tool_result(result)
                                    })

                                    try:
                                        text = serialize_tool_result(result)
                                    except TypeError:
                                        text = serialize_tool_result(str(result))

                                    text_event = OutputTextDelta(
                                        type="response.output_text.delta",
                                        item_id=tool_call["id"],
                                        output_index=0,
                                        content_index=0,
                                        delta=text
                                    )
                                    yield f"data: {json.dumps(text_event.dict())}\n\n"
                                    logger.info(f"[TOOL-CALLS-FINISH] Added function_call_output for MCP tool '{tool_call['function']['name']}'")
                                else:
                                    logger.info(f"[TOOL-CALLS-FINISH] Non-MCP tool '{tool_call['function']['name']}' completed, client will execute")

                            # Cache reasoning_content keyed by call_ids for CoT passback
                            # When Codex CLI sends these call_ids back, we inject the reasoning
                            if reasoning_content:
                                for tc in tool_calls.values():
                                    _cache_reasoning_content(tc["id"], reasoning_content)
                                logger.info(f"[COT-PASSBACK] Cached reasoning ({len(reasoning_content)} chars) for {len(tool_calls)} call_ids")

                            # After processing all tool calls, complete the response
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
                                    } for tool_call in tool_calls.values()]
                                }
                                # Preserve reasoning_content for CoT passback on tool call turns
                                if reasoning_content:
                                    assistant_message["reasoning_content"] = reasoning_content
                                    logger.info(f"[COT-PASSBACK] Stored {len(reasoning_content)} chars of reasoning_content in history")
                                messages.append(assistant_message)

                                # Add tool responses for executed MCP tools
                                for tool_call in tool_calls.values():
                                    if mcp_manager.is_mcp_tool(tool_call["function"]["name"]):
                                        # Find the result in the response output
                                        for output_item in response_obj.output:
                                            if (output_item.get("type") == "function_call_output" and 
                                                output_item.get("call_id") == tool_call["id"]):
                                                tool_message = {
                                                    "role": "tool",
                                                    "tool_call_id": tool_call["id"],
                                                    "content": serialize_tool_result(output_item.get("output", ""))
                                                }
                                                messages.append(tool_message)
                                                break
                                
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
                            
                            logger.info(f"[TOOL-CALLS-FINISH] Completed processing all tool calls, final output has {len(response_obj.output)} items")
                            logger.info(f"Emitting completed event after tool_calls: {completed_event}")
                            yield f"data: {json.dumps(completed_event.dict())}\n\n"
                            return  # End streaming after tool processing
                        
                        # If the finish reason is "stop", emit the completed event
                        if choice["finish_reason"] == "stop":
                            logger.info("Received stop finish reason")

                            final_text = output_text_content or ""
                            has_message_output = any(
                                output_item.get("type") == "message"
                                for output_item in response_obj.output
                            )

                            if not has_message_output:
                                added_msg_item = {
                                    "id": message_id,
                                    "type": "message",
                                    "role": "assistant",
                                    "status": "in_progress",
                                    "content": []
                                }
                                yield f"data: {json.dumps({'type': 'response.output_item.added', 'output_index': 0, 'item': added_msg_item})}\n\n"
                                yield f"data: {json.dumps({'type': 'response.content_part.added', 'item_id': message_id, 'output_index': 0, 'content_index': 0, 'part': {'type': 'output_text', 'text': '', 'annotations': []}})}\n\n"

                            # Emit text closing events: output_text.done, content_part.done, output_item.done
                            if final_text:
                                # output_text.done
                                yield f"data: {json.dumps({'type': 'response.output_text.done', 'item_id': message_id, 'output_index': 0, 'content_index': 0, 'text': final_text})}\n\n"
                                # content_part.done
                                yield f"data: {json.dumps({'type': 'response.content_part.done', 'item_id': message_id, 'output_index': 0, 'content_index': 0, 'part': {'type': 'output_text', 'text': final_text, 'annotations': []}})}\n\n"

                            # Build the final message item
                            final_msg_item = {
                                "id": message_id,
                                "type": "message",
                                "role": "assistant",
                                "status": "completed",
                                "content": [{"type": "output_text", "text": final_text, "annotations": []}]
                            }

                            # output_item.done
                            yield f"data: {json.dumps({'type': 'response.output_item.done', 'output_index': 0, 'item': final_msg_item})}\n\n"

                            logger.info(f"Response completed with text: {final_text[:100]}...")

                            response_obj.status = "completed"
                            response_obj.output = [final_msg_item]
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
                                    "content": final_text
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
        total_time = time.time() - request_start_time
        logger.error(
            f"[STREAM-ERROR] response_id={response_id} "
            f"error={str(e)} total_time={total_time:.1f}s "
            f"chunks={chunk_counter}"
        )
        # Emit a completion event if we haven't already
        if response_obj.status != "completed":
            response_obj.status = "completed"
            response_obj.error = {"message": str(e)}
            
            completed_event = ResponseCompleted(
                type="response.completed",
                response=response_obj
            )
            
            yield f"data: {json.dumps(completed_event.dict())}\n\n" 
