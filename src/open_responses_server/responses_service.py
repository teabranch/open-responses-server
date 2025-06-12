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
    logger.info(f"Request: model={request_data.get('model')}, " +
                f"tools={len(request_data.get('tools', []))}, " +
                f"has_instructions={'instructions' in request_data}")

    chat_request = {
        "model": request_data.get("model"),
        "temperature": request_data.get("temperature", 1.0),
        "top_p": request_data.get("top_p", 1.0),
        "stream": request_data.get("stream", False),
    }

    if "max_output_tokens" in request_data:
        chat_request["max_tokens"] = request_data["max_output_tokens"]

    messages = []
    previous_response_id = request_data.get("previous_response_id")
    if previous_response_id and previous_response_id in conversation_history:
        logger.info(f"Loading conversation history from previous_response_id: {previous_response_id}")
        messages = conversation_history[previous_response_id].copy()
    
    if "instructions" in request_data:
        has_system_message = any(msg.get("role") == "system" for msg in messages)
        if not has_system_message:
            messages.append({"role": "system", "content": request_data["instructions"]})
        else:
            for msg in messages:
                if msg.get("role") == "system":
                    msg["content"] = request_data["instructions"]
                    break
    
    if "input" in request_data and request_data["input"]:
        for item in request_data["input"]:
            if isinstance(item, dict):
                if item.get("type") == "message" and item.get("role") == "user":
                    content = ""
                    if "content" in item:
                        for content_item in item["content"]:
                            if isinstance(content_item, dict) and content_item.get("type") in ["input_text", "text"]:
                                content += content_item.get("text", "")
                            elif isinstance(content_item, str):
                                content += content_item
                    messages.append({"role": "user", "content": content})
                elif item.get("type") == "function_call_output":
                    call_id = item.get("call_id")
                    has_matching_tool_call = any(
                        tool_call.get("id") == call_id
                        for msg in messages if msg.get("role") == "assistant" and "tool_calls" in msg
                        for tool_call in msg["tool_calls"]
                    )
                    if has_matching_tool_call:
                        messages.append({"role": "tool", "tool_call_id": call_id, "content": item.get("output", "")})
                    else:
                        tool_name = item.get("name", "unknown_tool")
                        messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{"id": call_id, "type": "function", "function": {"name": tool_name, "arguments": item.get("arguments", "{}")}}]
                        })
                        messages.append({"role": "tool", "tool_call_id": call_id, "content": item.get("output", "")})
                elif item.get("type") == "message" and item.get("role") == "assistant":
                    content = "".join(
                        content_item.get("text", "")
                        for content_item in item.get("content", []) if isinstance(content_item, dict) and content_item.get("type") == "output_text"
                    )
                    if content:
                        messages.append({"role": "assistant", "content": content})
            elif isinstance(item, str):
                messages.append({"role": "user", "content": item})
    
    if not messages or (len(messages) == 1 and messages[0]["role"] == "system"):
        messages.append({"role": "user", "content": ""})
    
    chat_request["messages"] = messages

    if "tools" in request_data and request_data["tools"]:
        chat_request["tools"] = []
        for tool in request_data["tools"]:
            if isinstance(tool, dict) and tool.get("type") == "function" and "name" in tool:
                function_data = {"name": tool["name"]}
                if "description" in tool:
                    function_data["description"] = tool["description"]
                if "parameters" in tool:
                    function_data["parameters"] = tool["parameters"]
                chat_request["tools"].append({"type": "function", "function": function_data})

    if "tool_choice" in request_data:
        chat_request["tool_choice"] = request_data["tool_choice"]
    
    for key in ["user", "metadata"]:
        if key in request_data and request_data[key] is not None:
            chat_request[key] = request_data[key]
    
    logger.info(f"Converted to chat completions: {len(messages)} messages, {len(chat_request.get('tools', []))} tools")
    return chat_request


async def process_chat_completions_stream(response, chat_request=None):
    """
    Process the streaming response from chat.completions API and convert to Responses API format.
    """
    tool_calls = {}
    response_id = f"resp_{uuid.uuid4().hex}"
    message_id = f"msg_{uuid.uuid4().hex}"
    output_text_content = ""
    
    response_obj = ResponseModel(id=response_id, created_at=current_timestamp(), model="", output=[])
    
    yield f"data: {json.dumps(ResponseCreated(type='response.created', response=response_obj).dict())}\n\n"
    yield f"data: {json.dumps(ResponseInProgress(type='response.in_progress', response=response_obj).dict())}\n\n"
    
    async for chunk in response.aiter_lines():
        if not chunk.strip() or chunk.strip() == "data: [DONE]":
            continue

        if chunk.startswith("data: "):
            chunk = chunk[6:]
            
        try:
            data = json.loads(chunk)
            if "model" in data and not response_obj.model:
                response_obj.model = data["model"]
            
            if "choices" in data and data["choices"]:
                choice = data["choices"][0]
                delta = choice.get("delta", {})

                if "tool_calls" in delta and delta["tool_calls"]:
                    for tool_delta in delta["tool_calls"]:
                        index = tool_delta.get("index", 0)
                        if index not in tool_calls:
                            tool_calls[index] = {
                                "id": tool_delta.get("id", f"call_{uuid.uuid4().hex}"),
                                "type": tool_delta.get("type", "function"),
                                "function": {"name": "", "arguments": ""},
                                "item_id": f"tool_call_{uuid.uuid4().hex}",
                                "output_index": len(tool_calls)
                            }

                        if "function" in tool_delta:
                            if "name" in tool_delta["function"]:
                                tool_calls[index]["function"]["name"] = tool_delta["function"]["name"]
                                response_obj.output.append({
                                    "call_id": tool_calls[index]["id"],
                                    "name": tool_calls[index]["function"]["name"],
                                    "type": "function_call",
                                    "id": tool_calls[index]["id"],
                                    "status": "in_progress" if mcp_manager.is_mcp_tool(tool_calls[index]["function"]["name"]) else "ready"
                                })
                                yield f"data: {json.dumps(ResponseInProgress(type='response.in_progress', response=response_obj).dict())}\n\n"
                            if "arguments" in tool_delta["function"]:
                                arg_fragment = tool_delta["function"]["arguments"]
                                tool_calls[index]["function"]["arguments"] += arg_fragment
                                yield f"data: {json.dumps(ToolCallArgumentsDelta(type='response.function_call_arguments.delta', item_id=tool_calls[index]['item_id'], output_index=tool_calls[index]['output_index'], delta=arg_fragment).dict())}\n\n"

                elif "content" in delta and delta["content"] is not None:
                    output_text_content += delta["content"]
                    if not any(o.get("type") == "message" for o in response_obj.output):
                        response_obj.output.append({"id": message_id, "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": ""}]})
                    
                    yield f"data: {json.dumps(OutputTextDelta(type='response.output_text.delta', item_id=message_id, output_index=0, delta=delta['content']).dict())}\n\n"

                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                    if finish_reason == "tool_calls":
                        for index, tool_call in tool_calls.items():
                            is_mcp = mcp_manager.is_mcp_tool(tool_call["function"]["name"])
                            if not is_mcp:
                                for out in response_obj.output:
                                    if out.get("id") == tool_call["id"]:
                                        out["arguments"] = tool_call["function"]["arguments"]
                                        break
                            else:
                                # This case is handled below after execution
                                pass
                                
                    if finish_reason == "function_call" or finish_reason == "tool_calls":
                         for index, tool_call in tool_calls.items():
                            tool_name = tool_call["function"]["name"]
                            if mcp_manager.is_mcp_tool(tool_name):
                                try:
                                    args = json.loads(tool_call["function"]["arguments"])
                                    result = await mcp_manager.execute_mcp_tool(tool_name, args)
                                    response_obj.output.append({"id": tool_call["id"], "type": "function_call_output", "call_id": tool_call["id"], "output": result})
                                except Exception as e:
                                    result = {"error": str(e)}
                                    response_obj.output.append({"id": tool_call["id"], "type": "function_call_output", "call_id": tool_call["id"], "output": result})
                    
                    response_obj.status = "completed"
                    if not response_obj.output and output_text_content:
                        response_obj.output = [{"id": message_id, "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": output_text_content}]}]

                    # Save to history
                    if chat_request:
                        messages = chat_request.get("messages", [])
                        if output_text_content:
                            messages.append({"role": "assistant", "content": output_text_content})
                        if tool_calls:
                            messages.append({
                                "role": "assistant", "content": None,
                                "tool_calls": [{"id": tc["id"], "type": "function", "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}} for tc in tool_calls.values()]
                            })
                            for tc in tool_calls.values():
                                if mcp_manager.is_mcp_tool(tc["function"]["name"]):
                                    # This assumes result is available, might need rework
                                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": "{\"status\": \"executed\"}"}) # Placeholder
                        conversation_history[response_id] = messages
                        if len(conversation_history) > MAX_CONVERSATION_HISTORY:
                            oldest_key = min(conversation_history.keys())
                            del conversation_history[oldest_key]
                    
                    yield f"data: {json.dumps(ResponseCompleted(type='response.completed', response=response_obj).dict())}\n\n"
                    return
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from chunk: {chunk}")
            continue

    # Final cleanup if loop finishes without finish_reason
    if response_obj.status != "completed":
        response_obj.status = "completed"
        if not response_obj.output and output_text_content:
             response_obj.output = [{"id": message_id, "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": output_text_content}]}]
        yield f"data: {json.dumps(ResponseCompleted(type='response.completed', response=response_obj).dict())}\n\n" 