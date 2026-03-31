"""
Comprehensive tests for the responses_service module.

Covers:
  - validate_message_sequence
  - convert_responses_to_chat_completions
  - process_chat_completions_stream (async generator)
"""

import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from open_responses_server.responses_service import (
    convert_responses_to_chat_completions,
    validate_message_sequence,
    process_chat_completions_stream,
    conversation_history,
)


# ---------------------------------------------------------------------------
# Helper to parse SSE events yielded by process_chat_completions_stream
# ---------------------------------------------------------------------------

def parse_sse(raw: str) -> dict:
    """Strip 'data: ' prefix and trailing newlines, then parse JSON."""
    text = raw.strip()
    if text.startswith("data: "):
        text = text[6:]
    return json.loads(text)


# ===================================================================
# 1. validate_message_sequence
# ===================================================================

class TestValidateMessageSequence:
    """Tests for validate_message_sequence."""

    def test_valid_sequence_unchanged(self):
        """Valid user/assistant/tool sequence passes through unchanged."""
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "foo", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ]
        result = validate_message_sequence(messages)
        assert len(result) == 3
        assert result == messages

    def test_orphaned_tool_messages_removed(self):
        """Tool messages without a preceding assistant with matching tool_call are removed."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "call_orphan", "content": "orphan result"},
        ]
        result = validate_message_sequence(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_duplicate_tool_call_ids_deduplicated(self):
        """Second tool message with the same tool_call_id is skipped."""
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_dup", "type": "function", "function": {"name": "foo", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call_dup", "content": "first"},
            {"role": "tool", "tool_call_id": "call_dup", "content": "duplicate"},
        ]
        result = validate_message_sequence(messages)
        # Only the first tool message should remain
        tool_msgs = [m for m in result if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"] == "first"

    def test_mixed_valid_and_orphaned(self):
        """Mix of valid tool messages and orphaned ones; only valid kept."""
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_ok", "type": "function", "function": {"name": "foo", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call_ok", "content": "valid"},
            {"role": "tool", "tool_call_id": "call_missing", "content": "orphan"},
        ]
        result = validate_message_sequence(messages)
        assert len(result) == 3
        tool_msgs = [m for m in result if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "call_ok"

    def test_empty_messages(self):
        """Empty list returns empty list."""
        assert validate_message_sequence([]) == []

    def test_non_tool_messages_pass_through(self):
        """System, user, assistant messages without tool context are kept."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = validate_message_sequence(messages)
        assert len(result) == 3

    def test_multiple_tool_calls_same_assistant(self):
        """Multiple tool messages matching different tool_calls on the same assistant."""
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_a", "type": "function", "function": {"name": "foo", "arguments": "{}"}},
                    {"id": "call_b", "type": "function", "function": {"name": "bar", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_a", "content": "res_a"},
            {"role": "tool", "tool_call_id": "call_b", "content": "res_b"},
        ]
        result = validate_message_sequence(messages)
        assert len(result) == 4


# ===================================================================
# 2. convert_responses_to_chat_completions
# ===================================================================

class TestConvertResponsesToChatCompletions:
    """Tests for convert_responses_to_chat_completions."""

    def test_basic_passthrough_fields(self):
        """model, temperature, top_p, stream are passed through."""
        req = {"model": "gpt-4", "temperature": 0.5, "top_p": 0.9, "stream": True}
        result = convert_responses_to_chat_completions(req)
        assert result["model"] == "gpt-4"
        assert result["temperature"] == 0.5
        assert result["top_p"] == 0.9
        assert result["stream"] is True

    def test_max_output_tokens_conversion(self):
        """max_output_tokens is converted to max_tokens."""
        req = {"model": "m", "max_output_tokens": 1024}
        result = convert_responses_to_chat_completions(req)
        assert result["max_tokens"] == 1024
        assert "max_output_tokens" not in result

    def test_instructions_added_as_system_message(self):
        """Instructions field adds a system message."""
        req = {"model": "m", "instructions": "Be helpful."}
        result = convert_responses_to_chat_completions(req)
        msgs = result["messages"]
        system_msgs = [m for m in msgs if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "Be helpful."

    def test_instructions_replace_existing_system_message(self):
        """Instructions replace an existing system message from conversation history."""
        # Pre-populate conversation history with a system message
        conversation_history["prev_id"] = [
            {"role": "system", "content": "Old instructions."},
            {"role": "user", "content": "hi"},
        ]
        req = {
            "model": "m",
            "instructions": "New instructions.",
            "previous_response_id": "prev_id",
        }
        result = convert_responses_to_chat_completions(req)
        system_msgs = [m for m in result["messages"] if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "New instructions."

    def test_input_text_type_content(self):
        """input_text type content items are processed."""
        req = {
            "model": "m",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello from input_text"}],
                }
            ],
        }
        result = convert_responses_to_chat_completions(req)
        user_msgs = [m for m in result["messages"] if m["role"] == "user"]
        assert any("Hello from input_text" in m["content"] for m in user_msgs)

    def test_text_type_content(self):
        """text type content items are processed."""
        req = {
            "model": "m",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello from text"}],
                }
            ],
        }
        result = convert_responses_to_chat_completions(req)
        user_msgs = [m for m in result["messages"] if m["role"] == "user"]
        assert any("Hello from text" in m["content"] for m in user_msgs)

    def test_string_content_items(self):
        """String content items inside a message are concatenated."""
        req = {
            "model": "m",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": ["hello ", "world"],
                }
            ],
        }
        result = convert_responses_to_chat_completions(req)
        user_msgs = [m for m in result["messages"] if m["role"] == "user"]
        assert any("hello world" in m["content"] for m in user_msgs)

    def test_function_call_output_with_matching_tool_call(self):
        """function_call_output with an existing matching tool_call in messages."""
        conversation_history["prev_fc"] = [
            {"role": "user", "content": "do something"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_match", "type": "function", "function": {"name": "my_tool", "arguments": "{}"}},
                ],
            },
        ]
        req = {
            "model": "m",
            "previous_response_id": "prev_fc",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_match",
                    "name": "my_tool",
                    "output": "tool result here",
                }
            ],
        }
        result = convert_responses_to_chat_completions(req)
        tool_msgs = [m for m in result["messages"] if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "call_match"
        assert tool_msgs[0]["content"] == "tool result here"

    def test_function_call_output_without_matching_creates_pair(self):
        """function_call_output without matching tool_call creates assistant+tool messages."""
        req = {
            "model": "m",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_new",
                    "name": "new_tool",
                    "output": "new result",
                    "arguments": '{"x": 1}',
                }
            ],
        }
        result = convert_responses_to_chat_completions(req)
        msgs = result["messages"]
        # Should have an assistant with tool_calls and then a tool response
        assistant_tc = [m for m in msgs if m.get("role") == "assistant" and "tool_calls" in m]
        assert len(assistant_tc) >= 1
        assert assistant_tc[0]["tool_calls"][0]["id"] == "call_new"
        assert assistant_tc[0]["tool_calls"][0]["function"]["name"] == "new_tool"

        tool_msgs = [m for m in msgs if m.get("role") == "tool"]
        assert len(tool_msgs) >= 1
        assert tool_msgs[0]["tool_call_id"] == "call_new"

    def test_function_call_output_without_tool_name_skipped(self):
        """function_call_output without a tool name is skipped (continues)."""
        req = {
            "model": "m",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_noname",
                    "output": "result",
                    # no "name" field
                }
            ],
        }
        result = convert_responses_to_chat_completions(req)
        # Should not create an assistant tool_call message
        assistant_tc = [m for m in result["messages"] if m.get("role") == "assistant" and "tool_calls" in m]
        assert len(assistant_tc) == 0

    def test_assistant_message_with_output_text_content(self):
        """Assistant messages from previous conversations with output_text content."""
        req = {
            "model": "m",
            "input": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Previous assistant reply"}],
                }
            ],
        }
        result = convert_responses_to_chat_completions(req)
        assistant_msgs = [m for m in result["messages"] if m.get("role") == "assistant"]
        assert len(assistant_msgs) >= 1
        assert assistant_msgs[0]["content"] == "Previous assistant reply"

    def test_simple_string_input_items(self):
        """Simple string items in input become user messages."""
        req = {"model": "m", "input": ["Hello from string"]}
        result = convert_responses_to_chat_completions(req)
        user_msgs = [m for m in result["messages"] if m["role"] == "user"]
        assert any(m["content"] == "Hello from string" for m in user_msgs)

    def test_empty_input_adds_empty_user_message(self):
        """No input at all adds an empty user message."""
        req = {"model": "m"}
        result = convert_responses_to_chat_completions(req)
        user_msgs = [m for m in result["messages"] if m["role"] == "user"]
        assert len(user_msgs) >= 1
        assert user_msgs[0]["content"] == ""

    def test_only_system_message_adds_empty_user(self):
        """If only a system message exists, an empty user message is added."""
        req = {"model": "m", "instructions": "sys", "input": []}
        result = convert_responses_to_chat_completions(req)
        user_msgs = [m for m in result["messages"] if m["role"] == "user"]
        assert len(user_msgs) >= 1

    def test_previous_response_id_loads_history(self):
        """previous_response_id loads conversation history."""
        conversation_history["resp_hist"] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
        ]
        req = {
            "model": "m",
            "previous_response_id": "resp_hist",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "text", "text": "new question"}]}
            ],
        }
        result = convert_responses_to_chat_completions(req)
        msgs = result["messages"]
        # Should have old messages + new question
        assert len(msgs) >= 4
        assert msgs[0]["role"] == "system"
        assert msgs[1]["content"] == "old question"

    def test_tool_conversion_function_type(self):
        """Function type tools with name/description/parameters are converted."""
        req = {
            "model": "m",
            "tools": [
                {
                    "type": "function",
                    "name": "my_func",
                    "description": "Does things",
                    "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}},
                }
            ],
        }
        result = convert_responses_to_chat_completions(req)
        assert len(result["tools"]) == 1
        tool = result["tools"][0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "my_func"
        assert tool["function"]["description"] == "Does things"
        assert "parameters" in tool["function"]

    def test_tool_conversion_skips_non_function(self):
        """Non-function type tools are skipped."""
        req = {
            "model": "m",
            "tools": [
                {"type": "code_interpreter", "name": "ci"},
                {"type": "function", "name": "valid_func"},
            ],
        }
        result = convert_responses_to_chat_completions(req)
        assert len(result["tools"]) == 1
        assert result["tools"][0]["function"]["name"] == "valid_func"

    def test_tool_choice_passthrough(self):
        """tool_choice is passed through."""
        req = {"model": "m", "tool_choice": "required"}
        result = convert_responses_to_chat_completions(req)
        assert result["tool_choice"] == "required"

    def test_reasoning_included_when_non_null(self):
        """reasoning parameter is included when it has non-null values."""
        req = {"model": "m", "reasoning": {"effort": "high", "summary": None}}
        result = convert_responses_to_chat_completions(req)
        assert "reasoning" in result
        assert result["reasoning"]["effort"] == "high"

    def test_reasoning_skipped_when_all_null(self):
        """reasoning parameter is skipped when all values are null."""
        req = {"model": "m", "reasoning": {"effort": None, "summary": None}}
        result = convert_responses_to_chat_completions(req)
        assert "reasoning" not in result

    def test_user_and_metadata_passthrough(self):
        """user and metadata are passed through when present."""
        req = {"model": "m", "user": "user-123", "metadata": {"key": "value"}}
        result = convert_responses_to_chat_completions(req)
        assert result["user"] == "user-123"
        assert result["metadata"] == {"key": "value"}

    def test_user_and_metadata_not_added_when_absent(self):
        """user and metadata are not in result when not in request."""
        req = {"model": "m"}
        result = convert_responses_to_chat_completions(req)
        assert "user" not in result
        assert "metadata" not in result

    def test_final_cleanup_removes_all_null_reasoning(self):
        """Final cleanup removes reasoning if all values became null."""
        # This tests the second cleanup pass at the end of the function
        req = {"model": "m", "reasoning": {"effort": None, "summary": None}}
        result = convert_responses_to_chat_completions(req)
        assert "reasoning" not in result

    def test_default_temperature_and_top_p(self):
        """Default temperature is 1.0 and top_p is 1.0 when not specified."""
        req = {"model": "m"}
        result = convert_responses_to_chat_completions(req)
        assert result["temperature"] == 1.0
        assert result["top_p"] == 1.0

    def test_default_stream_false(self):
        """Default stream is False."""
        req = {"model": "m"}
        result = convert_responses_to_chat_completions(req)
        assert result["stream"] is False

    def test_messages_are_validated(self):
        """Messages go through validate_message_sequence before being returned."""
        # Create a scenario with an orphaned tool message
        conversation_history["prev_orphan"] = [
            {"role": "user", "content": "hello"},
            {"role": "tool", "tool_call_id": "call_orphan", "content": "orphan"},
        ]
        req = {"model": "m", "previous_response_id": "prev_orphan"}
        result = convert_responses_to_chat_completions(req)
        # The orphaned tool message should be removed by validation
        tool_msgs = [m for m in result["messages"] if m.get("role") == "tool"]
        assert len(tool_msgs) == 0

    def test_tool_without_name_skipped(self):
        """Tool dict without 'name' key is skipped."""
        req = {
            "model": "m",
            "tools": [
                {"type": "function"},  # missing name
            ],
        }
        result = convert_responses_to_chat_completions(req)
        assert len(result.get("tools", [])) == 0

    def test_no_tools_key_in_result_when_absent(self):
        """When no tools in request, tools key is absent from result."""
        req = {"model": "m"}
        result = convert_responses_to_chat_completions(req)
        assert "tools" not in result


# ===================================================================
# 3. process_chat_completions_stream (async generator)
# ===================================================================

@pytest.mark.asyncio
class TestProcessChatCompletionsStream:
    """Tests for the process_chat_completions_stream async generator."""

    async def test_text_streaming_basic(self, mock_stream_response):
        """Text content streaming yields created, in_progress, delta, completed events."""
        lines = [
            'data: {"choices":[{"delta":{"content":"Hello"},"index":0}],"model":"test-model"}',
            'data: {"choices":[{"delta":{"content":" world"},"index":0}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}',
            'data: [DONE]',
        ]
        mock_resp = mock_stream_response(lines)
        chat_req = {"messages": [{"role": "user", "content": "hi"}]}

        events = [parse_sse(e) async for e in process_chat_completions_stream(mock_resp, chat_req)]

        # First two events: response.created and response.in_progress
        assert events[0]["type"] == "response.created"
        assert events[1]["type"] == "response.in_progress"

        # Should have text deltas
        deltas = [e for e in events if e["type"] == "response.output_text.delta"]
        assert len(deltas) >= 2
        assert deltas[0]["delta"] == "Hello"
        assert deltas[0]["content_index"] == 0
        assert deltas[1]["delta"] == " world"
        assert deltas[1]["content_index"] == 0

        # Should end with completed
        completed = [e for e in events if e["type"] == "response.completed"]
        assert len(completed) >= 1
        assert completed[0]["response"]["status"] == "completed"

    async def test_model_name_extracted_from_first_chunk(self, mock_stream_response):
        """Model name is extracted from the first chunk."""
        lines = [
            'data: {"choices":[{"delta":{"content":"x"},"index":0}],"model":"my-model-v1"}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}',
            'data: [DONE]',
        ]
        mock_resp = mock_stream_response(lines)
        events = [parse_sse(e) async for e in process_chat_completions_stream(mock_resp)]

        # The completed event should carry the model name
        completed = [e for e in events if e["type"] == "response.completed"]
        assert completed[0]["response"]["model"] == "my-model-v1"

    async def test_done_message_yields_completed(self, mock_stream_response):
        """[DONE] message triggers a completed event if not already completed."""
        lines = [
            'data: {"choices":[{"delta":{"content":"Hi"},"index":0}],"model":"m"}',
            'data: [DONE]',
        ]
        mock_resp = mock_stream_response(lines)
        events = [parse_sse(e) async for e in process_chat_completions_stream(mock_resp)]

        completed = [e for e in events if e["type"] == "response.completed"]
        assert len(completed) == 1

    async def test_done_without_prefix(self, mock_stream_response):
        """[DONE] without 'data: ' prefix is also handled."""
        lines = [
            'data: {"choices":[{"delta":{"content":"Hi"},"index":0}],"model":"m"}',
            '[DONE]',
        ]
        mock_resp = mock_stream_response(lines)
        events = [parse_sse(e) async for e in process_chat_completions_stream(mock_resp)]

        completed = [e for e in events if e["type"] == "response.completed"]
        assert len(completed) == 1

    async def test_empty_chunks_skipped(self, mock_stream_response):
        """Empty chunks are silently skipped."""
        lines = [
            '',
            '   ',
            'data: {"choices":[{"delta":{"content":"A"},"index":0}],"model":"m"}',
            '',
            'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}',
            'data: [DONE]',
        ]
        mock_resp = mock_stream_response(lines)
        events = [parse_sse(e) async for e in process_chat_completions_stream(mock_resp)]

        deltas = [e for e in events if e["type"] == "response.output_text.delta"]
        assert len(deltas) == 1
        assert deltas[0]["delta"] == "A"
        assert deltas[0]["content_index"] == 0

    async def test_json_parse_error_continues(self, mock_stream_response):
        """JSON parse errors in chunks are logged and processing continues."""
        lines = [
            'data: not-json-at-all',
            'data: {"choices":[{"delta":{"content":"OK"},"index":0}],"model":"m"}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}',
            'data: [DONE]',
        ]
        mock_resp = mock_stream_response(lines)
        events = [parse_sse(e) async for e in process_chat_completions_stream(mock_resp)]

        # Should still get the valid text delta
        deltas = [e for e in events if e["type"] == "response.output_text.delta"]
        assert len(deltas) == 1
        assert deltas[0]["delta"] == "OK"
        assert deltas[0]["content_index"] == 0

    async def test_tool_calls_finish_with_mcp_tool(
        self, mock_stream_response, mock_mcp_manager_fixture
    ):
        """finish_reason 'tool_calls' with an MCP tool executes the tool and yields result events."""
        mock_mcp = mock_mcp_manager_fixture
        mock_mcp.is_mcp_tool.return_value = True

        # Create a mock result with .content attribute
        mock_result = MagicMock()
        mock_result.content = [MagicMock(text="tool output text")]
        mock_mcp.execute_mcp_tool.return_value = mock_result

        lines = [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"mcp_tool","arguments":""}}]},"index":0}],"model":"m"}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"q\\":\\"test\\"}"}}]},"index":0}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls","index":0}]}',
            'data: [DONE]',
        ]
        mock_resp = mock_stream_response(lines)
        chat_req = {"messages": [{"role": "user", "content": "hi"}]}

        events = [parse_sse(e) async for e in process_chat_completions_stream(mock_resp, chat_req)]

        # Should have arguments.done event
        done_evts = [e for e in events if e["type"] == "response.function_call_arguments.done"]
        assert len(done_evts) >= 1

        # Should have executed the MCP tool
        mock_mcp.execute_mcp_tool.assert_awaited_once()

        # Should have a completed event
        completed = [e for e in events if e["type"] == "response.completed"]
        assert len(completed) == 1
        assert completed[0]["response"]["status"] == "completed"

        # Response output should include function_call_output
        output_types = [o["type"] for o in completed[0]["response"]["output"]]
        assert "function_call_output" in output_types

    async def test_tool_calls_finish_with_non_mcp_tool(
        self, mock_stream_response, mock_mcp_manager_fixture
    ):
        """finish_reason 'tool_calls' with non-MCP tool yields arguments.done and keeps 'ready' status."""
        mock_mcp = mock_mcp_manager_fixture
        mock_mcp.is_mcp_tool.return_value = False

        lines = [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_xyz","type":"function","function":{"name":"user_tool","arguments":""}}]},"index":0}],"model":"m"}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"a\\":1}"}}]},"index":0}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls","index":0}]}',
            'data: [DONE]',
        ]
        mock_resp = mock_stream_response(lines)
        chat_req = {"messages": [{"role": "user", "content": "hi"}]}

        events = [parse_sse(e) async for e in process_chat_completions_stream(mock_resp, chat_req)]

        # Should NOT have called execute_mcp_tool
        mock_mcp.execute_mcp_tool.assert_not_awaited()

        # Should have arguments.done event
        done_evts = [e for e in events if e["type"] == "response.function_call_arguments.done"]
        assert len(done_evts) >= 1

        # The tool call in the completed response should have status "ready"
        completed = [e for e in events if e["type"] == "response.completed"]
        assert len(completed) == 1
        fc_items = [o for o in completed[0]["response"]["output"] if o.get("type") == "function_call"]
        assert len(fc_items) >= 1
        assert fc_items[0]["status"] == "ready"

    async def test_function_call_finish_with_mcp_tool(
        self, mock_stream_response, mock_mcp_manager_fixture
    ):
        """finish_reason 'function_call' (legacy) with MCP tool executes tool and yields result."""
        mock_mcp = mock_mcp_manager_fixture
        mock_mcp.is_mcp_tool.return_value = True

        mock_result = MagicMock()
        mock_result.content = [MagicMock(text="mcp result")]
        mock_mcp.execute_mcp_tool.return_value = mock_result

        lines = [
            'data: {"choices":[{"delta":{"function_call":{"name":"legacy_mcp","arguments":""}},"index":0}],"model":"m"}',
            'data: {"choices":[{"delta":{"function_call":{"arguments":"{\\"k\\":\\"v\\"}"}},"index":0}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"function_call","index":0}]}',
        ]
        mock_resp = mock_stream_response(lines)
        chat_req = {"messages": [{"role": "user", "content": "hi"}]}

        events = [parse_sse(e) async for e in process_chat_completions_stream(mock_resp, chat_req)]

        # MCP tool should be executed
        mock_mcp.execute_mcp_tool.assert_awaited_once()

        # Should have a completed event with function_call_output
        completed = [e for e in events if e["type"] == "response.completed"]
        assert len(completed) == 1
        output_types = [o["type"] for o in completed[0]["response"]["output"]]
        assert "function_call_output" in output_types

    async def test_function_call_finish_with_non_mcp_tool(
        self, mock_stream_response, mock_mcp_manager_fixture
    ):
        """finish_reason 'function_call' with non-MCP tool forwards function call to client."""
        mock_mcp = mock_mcp_manager_fixture
        mock_mcp.is_mcp_tool.return_value = False

        lines = [
            'data: {"choices":[{"delta":{"function_call":{"name":"client_tool","arguments":""}},"index":0}],"model":"m"}',
            'data: {"choices":[{"delta":{"function_call":{"arguments":"{\\"x\\":1}"}},"index":0}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"function_call","index":0}]}',
        ]
        mock_resp = mock_stream_response(lines)
        chat_req = {"messages": [{"role": "user", "content": "hi"}]}

        events = [parse_sse(e) async for e in process_chat_completions_stream(mock_resp, chat_req)]

        # Should NOT call execute_mcp_tool
        mock_mcp.execute_mcp_tool.assert_not_awaited()

        # Completed response should have a function_call item
        completed = [e for e in events if e["type"] == "response.completed"]
        assert len(completed) == 1
        fc_items = [o for o in completed[0]["response"]["output"] if o.get("type") == "function_call"]
        assert len(fc_items) >= 1
        assert fc_items[0]["name"] == "client_tool"
        assert fc_items[0]["status"] == "ready"

    async def test_conversation_history_saved_on_stop(self, mock_stream_response):
        """Conversation history is saved when finish_reason is 'stop'."""
        lines = [
            'data: {"choices":[{"delta":{"content":"Reply text"},"index":0}],"model":"m"}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}',
            'data: [DONE]',
        ]
        mock_resp = mock_stream_response(lines)
        chat_req = {"messages": [{"role": "user", "content": "hi"}]}

        events = [parse_sse(e) async for e in process_chat_completions_stream(mock_resp, chat_req)]

        # Conversation history should have been saved
        assert len(conversation_history) == 1
        saved_key = list(conversation_history.keys())[0]
        saved_msgs = conversation_history[saved_key]
        # Should have original user message + assistant response
        assert any(m["role"] == "assistant" for m in saved_msgs)

    async def test_conversation_history_saved_on_tool_calls(
        self, mock_stream_response, mock_mcp_manager_fixture
    ):
        """Conversation history is saved when finish_reason is 'tool_calls'."""
        mock_mcp = mock_mcp_manager_fixture
        mock_mcp.is_mcp_tool.return_value = False

        lines = [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_hist","type":"function","function":{"name":"t","arguments":""}}]},"index":0}],"model":"m"}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]},"index":0}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls","index":0}]}',
            'data: [DONE]',
        ]
        mock_resp = mock_stream_response(lines)
        chat_req = {"messages": [{"role": "user", "content": "hi"}]}

        events = [parse_sse(e) async for e in process_chat_completions_stream(mock_resp, chat_req)]

        # Conversation history should be saved
        assert len(conversation_history) == 1
        saved_msgs = list(conversation_history.values())[0]
        # Should have assistant message with tool_calls
        assistant_tc = [m for m in saved_msgs if m.get("role") == "assistant" and "tool_calls" in m]
        assert len(assistant_tc) >= 1

    async def test_conversation_history_trimming(self, mock_stream_response):
        """Conversation history is trimmed when exceeding MAX_CONVERSATION_HISTORY."""
        # Pre-fill conversation_history to be at the limit
        with patch("open_responses_server.responses_service.MAX_CONVERSATION_HISTORY", 3):
            # Add 3 existing entries
            conversation_history["aaa_old1"] = [{"role": "user", "content": "old1"}]
            conversation_history["aaa_old2"] = [{"role": "user", "content": "old2"}]
            conversation_history["aaa_old3"] = [{"role": "user", "content": "old3"}]

            lines = [
                'data: {"choices":[{"delta":{"content":"New"},"index":0}],"model":"m"}',
                'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}',
                'data: [DONE]',
            ]
            mock_resp = mock_stream_response(lines)
            chat_req = {"messages": [{"role": "user", "content": "new"}]}

            events = [parse_sse(e) async for e in process_chat_completions_stream(mock_resp, chat_req)]

            # History should have been trimmed to MAX_CONVERSATION_HISTORY
            assert len(conversation_history) <= 3

    async def test_exception_during_streaming_yields_error(self, mock_stream_response):
        """Exception during streaming yields a completed event with error."""

        class FailingResponse:
            async def aiter_lines(self):
                yield 'data: {"choices":[{"delta":{"content":"OK"},"index":0}],"model":"m"}'
                raise RuntimeError("connection lost")

        mock_resp = FailingResponse()
        events = [parse_sse(e) async for e in process_chat_completions_stream(mock_resp)]

        # Should still get initial events and then a completed event with error
        completed = [e for e in events if e["type"] == "response.completed"]
        assert len(completed) == 1
        assert completed[0]["response"]["status"] == "completed"
        assert completed[0]["response"]["error"] is not None
        assert "connection lost" in completed[0]["response"]["error"]["message"]

    async def test_data_prefix_handling(self, mock_stream_response):
        """Chunks with and without 'data: ' prefix are both handled."""
        lines = [
            'data: {"choices":[{"delta":{"content":"A"},"index":0}],"model":"m"}',
            '{"choices":[{"delta":{"content":"B"},"index":0}]}',  # no prefix
            'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}',
            'data: [DONE]',
        ]
        mock_resp = mock_stream_response(lines)
        events = [parse_sse(e) async for e in process_chat_completions_stream(mock_resp)]

        deltas = [e for e in events if e["type"] == "response.output_text.delta"]
        assert len(deltas) == 2
        assert deltas[0]["delta"] == "A"
        assert deltas[0]["content_index"] == 0
        assert deltas[1]["delta"] == "B"
        assert deltas[1]["content_index"] == 0

    async def test_no_chat_request_no_history_saved(self, mock_stream_response):
        """When chat_request is None, no conversation history is saved."""
        lines = [
            'data: {"choices":[{"delta":{"content":"No save"},"index":0}],"model":"m"}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}',
            'data: [DONE]',
        ]
        mock_resp = mock_stream_response(lines)

        events = [parse_sse(e) async for e in process_chat_completions_stream(mock_resp)]

        # No conversation history should be saved
        assert len(conversation_history) == 0

    async def test_tool_call_arguments_delta_events(
        self, mock_stream_response, mock_mcp_manager_fixture
    ):
        """Tool call argument deltas emit response.function_call_arguments.delta events."""
        mock_mcp = mock_mcp_manager_fixture
        mock_mcp.is_mcp_tool.return_value = False

        lines = [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_d","type":"function","function":{"name":"tfn","arguments":""}}]},"index":0}],"model":"m"}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"a\\""}}]},"index":0}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":1}"}}]},"index":0}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls","index":0}]}',
            'data: [DONE]',
        ]
        mock_resp = mock_stream_response(lines)
        chat_req = {"messages": [{"role": "user", "content": "hi"}]}

        events = [parse_sse(e) async for e in process_chat_completions_stream(mock_resp, chat_req)]

        arg_deltas = [e for e in events if e["type"] == "response.function_call_arguments.delta"]
        # Should have argument delta events for the fragments
        assert len(arg_deltas) >= 2

    async def test_tool_calls_created_event_emitted(
        self, mock_stream_response, mock_mcp_manager_fixture
    ):
        """When a tool call is first seen, an in_progress event is emitted."""
        mock_mcp = mock_mcp_manager_fixture
        mock_mcp.is_mcp_tool.return_value = False

        lines = [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_cr","type":"function","function":{"name":"created_fn","arguments":"{}"}}]},"index":0}],"model":"m"}',
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls","index":0}]}',
            'data: [DONE]',
        ]
        mock_resp = mock_stream_response(lines)
        chat_req = {"messages": [{"role": "user", "content": "hi"}]}

        events = [parse_sse(e) async for e in process_chat_completions_stream(mock_resp, chat_req)]

        # Should have in_progress event for the tool call
        in_progress = [e for e in events if e["type"] == "response.in_progress"]
        # At least 2: one initial, one when tool call is created
        assert len(in_progress) >= 2

    async def test_function_call_legacy_created_event(
        self, mock_stream_response, mock_mcp_manager_fixture
    ):
        """Legacy function_call format emits tool_calls.created event."""
        mock_mcp = mock_mcp_manager_fixture
        mock_mcp.is_mcp_tool.return_value = False

        lines = [
            'data: {"choices":[{"delta":{"function_call":{"name":"legacy_fn","arguments":""}},"index":0}],"model":"m"}',
            'data: {"choices":[{"delta":{"function_call":{"arguments":"{\\"x\\":1}"}},"index":0}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"function_call","index":0}]}',
        ]
        mock_resp = mock_stream_response(lines)
        chat_req = {"messages": [{"role": "user", "content": "hi"}]}

        events = [parse_sse(e) async for e in process_chat_completions_stream(mock_resp, chat_req)]

        # Should have a tool_calls.created event
        created = [e for e in events if e["type"] == "response.tool_calls.created"]
        assert len(created) >= 1
        assert created[0]["tool_call"]["name"] == "legacy_fn"

    async def test_mcp_tool_execution_failure_returns_error(
        self, mock_stream_response, mock_mcp_manager_fixture
    ):
        """When MCP tool execution fails, error result is included."""
        mock_mcp = mock_mcp_manager_fixture
        mock_mcp.is_mcp_tool.return_value = True
        mock_mcp.execute_mcp_tool.side_effect = RuntimeError("tool crashed")

        lines = [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_err","type":"function","function":{"name":"bad_tool","arguments":""}}]},"index":0}],"model":"m"}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]},"index":0}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls","index":0}]}',
            'data: [DONE]',
        ]
        mock_resp = mock_stream_response(lines)
        chat_req = {"messages": [{"role": "user", "content": "hi"}]}

        events = [parse_sse(e) async for e in process_chat_completions_stream(mock_resp, chat_req)]

        # Should still complete
        completed = [e for e in events if e["type"] == "response.completed"]
        assert len(completed) == 1

        # The output should include the error result
        outputs = completed[0]["response"]["output"]
        fc_outputs = [o for o in outputs if o.get("type") == "function_call_output"]
        assert len(fc_outputs) >= 1
        # Error should be captured in the output
        assert "tool crashed" in fc_outputs[0]["output"]

    async def test_done_saves_history_with_output_text(self, mock_stream_response):
        """[DONE] event saves conversation history when there is output text and chat_request."""
        lines = [
            'data: {"choices":[{"delta":{"content":"Saved text"},"index":0}],"model":"m"}',
            'data: [DONE]',
        ]
        mock_resp = mock_stream_response(lines)
        chat_req = {"messages": [{"role": "user", "content": "question"}]}

        events = [parse_sse(e) async for e in process_chat_completions_stream(mock_resp, chat_req)]

        # Conversation history should be saved from the [DONE] handler
        assert len(conversation_history) == 1
        saved = list(conversation_history.values())[0]
        assistant_msgs = [m for m in saved if m["role"] == "assistant"]
        assert len(assistant_msgs) >= 1
        assert "Saved text" in assistant_msgs[0]["content"]

    async def test_conversation_history_saved_on_function_call(
        self, mock_stream_response, mock_mcp_manager_fixture
    ):
        """Conversation history is saved when finish_reason is 'function_call'."""
        mock_mcp = mock_mcp_manager_fixture
        mock_mcp.is_mcp_tool.return_value = False

        lines = [
            'data: {"choices":[{"delta":{"function_call":{"name":"save_fn","arguments":""}},"index":0}],"model":"m"}',
            'data: {"choices":[{"delta":{"function_call":{"arguments":"{}"}},"index":0}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"function_call","index":0}]}',
        ]
        mock_resp = mock_stream_response(lines)
        chat_req = {"messages": [{"role": "user", "content": "do it"}]}

        events = [parse_sse(e) async for e in process_chat_completions_stream(mock_resp, chat_req)]

        # Conversation history should be saved with the tool call
        assert len(conversation_history) == 1
        saved = list(conversation_history.values())[0]
        assistant_tc = [m for m in saved if m.get("role") == "assistant" and "tool_calls" in m]
        assert len(assistant_tc) >= 1

    async def test_history_trimming_on_done_event(self, mock_stream_response):
        """Conversation history is trimmed at [DONE] when exceeding max."""
        with patch("open_responses_server.responses_service.MAX_CONVERSATION_HISTORY", 2):
            conversation_history["zzz_old1"] = [{"role": "user", "content": "1"}]
            conversation_history["zzz_old2"] = [{"role": "user", "content": "2"}]

            lines = [
                'data: {"choices":[{"delta":{"content":"New text"},"index":0}],"model":"m"}',
                'data: [DONE]',
            ]
            mock_resp = mock_stream_response(lines)
            chat_req = {"messages": [{"role": "user", "content": "new"}]}

            events = [parse_sse(e) async for e in process_chat_completions_stream(mock_resp, chat_req)]

            # Should be trimmed to max
            assert len(conversation_history) <= 2

    async def test_multiple_tool_calls_in_single_response(
        self, mock_stream_response, mock_mcp_manager_fixture
    ):
        """Multiple tool calls in a single response are all processed."""
        mock_mcp = mock_mcp_manager_fixture
        mock_mcp.is_mcp_tool.return_value = False

        lines = [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"fn1","arguments":""}},{"index":1,"id":"call_2","type":"function","function":{"name":"fn2","arguments":""}}]},"index":0}],"model":"m"}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}},{"index":1,"function":{"arguments":"{}"}}]},"index":0}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls","index":0}]}',
            'data: [DONE]',
        ]
        mock_resp = mock_stream_response(lines)
        chat_req = {"messages": [{"role": "user", "content": "hi"}]}

        events = [parse_sse(e) async for e in process_chat_completions_stream(mock_resp, chat_req)]

        completed = [e for e in events if e["type"] == "response.completed"]
        assert len(completed) == 1
        fc_items = [o for o in completed[0]["response"]["output"] if o.get("type") == "function_call"]
        assert len(fc_items) >= 2

    async def test_response_id_format(self, mock_stream_response):
        """Response ID follows the 'resp_' prefix format."""
        lines = [
            'data: {"choices":[{"delta":{"content":"x"},"index":0}],"model":"m"}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}',
            'data: [DONE]',
        ]
        mock_resp = mock_stream_response(lines)
        events = [parse_sse(e) async for e in process_chat_completions_stream(mock_resp)]

        created = [e for e in events if e["type"] == "response.created"]
        assert created[0]["response"]["id"].startswith("resp_")

    async def test_stop_with_no_output_adds_empty_message(self, mock_stream_response):
        """Stop finish_reason with empty output_text adds message with fallback text."""
        lines = [
            'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}],"model":"m"}',
            'data: [DONE]',
        ]
        mock_resp = mock_stream_response(lines)
        events = [parse_sse(e) async for e in process_chat_completions_stream(mock_resp)]

        completed = [e for e in events if e["type"] == "response.completed"]
        assert len(completed) >= 1
        output = completed[0]["response"]["output"]
        assert len(output) >= 1
        # Should have the fallback "(No update)" text
        assert output[0]["content"][0]["text"] == "(No update)"
