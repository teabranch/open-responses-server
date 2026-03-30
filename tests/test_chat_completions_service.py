"""
Comprehensive tests for the chat_completions_service module.

Tests cover:
- _handle_non_streaming_request: simple responses, tool call loops, MCP/non-MCP tools,
  error handling, max iterations, reasoning parameter cleanup
- _handle_streaming_request: no tool calls (stream proxy), tool call loops,
  error handling, max iterations, reasoning parameter cleanup
- handle_chat_completions: MCP tool injection, deduplication, dispatch routing
"""
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from fastapi.responses import StreamingResponse

from open_responses_server.chat_completions_service import (
    _handle_non_streaming_request,
    _handle_streaming_request,
    handle_chat_completions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_response(content="Hello", finish_reason="stop", tool_calls=None):
    """Build a mock httpx response whose .json() returns a chat completion."""
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
        # content is typically None when tool_calls are present
        if content == "Hello":
            message["content"] = None

    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
    }
    return resp


def _make_tool_call(tool_call_id="call_1", name="my_tool", arguments=None):
    """Build a tool_call dict as returned by the LLM."""
    return {
        "id": tool_call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments or {"arg": "val"}),
        },
    }


def _make_mcp_result(texts):
    """Build a fake MCP result object with .content list of TextContent-like items."""
    content_items = []
    for t in texts:
        item = MagicMock()
        item.text = t
        content_items.append(item)
    result = MagicMock()
    result.content = content_items
    return result


def _base_request_data():
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
    }


async def _collect_streaming_body(response: StreamingResponse) -> bytes:
    """Consume a StreamingResponse's body generator and return raw bytes."""
    chunks = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, str):
            chunks.append(chunk.encode())
        else:
            chunks.append(chunk)
    return b"".join(chunks)


# ============================================================================
# _handle_non_streaming_request
# ============================================================================

@pytest.mark.asyncio
class TestHandleNonStreamingRequest:
    """Tests for _handle_non_streaming_request."""

    async def test_simple_response_no_tool_calls(self):
        """A normal assistant response is returned directly."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            return_value=_make_llm_response(content="Hello!", finish_reason="stop")
        )
        request_data = _base_request_data()

        result = await _handle_non_streaming_request(mock_client, request_data)

        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["finish_reason"] == "stop"
        mock_client.post.assert_awaited_once()

    @patch("open_responses_server.chat_completions_service.mcp_manager")
    async def test_tool_call_with_mcp_tool(self, mock_mcp):
        """MCP tool call is executed, result fed back, and final response returned."""
        mock_mcp.is_mcp_tool.return_value = True
        mcp_result = _make_mcp_result(["tool output text"])
        mock_mcp.execute_mcp_tool = AsyncMock(return_value=mcp_result)

        tool_call = _make_tool_call(name="mcp_tool")
        first_response = _make_llm_response(
            finish_reason="tool_calls", tool_calls=[tool_call]
        )
        final_response = _make_llm_response(content="Done", finish_reason="stop")

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[first_response, final_response])

        request_data = _base_request_data()
        result = await _handle_non_streaming_request(mock_client, request_data)

        assert result["choices"][0]["message"]["content"] == "Done"
        mock_mcp.execute_mcp_tool.assert_awaited_once_with("mcp_tool", {"arg": "val"})
        # Two POST calls: first triggers tool, second gets final answer
        assert mock_client.post.await_count == 2

        # Verify tool result message was appended to messages
        second_call_kwargs = mock_client.post.call_args_list[1]
        sent_messages = second_call_kwargs.kwargs.get("json", second_call_kwargs[1]["json"])["messages"]
        tool_msg = [m for m in sent_messages if m.get("role") == "tool"]
        assert len(tool_msg) == 1
        assert tool_msg[0]["tool_call_id"] == "call_1"
        assert "tool output text" in tool_msg[0]["content"]

    @patch("open_responses_server.chat_completions_service.mcp_manager")
    async def test_tool_call_with_non_mcp_tool(self, mock_mcp):
        """Non-MCP tool results in an error content message but loop continues."""
        mock_mcp.is_mcp_tool.return_value = False

        tool_call = _make_tool_call(name="unknown_tool")
        first_response = _make_llm_response(
            finish_reason="tool_calls", tool_calls=[tool_call]
        )
        final_response = _make_llm_response(content="Acknowledged", finish_reason="stop")

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[first_response, final_response])

        result = await _handle_non_streaming_request(mock_client, _base_request_data())

        assert result["choices"][0]["message"]["content"] == "Acknowledged"
        # Tool result should contain the error about non-MCP tool
        second_call_json = mock_client.post.call_args_list[1].kwargs.get(
            "json", mock_client.post.call_args_list[1][1]["json"]
        )
        tool_msgs = [m for m in second_call_json["messages"] if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        parsed = json.loads(tool_msgs[0]["content"])
        assert "not a registered MCP tool" in parsed["error"]

    @patch("open_responses_server.chat_completions_service.mcp_manager")
    async def test_mcp_tool_execution_failure(self, mock_mcp):
        """When MCP tool execution raises, error JSON is placed in tool content."""
        mock_mcp.is_mcp_tool.return_value = True
        mock_mcp.execute_mcp_tool = AsyncMock(side_effect=RuntimeError("connection refused"))

        tool_call = _make_tool_call(name="failing_tool")
        first_response = _make_llm_response(
            finish_reason="tool_calls", tool_calls=[tool_call]
        )
        final_response = _make_llm_response(content="Sorry", finish_reason="stop")

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[first_response, final_response])

        result = await _handle_non_streaming_request(mock_client, _base_request_data())

        assert result["choices"][0]["message"]["content"] == "Sorry"
        second_call_json = mock_client.post.call_args_list[1].kwargs.get(
            "json", mock_client.post.call_args_list[1][1]["json"]
        )
        tool_msgs = [m for m in second_call_json["messages"] if m.get("role") == "tool"]
        parsed = json.loads(tool_msgs[0]["content"])
        assert "connection refused" in parsed["error"]

    @patch("open_responses_server.chat_completions_service.MAX_TOOL_CALL_ITERATIONS", 2)
    @patch("open_responses_server.chat_completions_service.mcp_manager")
    async def test_max_iterations_reached(self, mock_mcp):
        """When the loop exhausts MAX_TOOL_CALL_ITERATIONS, return error dict."""
        mock_mcp.is_mcp_tool.return_value = True
        mock_mcp.execute_mcp_tool = AsyncMock(return_value=_make_mcp_result(["ok"]))

        tool_call = _make_tool_call(name="loop_tool")
        always_tools_response = _make_llm_response(
            finish_reason="tool_calls", tool_calls=[tool_call]
        )

        mock_client = AsyncMock()
        # Always return tool_calls so the loop never terminates normally
        mock_client.post = AsyncMock(return_value=always_tools_response)

        result = await _handle_non_streaming_request(mock_client, _base_request_data())

        assert result == {"error": "Max tool call iterations reached"}
        assert mock_client.post.await_count == 2

    async def test_llm_api_exception(self):
        """When client.post raises, return error dict."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("timeout"))

        result = await _handle_non_streaming_request(mock_client, _base_request_data())

        assert result == {"error": "timeout"}

    async def test_raise_for_status_error(self):
        """When raise_for_status() throws, the exception is caught."""
        resp = MagicMock()
        resp.status_code = 500
        resp.raise_for_status.side_effect = Exception("500 Internal Server Error")

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=resp)

        result = await _handle_non_streaming_request(mock_client, _base_request_data())

        assert "500 Internal Server Error" in result["error"]

    async def test_reasoning_null_values_removed(self):
        """Reasoning parameter with all-null values is stripped from the request."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            return_value=_make_llm_response(content="ok", finish_reason="stop")
        )
        request_data = _base_request_data()
        request_data["reasoning"] = {"effort": None, "summary": None}

        await _handle_non_streaming_request(mock_client, request_data)

        sent_json = mock_client.post.call_args.kwargs.get(
            "json", mock_client.post.call_args[1]["json"]
        )
        assert "reasoning" not in sent_json

    async def test_reasoning_with_real_values_kept(self):
        """Reasoning parameter with non-null values is preserved."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            return_value=_make_llm_response(content="ok", finish_reason="stop")
        )
        request_data = _base_request_data()
        request_data["reasoning"] = {"effort": "high", "summary": None}

        await _handle_non_streaming_request(mock_client, request_data)

        sent_json = mock_client.post.call_args.kwargs.get(
            "json", mock_client.post.call_args[1]["json"]
        )
        # Not all values are None, so reasoning should remain
        assert "reasoning" in sent_json

    async def test_stream_flag_removed_from_request(self):
        """The 'stream' key is popped from the request data in each iteration."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            return_value=_make_llm_response(content="ok", finish_reason="stop")
        )
        request_data = _base_request_data()
        request_data["stream"] = True

        await _handle_non_streaming_request(mock_client, request_data)

        sent_json = mock_client.post.call_args.kwargs.get(
            "json", mock_client.post.call_args[1]["json"]
        )
        assert "stream" not in sent_json

    @patch("open_responses_server.chat_completions_service.mcp_manager")
    async def test_multiple_tool_calls_in_single_response(self, mock_mcp):
        """Multiple tool calls in one response are all processed."""
        mock_mcp.is_mcp_tool.return_value = True
        mock_mcp.execute_mcp_tool = AsyncMock(
            side_effect=[
                _make_mcp_result(["result_a"]),
                _make_mcp_result(["result_b"]),
            ]
        )

        tc1 = _make_tool_call(tool_call_id="call_a", name="tool_a", arguments={"x": 1})
        tc2 = _make_tool_call(tool_call_id="call_b", name="tool_b", arguments={"y": 2})
        first_response = _make_llm_response(
            finish_reason="tool_calls", tool_calls=[tc1, tc2]
        )
        final_response = _make_llm_response(content="Both done", finish_reason="stop")

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[first_response, final_response])

        result = await _handle_non_streaming_request(mock_client, _base_request_data())

        assert result["choices"][0]["message"]["content"] == "Both done"
        assert mock_mcp.execute_mcp_tool.await_count == 2

        # Verify both tool result messages were appended
        second_call_json = mock_client.post.call_args_list[1].kwargs.get(
            "json", mock_client.post.call_args_list[1][1]["json"]
        )
        tool_msgs = [m for m in second_call_json["messages"] if m.get("role") == "tool"]
        assert len(tool_msgs) == 2
        ids = {m["tool_call_id"] for m in tool_msgs}
        assert ids == {"call_a", "call_b"}


# ============================================================================
# _handle_streaming_request
# ============================================================================

@pytest.mark.asyncio
class TestHandleStreamingRequest:
    """Tests for _handle_streaming_request."""

    async def test_no_tool_calls_returns_streaming_response(self):
        """When no tool calls, a StreamingResponse proxying the final stream is returned."""
        # The function first makes a non-streaming POST to check for tool calls
        non_stream_resp = _make_llm_response(content="Hi", finish_reason="stop")

        # Then it opens a streaming request
        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.aiter_bytes = MagicMock(
            return_value=_async_iter([b'data: {"chunk": 1}\n\n', b"data: [DONE]\n\n"])
        )
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=non_stream_resp)
        mock_client.stream = MagicMock(return_value=mock_stream_ctx)

        request_data = {**_base_request_data(), "stream": True}
        result = await _handle_streaming_request(mock_client, request_data)

        assert isinstance(result, StreamingResponse)
        assert result.media_type == "text/event-stream"

        # Consume the body to verify it streams data
        body = await _collect_streaming_body(result)
        assert b"chunk" in body

    @patch("open_responses_server.chat_completions_service.mcp_manager")
    async def test_tool_call_detected_executes_and_loops(self, mock_mcp):
        """Tool call in streaming mode: execute MCP tool, loop, then stream final."""
        mock_mcp.is_mcp_tool.return_value = True
        mock_mcp.execute_mcp_tool = AsyncMock(
            return_value=_make_mcp_result(["stream result"])
        )

        tool_call = _make_tool_call(name="stream_tool")
        first_resp = _make_llm_response(
            finish_reason="tool_calls", tool_calls=[tool_call]
        )
        second_resp = _make_llm_response(content="Final", finish_reason="stop")

        # Streaming context for the final response
        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.aiter_bytes = MagicMock(
            return_value=_async_iter([b'data: {"final": true}\n\n'])
        )
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        # First call: tool_calls response; second call: final non-stream response
        mock_client.post = AsyncMock(side_effect=[first_resp, second_resp])
        mock_client.stream = MagicMock(return_value=mock_stream_ctx)

        request_data = {**_base_request_data(), "stream": True}
        result = await _handle_streaming_request(mock_client, request_data)

        assert isinstance(result, StreamingResponse)
        mock_mcp.execute_mcp_tool.assert_awaited_once()

    @patch("open_responses_server.chat_completions_service.mcp_manager")
    async def test_non_mcp_tool_in_streaming(self, mock_mcp):
        """Non-MCP tool in streaming produces error content, loop continues."""
        mock_mcp.is_mcp_tool.return_value = False

        tool_call = _make_tool_call(name="external_tool")
        first_resp = _make_llm_response(
            finish_reason="tool_calls", tool_calls=[tool_call]
        )
        second_resp = _make_llm_response(content="After error", finish_reason="stop")

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.aiter_bytes = MagicMock(
            return_value=_async_iter([b'data: {"ok": true}\n\n'])
        )
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[first_resp, second_resp])
        mock_client.stream = MagicMock(return_value=mock_stream_ctx)

        result = await _handle_streaming_request(
            mock_client, {**_base_request_data(), "stream": True}
        )

        assert isinstance(result, StreamingResponse)
        # The second POST should have a tool message with the error
        second_call_json = mock_client.post.call_args_list[1].kwargs.get(
            "json", mock_client.post.call_args_list[1][1]["json"]
        )
        tool_msgs = [m for m in second_call_json["messages"] if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        parsed = json.loads(tool_msgs[0]["content"])
        assert "not a registered MCP tool" in parsed["error"]

    @patch("open_responses_server.chat_completions_service.mcp_manager")
    async def test_mcp_tool_execution_failure_in_streaming(self, mock_mcp):
        """MCP tool execution failure in streaming mode puts error JSON in tool content."""
        mock_mcp.is_mcp_tool.return_value = True
        mock_mcp.execute_mcp_tool = AsyncMock(
            side_effect=RuntimeError("tool crashed")
        )

        tool_call = _make_tool_call(name="crash_tool")
        first_resp = _make_llm_response(
            finish_reason="tool_calls", tool_calls=[tool_call]
        )
        second_resp = _make_llm_response(content="Recovered", finish_reason="stop")

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.aiter_bytes = MagicMock(
            return_value=_async_iter([b'data: {"ok": true}\n\n'])
        )
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[first_resp, second_resp])
        mock_client.stream = MagicMock(return_value=mock_stream_ctx)

        result = await _handle_streaming_request(
            mock_client, {**_base_request_data(), "stream": True}
        )

        assert isinstance(result, StreamingResponse)
        # The second POST should have a tool message with the error
        second_call_json = mock_client.post.call_args_list[1].kwargs.get(
            "json", mock_client.post.call_args_list[1][1]["json"]
        )
        tool_msgs = [m for m in second_call_json["messages"] if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        parsed = json.loads(tool_msgs[0]["content"])
        assert "tool crashed" in parsed["error"]

    async def test_stream_proxy_exception_yields_error(self):
        """When the streaming connection itself fails, the error is yielded as SSE."""
        non_stream_resp = _make_llm_response(content="ok", finish_reason="stop")

        # Make the stream context manager's __aenter__ raise
        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__aenter__ = AsyncMock(
            side_effect=Exception("stream connection failed")
        )
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=non_stream_resp)
        mock_client.stream = MagicMock(return_value=mock_stream_ctx)

        result = await _handle_streaming_request(
            mock_client, {**_base_request_data(), "stream": True}
        )

        assert isinstance(result, StreamingResponse)

        body = await _collect_streaming_body(result)
        parsed = json.loads(body.decode().replace("data: ", "").strip())
        assert "stream connection failed" in parsed["error"]

    @patch("open_responses_server.chat_completions_service.MAX_TOOL_CALL_ITERATIONS", 2)
    @patch("open_responses_server.chat_completions_service.mcp_manager")
    async def test_max_iterations_returns_error_stream(self, mock_mcp):
        """When max iterations reached in streaming mode, return error StreamingResponse."""
        mock_mcp.is_mcp_tool.return_value = True
        mock_mcp.execute_mcp_tool = AsyncMock(
            return_value=_make_mcp_result(["ok"])
        )

        tool_call = _make_tool_call(name="loop_tool")
        always_tools_resp = _make_llm_response(
            finish_reason="tool_calls", tool_calls=[tool_call]
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=always_tools_resp)

        result = await _handle_streaming_request(
            mock_client, {**_base_request_data(), "stream": True}
        )

        assert isinstance(result, StreamingResponse)
        assert result.status_code == 500

        body = await _collect_streaming_body(result)
        parsed = json.loads(body.decode().replace("data: ", "").strip())
        assert parsed["error"] == "Max tool call iterations reached"

    async def test_exception_returns_error_stream(self):
        """When an exception occurs during the POST, return error StreamingResponse."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("network error"))

        result = await _handle_streaming_request(
            mock_client, {**_base_request_data(), "stream": True}
        )

        assert isinstance(result, StreamingResponse)
        assert result.status_code == 500

        body = await _collect_streaming_body(result)
        assert b"network error" in body

    async def test_reasoning_null_values_removed_from_non_stream_copy(self):
        """Reasoning with all nulls is removed from the non-stream pre-check request."""
        non_stream_resp = _make_llm_response(content="ok", finish_reason="stop")

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.aiter_bytes = MagicMock(
            return_value=_async_iter([b"data: [DONE]\n\n"])
        )
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=non_stream_resp)
        mock_client.stream = MagicMock(return_value=mock_stream_ctx)

        request_data = {
            **_base_request_data(),
            "stream": True,
            "reasoning": {"effort": None, "summary": None},
        }
        await _handle_streaming_request(mock_client, request_data)

        # Check the non-stream POST call
        first_call_json = mock_client.post.call_args.kwargs.get(
            "json", mock_client.post.call_args[1]["json"]
        )
        assert "reasoning" not in first_call_json

    async def test_reasoning_null_values_removed_from_final_stream_copy(self):
        """Reasoning with all nulls is removed from the final streaming request too."""
        non_stream_resp = _make_llm_response(content="ok", finish_reason="stop")

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.aiter_bytes = MagicMock(
            return_value=_async_iter([b"data: [DONE]\n\n"])
        )
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=non_stream_resp)
        mock_client.stream = MagicMock(return_value=mock_stream_ctx)

        request_data = {
            **_base_request_data(),
            "stream": True,
            "reasoning": {"effort": None, "summary": None},
        }
        result = await _handle_streaming_request(mock_client, request_data)

        # Must consume the body to trigger the lazy stream_proxy generator
        await _collect_streaming_body(result)

        # Check the stream() call's json payload
        stream_call_kwargs = mock_client.stream.call_args
        sent_json = stream_call_kwargs.kwargs.get(
            "json", stream_call_kwargs[1]["json"]
        )
        assert "reasoning" not in sent_json

    async def test_reasoning_with_real_values_kept_in_stream(self):
        """Reasoning with at least one non-null value is kept in both requests."""
        non_stream_resp = _make_llm_response(content="ok", finish_reason="stop")

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.aiter_bytes = MagicMock(
            return_value=_async_iter([b"data: [DONE]\n\n"])
        )
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=non_stream_resp)
        mock_client.stream = MagicMock(return_value=mock_stream_ctx)

        request_data = {
            **_base_request_data(),
            "stream": True,
            "reasoning": {"effort": "high", "summary": None},
        }
        await _handle_streaming_request(mock_client, request_data)

        # Non-stream POST should keep reasoning
        first_call_json = mock_client.post.call_args.kwargs.get(
            "json", mock_client.post.call_args[1]["json"]
        )
        assert "reasoning" in first_call_json

    async def test_final_stream_request_has_stream_true(self):
        """The final streaming request always has stream=True."""
        non_stream_resp = _make_llm_response(content="ok", finish_reason="stop")

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.aiter_bytes = MagicMock(
            return_value=_async_iter([b"data: [DONE]\n\n"])
        )
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=non_stream_resp)
        mock_client.stream = MagicMock(return_value=mock_stream_ctx)

        request_data = {**_base_request_data(), "stream": True}
        result = await _handle_streaming_request(mock_client, request_data)

        # Must consume the body to trigger the lazy stream_proxy generator
        await _collect_streaming_body(result)

        stream_call_kwargs = mock_client.stream.call_args
        sent_json = stream_call_kwargs.kwargs.get(
            "json", stream_call_kwargs[1]["json"]
        )
        assert sent_json["stream"] is True

    async def test_non_stream_precheck_has_stream_false(self):
        """The non-streaming pre-check always sends stream=False."""
        non_stream_resp = _make_llm_response(content="ok", finish_reason="stop")

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.aiter_bytes = MagicMock(
            return_value=_async_iter([b"data: [DONE]\n\n"])
        )
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=non_stream_resp)
        mock_client.stream = MagicMock(return_value=mock_stream_ctx)

        request_data = {**_base_request_data(), "stream": True}
        await _handle_streaming_request(mock_client, request_data)

        # Check the non-streaming POST
        first_call_json = mock_client.post.call_args.kwargs.get(
            "json", mock_client.post.call_args[1]["json"]
        )
        assert first_call_json.get("stream") is False


# ============================================================================
# handle_chat_completions
# ============================================================================

@pytest.mark.asyncio
class TestHandleChatCompletions:
    """Tests for handle_chat_completions."""

    @patch("open_responses_server.chat_completions_service.mcp_manager")
    @patch("open_responses_server.chat_completions_service.LLMClient")
    async def test_mcp_tools_injected(self, mock_llm_cls, mock_mcp):
        """MCP tools are injected into the request when available."""
        mock_client = AsyncMock()
        mock_llm_cls.get_client = AsyncMock(return_value=mock_client)
        mock_client.post = AsyncMock(
            return_value=_make_llm_response(content="ok", finish_reason="stop")
        )

        mock_mcp.get_mcp_tools.return_value = [
            {"name": "mcp_tool_1", "description": "desc", "parameters": {}},
        ]

        mock_request = AsyncMock()
        mock_request.json = AsyncMock(return_value={
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        })

        await handle_chat_completions(mock_request)

        # Verify the request data sent to the non-streaming handler has tools
        sent_json = mock_client.post.call_args.kwargs.get(
            "json", mock_client.post.call_args[1]["json"]
        )
        tool_names = [
            t.get("function", {}).get("name") for t in sent_json.get("tools", [])
        ]
        assert "mcp_tool_1" in tool_names

    @patch("open_responses_server.chat_completions_service.mcp_manager")
    @patch("open_responses_server.chat_completions_service.LLMClient")
    async def test_mcp_tools_avoid_duplicates(self, mock_llm_cls, mock_mcp):
        """MCP tools that already exist in the request are not duplicated."""
        mock_client = AsyncMock()
        mock_llm_cls.get_client = AsyncMock(return_value=mock_client)
        mock_client.post = AsyncMock(
            return_value=_make_llm_response(content="ok", finish_reason="stop")
        )

        mock_mcp.get_mcp_tools.return_value = [
            {"name": "existing_tool", "description": "desc", "parameters": {}},
            {"name": "new_tool", "description": "desc2", "parameters": {}},
        ]

        mock_request = AsyncMock()
        mock_request.json = AsyncMock(return_value={
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "existing_tool", "description": "already here", "parameters": {}},
                }
            ],
        })

        await handle_chat_completions(mock_request)

        sent_json = mock_client.post.call_args.kwargs.get(
            "json", mock_client.post.call_args[1]["json"]
        )
        tool_names = [
            t.get("function", {}).get("name") for t in sent_json.get("tools", [])
        ]
        # existing_tool should appear exactly once, new_tool added
        assert tool_names.count("existing_tool") == 1
        assert "new_tool" in tool_names
        assert len(sent_json["tools"]) == 2

    @patch("open_responses_server.chat_completions_service.mcp_manager")
    @patch("open_responses_server.chat_completions_service.LLMClient")
    async def test_no_mcp_tools_available(self, mock_llm_cls, mock_mcp):
        """When no MCP tools exist, no injection happens."""
        mock_client = AsyncMock()
        mock_llm_cls.get_client = AsyncMock(return_value=mock_client)
        mock_client.post = AsyncMock(
            return_value=_make_llm_response(content="ok", finish_reason="stop")
        )

        mock_mcp.get_mcp_tools.return_value = []

        mock_request = AsyncMock()
        mock_request.json = AsyncMock(return_value={
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        })

        await handle_chat_completions(mock_request)

        sent_json = mock_client.post.call_args.kwargs.get(
            "json", mock_client.post.call_args[1]["json"]
        )
        assert "tools" not in sent_json

    @patch("open_responses_server.chat_completions_service._handle_streaming_request")
    @patch("open_responses_server.chat_completions_service.mcp_manager")
    @patch("open_responses_server.chat_completions_service.LLMClient")
    async def test_streaming_dispatches_to_streaming_handler(
        self, mock_llm_cls, mock_mcp, mock_stream_handler
    ):
        """stream=True dispatches to _handle_streaming_request."""
        mock_client = AsyncMock()
        mock_llm_cls.get_client = AsyncMock(return_value=mock_client)
        mock_mcp.get_mcp_tools.return_value = []
        mock_stream_handler.return_value = StreamingResponse(
            iter([]), media_type="text/event-stream"
        )

        mock_request = AsyncMock()
        mock_request.json = AsyncMock(return_value={
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })

        await handle_chat_completions(mock_request)

        mock_stream_handler.assert_awaited_once()
        args = mock_stream_handler.call_args
        assert args[0][1]["stream"] is True

    @patch("open_responses_server.chat_completions_service._handle_non_streaming_request")
    @patch("open_responses_server.chat_completions_service.mcp_manager")
    @patch("open_responses_server.chat_completions_service.LLMClient")
    async def test_non_streaming_dispatches_to_non_streaming_handler(
        self, mock_llm_cls, mock_mcp, mock_non_stream_handler
    ):
        """stream=False dispatches to _handle_non_streaming_request."""
        mock_client = AsyncMock()
        mock_llm_cls.get_client = AsyncMock(return_value=mock_client)
        mock_mcp.get_mcp_tools.return_value = []
        mock_non_stream_handler.return_value = {"choices": []}

        mock_request = AsyncMock()
        mock_request.json = AsyncMock(return_value={
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        })

        await handle_chat_completions(mock_request)

        mock_non_stream_handler.assert_awaited_once()
        args = mock_non_stream_handler.call_args
        assert args[0][1].get("stream") is False

    @patch("open_responses_server.chat_completions_service._handle_non_streaming_request")
    @patch("open_responses_server.chat_completions_service.mcp_manager")
    @patch("open_responses_server.chat_completions_service.LLMClient")
    async def test_default_stream_false(
        self, mock_llm_cls, mock_mcp, mock_non_stream_handler
    ):
        """When 'stream' is not in request, defaults to non-streaming."""
        mock_client = AsyncMock()
        mock_llm_cls.get_client = AsyncMock(return_value=mock_client)
        mock_mcp.get_mcp_tools.return_value = []
        mock_non_stream_handler.return_value = {"choices": []}

        mock_request = AsyncMock()
        mock_request.json = AsyncMock(return_value={
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
        })

        await handle_chat_completions(mock_request)

        mock_non_stream_handler.assert_awaited_once()


# ---------------------------------------------------------------------------
# Async iterator helper (defined at module level for pickling compatibility)
# ---------------------------------------------------------------------------

async def _async_iter(items):
    """Turn a list into an async iterator."""
    for item in items:
        yield item
