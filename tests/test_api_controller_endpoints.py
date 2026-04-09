"""
Tests for api_controller.py endpoints.
"""
import asyncio
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from fastapi.responses import StreamingResponse

from open_responses_server.api_controller import app, _stream_with_keepalive, _HEARTBEAT_EVENT


class TestResponsesEndpoint:
    @pytest.fixture
    def client(self, mock_mcp_manager_fixture, mock_llm_client_fixture):
        # Ensure mocks are applied before app startup runs via TestClient.
        return TestClient(app)

    def test_responses_streaming_no_mcp(self, client, mock_llm_client_fixture):
        """POST /responses streaming with no MCP tools."""
        mock_client = mock_llm_client_fixture

        # Mock the stream context manager
        mock_stream_resp = MagicMock()
        mock_stream_resp.status_code = 200
        mock_stream_resp.__aenter__ = AsyncMock(return_value=mock_stream_resp)
        mock_stream_resp.__aexit__ = AsyncMock(return_value=False)

        async def fake_aiter_lines():
            yield 'data: {"choices":[{"delta":{"content":"Hi"},"index":0}],"model":"test"}'
            yield 'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}'
            yield 'data: [DONE]'

        mock_stream_resp.aiter_lines = fake_aiter_lines
        mock_stream_resp.aread = AsyncMock(return_value=b'error')

        mock_client.stream = MagicMock(return_value=mock_stream_resp)

        request_data = {
            "model": "test-model",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]}],
            "stream": True,
        }

        response = client.post("/responses", json=request_data)
        assert response.status_code == 200

    def test_responses_streaming_with_mcp_tools(self, client, mock_mcp_manager_fixture, mock_llm_client_fixture):
        """POST /responses streaming with MCP tools injected."""
        mock_mgr = mock_mcp_manager_fixture
        mock_mgr.mcp_functions_cache = [
            {"name": "mcp_tool", "description": "An MCP tool", "parameters": {}}
        ]
        mock_mgr.mcp_servers = []

        mock_client = mock_llm_client_fixture

        mock_stream_resp = MagicMock()
        mock_stream_resp.status_code = 200
        mock_stream_resp.__aenter__ = AsyncMock(return_value=mock_stream_resp)
        mock_stream_resp.__aexit__ = AsyncMock(return_value=False)

        async def fake_aiter_lines():
            yield 'data: {"choices":[{"delta":{"content":"Hi"},"index":0}],"model":"test"}'
            yield 'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}'
            yield 'data: [DONE]'

        mock_stream_resp.aiter_lines = fake_aiter_lines
        mock_stream_resp.aread = AsyncMock(return_value=b'error')
        mock_client.stream = MagicMock(return_value=mock_stream_resp)

        request_data = {
            "model": "test-model",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]}],
            "stream": True,
        }

        response = client.post("/responses", json=request_data)
        assert response.status_code == 200

    def test_responses_non_streaming(self, client):
        """POST /responses non-streaming returns 501."""
        request_data = {
            "model": "test-model",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]}],
            "stream": False,
        }
        response = client.post("/responses", json=request_data)
        assert response.status_code == 501

    def test_responses_exception_returns_500(self, client):
        """POST /responses with exception raises 500."""
        with patch("open_responses_server.api_controller.convert_responses_to_chat_completions",
                   side_effect=Exception("conversion error")):
            request_data = {
                "model": "test-model",
                "input": [{"type": "message", "role": "user", "content": [{"type": "text", "text": "Hello"}]}],
                "stream": True,
            }
            response = client.post("/responses", json=request_data)
            assert response.status_code == 500

    def test_responses_input_logging_variants(self, client):
        """POST /responses with various input types for logging coverage (non-streaming returns 501)."""
        request_data = {
            "model": "test-model",
            "input": [
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "Hello"},
                    {"type": "text", "text": "World"},
                    {"type": "image_url", "url": "http://example.com/img.png"},
                    "plain string content",
                ]},
                {"type": "function_call_output", "call_id": "call_1", "output": "result"},
                "plain string input",
            ],
            "stream": False,
        }
        response = client.post("/responses", json=request_data)
        assert response.status_code == 501

    def test_responses_stream_error_from_llm(self, client, mock_llm_client_fixture):
        """POST /responses streaming when LLM returns error status."""
        mock_client = mock_llm_client_fixture

        mock_stream_resp = MagicMock()
        mock_stream_resp.status_code = 500
        mock_stream_resp.__aenter__ = AsyncMock(return_value=mock_stream_resp)
        mock_stream_resp.__aexit__ = AsyncMock(return_value=False)
        mock_stream_resp.aread = AsyncMock(return_value=b'Internal Server Error')
        mock_client.stream = MagicMock(return_value=mock_stream_resp)

        request_data = {
            "model": "test-model",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]}],
            "stream": True,
        }

        response = client.post("/responses", json=request_data)
        assert response.status_code == 200  # StreamingResponse itself is 200, error is in SSE data

    def test_responses_streaming_with_existing_tools(self, client, mock_mcp_manager_fixture, mock_llm_client_fixture):
        """POST /responses with existing user tools and MCP tools - no duplicates."""
        mock_mgr = mock_mcp_manager_fixture
        mock_mgr.mcp_functions_cache = [
            {"name": "mcp_tool", "description": "MCP", "parameters": {}},
            {"name": "existing_tool", "description": "Dup", "parameters": {}},
        ]

        mock_client = mock_llm_client_fixture
        mock_stream_resp = MagicMock()
        mock_stream_resp.status_code = 200
        mock_stream_resp.__aenter__ = AsyncMock(return_value=mock_stream_resp)
        mock_stream_resp.__aexit__ = AsyncMock(return_value=False)

        async def fake_aiter_lines():
            yield 'data: {"choices":[{"delta":{"content":"Hi"},"index":0}],"model":"test"}'
            yield 'data: [DONE]'

        mock_stream_resp.aiter_lines = fake_aiter_lines
        mock_stream_resp.aread = AsyncMock(return_value=b'error')
        mock_client.stream = MagicMock(return_value=mock_stream_resp)

        request_data = {
            "model": "test-model",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]}],
            "stream": True,
            "tools": [{"type": "function", "name": "existing_tool", "description": "Exists", "parameters": {}}],
        }

        response = client.post("/responses", json=request_data)
        assert response.status_code == 200

    def test_responses_streaming_sends_keepalive_before_backend_yields(self, client, mock_llm_client_fixture, monkeypatch):
        """POST /responses emits SSE heartbeat events while backend setup is still waiting."""
        monkeypatch.setattr("open_responses_server.api_controller.HEARTBEAT_INTERVAL", 0.05)

        mock_client = mock_llm_client_fixture

        mock_stream_resp = MagicMock()
        mock_stream_resp.status_code = 200

        async def delayed_enter(*_args, **_kwargs):
            await asyncio.sleep(0.16)
            return mock_stream_resp

        mock_stream_resp.__aenter__ = delayed_enter
        mock_stream_resp.__aexit__ = AsyncMock(return_value=False)

        async def fake_aiter_lines():
            yield 'data: {"choices":[{"delta":{"content":"Hi"},"index":0}],"model":"test"}'
            yield 'data: [DONE]'

        mock_stream_resp.aiter_lines = fake_aiter_lines
        mock_stream_resp.aread = AsyncMock(return_value=b'error')
        mock_client.stream = MagicMock(return_value=mock_stream_resp)

        request_data = {
            "model": "test-model",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]}],
            "stream": True,
        }

        with client.stream("POST", "/responses", json=request_data) as response:
            assert response.status_code == 200

            lines = []
            for idx, line in enumerate(response.iter_lines(), start=1):
                if line:
                    lines.append(line)
                if 'event: response.heartbeat' in lines and any(
                    item.startswith('data: {"type":"response.heartbeat"}') for item in lines
                ):
                    break
                if any(
                    item.startswith("data: ") and 'response.created' in item for item in lines
                ):
                    break
                if idx >= 20:
                    pytest.fail(f"Timed out waiting for heartbeat/data lines: {lines}")

        assert 'event: response.heartbeat' in lines
        assert any(item.startswith('data: {"type":"response.heartbeat"}') for item in lines)
        assert any(item.startswith("data: ") for item in lines)


class TestChatCompletionsEndpoint:
    @pytest.fixture
    def client(self, mock_mcp_manager_fixture, mock_llm_client_fixture):
        # Ensure mocks are applied before app startup runs via TestClient.
        return TestClient(app)

    def test_chat_completions_delegates(self, client, mock_llm_client_fixture):
        """POST /v1/chat/completions delegates to handle_chat_completions."""
        mock_client = mock_llm_client_fixture
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={
            "choices": [{"message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}]
        })
        mock_client.post = AsyncMock(return_value=mock_response)

        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200


class TestProxyEndpoint:
    @pytest.fixture
    def client(self, mock_mcp_manager_fixture, mock_llm_client_fixture):
        # Ensure mocks are applied before app startup runs via TestClient.
        return TestClient(app)

    def test_proxy_get_request(self, client, mock_llm_client_fixture):
        """GET /v1/models proxies to backend."""
        mock_client = mock_llm_client_fixture
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"data": []}'
        mock_response.headers = {}
        mock_client.request = AsyncMock(return_value=mock_response)

        response = client.get("/v1/models")
        assert response.status_code == 200

    def test_proxy_post_non_streaming(self, client, mock_llm_client_fixture):
        """POST proxy with non-streaming body."""
        mock_client = mock_llm_client_fixture
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"result": "ok"}'
        mock_response.headers = {}
        mock_client.request = AsyncMock(return_value=mock_response)

        response = client.post("/v1/embeddings", json={"input": "hello", "model": "test"})
        assert response.status_code == 200

    def test_proxy_streaming_body(self, client, mock_llm_client_fixture):
        """POST proxy with stream=true in body."""
        mock_client = mock_llm_client_fixture

        mock_stream_resp = MagicMock()
        mock_stream_resp.__aenter__ = AsyncMock(return_value=mock_stream_resp)
        mock_stream_resp.__aexit__ = AsyncMock(return_value=False)

        async def fake_aiter_bytes():
            yield b'data: {"chunk": 1}\n\n'
            yield b'data: [DONE]\n\n'

        mock_stream_resp.aiter_bytes = fake_aiter_bytes
        mock_client.stream = MagicMock(return_value=mock_stream_resp)

        response = client.post(
            "/v1/some/streaming/endpoint",
            content=json.dumps({"stream": True, "model": "test"}),
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 200

    def test_proxy_error(self, client, mock_llm_client_fixture):
        """Proxy endpoint error returns 500."""
        mock_client = mock_llm_client_fixture
        mock_client.request = AsyncMock(side_effect=Exception("backend down"))

        response = client.get("/v1/models")
        assert response.status_code == 500

    def test_proxy_invalid_json_body(self, client, mock_llm_client_fixture):
        """POST proxy with invalid JSON body still works (non-streaming)."""
        mock_client = mock_llm_client_fixture
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'ok'
        mock_response.headers = {}
        mock_client.request = AsyncMock(return_value=mock_response)

        response = client.post(
            "/v1/some/endpoint",
            content=b"not json",
            headers={"content-type": "text/plain"},
        )
        assert response.status_code == 200


@pytest.mark.asyncio
class TestStreamWithKeepalive:
    """Tests for heartbeat events while upstream setup is still waiting."""

    async def test_fast_stream_emits_no_heartbeats(self):
        """Immediate upstream items should pass through without heartbeat events."""
        async def fast_stream():
            yield "data: first\n\n"
            yield "data: second\n\n"

        results = [item async for item in _stream_with_keepalive(fast_stream, interval=1.0)]

        assert results == ["data: first\n\n", "data: second\n\n"]

    async def test_emits_keepalives_before_first_upstream_item(self):
        """Heartbeat events should flow before the upstream stream yields its first item."""
        async def delayed_stream():
            await asyncio.sleep(0.35)
            yield "data: first\n\n"

        results = [item async for item in _stream_with_keepalive(delayed_stream, interval=0.1)]

        heartbeats = [item for item in results if item == _HEARTBEAT_EVENT]
        data = [item for item in results if item != _HEARTBEAT_EVENT]

        assert len(heartbeats) >= 2
        assert data == ["data: first\n\n"]

    async def test_empty_stream_produces_no_output(self):
        """An upstream stream that completes immediately should stay silent."""
        async def empty_stream():
            return
            yield  # noqa: unreachable - keeps this as an async generator

        results = [item async for item in _stream_with_keepalive(empty_stream, interval=0.1)]

        assert results == []

    async def test_error_propagates_after_heartbeats(self):
        """Producer exceptions should surface to the consumer."""
        async def error_stream():
            await asyncio.sleep(0.15)
            raise ValueError("upstream failed")
            yield  # noqa: unreachable - keeps this as an async generator

        results = []
        with pytest.raises(ValueError, match="upstream failed"):
            async for item in _stream_with_keepalive(error_stream, interval=0.05):
                results.append(item)

        assert _HEARTBEAT_EVENT in results
