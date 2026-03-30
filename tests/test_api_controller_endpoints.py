"""
Tests for api_controller.py endpoints.
"""
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from fastapi.responses import StreamingResponse

from open_responses_server.api_controller import app


class TestResponsesEndpoint:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_responses_streaming_no_mcp(self, client, mock_mcp_manager_fixture, mock_llm_client_fixture):
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

    def test_responses_non_streaming(self, client, mock_mcp_manager_fixture):
        """POST /responses non-streaming logs unsupported and returns 200."""
        request_data = {
            "model": "test-model",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]}],
            "stream": False,
        }
        response = client.post("/responses", json=request_data)
        # Non-streaming returns 200 with None body
        assert response.status_code == 200

    def test_responses_exception_returns_500(self, client, mock_mcp_manager_fixture):
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

    def test_responses_input_logging_variants(self, client, mock_mcp_manager_fixture):
        """POST /responses with various input types for logging coverage."""
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
        assert response.status_code == 200

    def test_responses_stream_error_from_llm(self, client, mock_mcp_manager_fixture, mock_llm_client_fixture):
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
        """POST /responses with existing tools and MCP tools - no duplicates."""
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


class TestChatCompletionsEndpoint:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_chat_completions_delegates(self, client, mock_mcp_manager_fixture, mock_llm_client_fixture):
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
    def client(self):
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
