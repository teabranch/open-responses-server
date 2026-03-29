"""
Tests for MCP transport type dispatch in MCPServer.initialize().
"""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from contextlib import asynccontextmanager

from open_responses_server.common.mcp_manager import MCPServer


def _mock_session():
    """Create a mock ClientSession that supports async context manager."""
    session = AsyncMock()
    session.initialize = AsyncMock()
    return session


@asynccontextmanager
async def _fake_stdio_client(params):
    read, write = MagicMock(), MagicMock()
    yield read, write


@asynccontextmanager
async def _fake_sse_client(url, headers=None):
    read, write = MagicMock(), MagicMock()
    yield read, write


@asynccontextmanager
async def _fake_streamablehttp_client(url, headers=None):
    read, write, get_session_id = MagicMock(), MagicMock(), MagicMock()
    yield read, write, get_session_id


@pytest.mark.asyncio
class TestMCPServerTransport:
    """Tests for MCPServer.initialize() transport dispatch."""

    @patch("open_responses_server.common.mcp_manager.ClientSession")
    @patch("open_responses_server.common.mcp_manager.stdio_client", side_effect=_fake_stdio_client)
    @patch("shutil.which", return_value="/usr/bin/uvx")
    async def test_stdio_default_type(self, mock_which, mock_stdio, mock_cs):
        """Config without 'type' defaults to stdio transport."""
        mock_cs.return_value = _mock_session()
        server = MCPServer("test", {"command": "uvx", "args": ["some-server"]})
        await server.initialize()
        mock_stdio.assert_called_once()
        assert server.session is not None

    @patch("open_responses_server.common.mcp_manager.ClientSession")
    @patch("open_responses_server.common.mcp_manager.stdio_client", side_effect=_fake_stdio_client)
    @patch("shutil.which", return_value="/usr/bin/uvx")
    async def test_stdio_explicit_type(self, mock_which, mock_stdio, mock_cs):
        """Config with 'type': 'stdio' uses stdio transport."""
        mock_cs.return_value = _mock_session()
        server = MCPServer("test", {"type": "stdio", "command": "uvx"})
        await server.initialize()
        mock_stdio.assert_called_once()

    @patch("open_responses_server.common.mcp_manager.ClientSession")
    @patch("open_responses_server.common.mcp_manager.streamablehttp_client", side_effect=_fake_streamablehttp_client)
    async def test_streamable_http_transport(self, mock_http, mock_cs):
        """Config with type 'streamable-http' uses streamablehttp_client."""
        mock_cs.return_value = _mock_session()
        server = MCPServer("remote", {
            "type": "streamable-http",
            "url": "https://example.com/mcp",
        })
        await server.initialize()
        mock_http.assert_called_once_with(url="https://example.com/mcp", headers=None)
        assert server.session is not None

    @patch("open_responses_server.common.mcp_manager.ClientSession")
    @patch("open_responses_server.common.mcp_manager.sse_client", side_effect=_fake_sse_client)
    async def test_sse_transport(self, mock_sse, mock_cs):
        """Config with type 'sse' uses sse_client."""
        mock_cs.return_value = _mock_session()
        server = MCPServer("sse-server", {
            "type": "sse",
            "url": "https://example.com/sse",
        })
        await server.initialize()
        mock_sse.assert_called_once_with(url="https://example.com/sse", headers=None)
        assert server.session is not None

    @patch("open_responses_server.common.mcp_manager.ClientSession")
    @patch("open_responses_server.common.mcp_manager.streamablehttp_client", side_effect=_fake_streamablehttp_client)
    async def test_headers_passthrough(self, mock_http, mock_cs):
        """Headers from config are forwarded to HTTP transport."""
        mock_cs.return_value = _mock_session()
        headers = {"Authorization": "Bearer tok123"}
        server = MCPServer("authed", {
            "type": "streamable-http",
            "url": "https://example.com/mcp",
            "headers": headers,
        })
        await server.initialize()
        mock_http.assert_called_once_with(url="https://example.com/mcp", headers=headers)

    async def test_streamable_http_missing_url(self):
        """streamable-http without url raises ValueError."""
        server = MCPServer("bad", {"type": "streamable-http"})
        with pytest.raises(ValueError, match="requires a 'url'"):
            await server.initialize()

    async def test_sse_missing_url(self):
        """sse without url raises ValueError."""
        server = MCPServer("bad", {"type": "sse"})
        with pytest.raises(ValueError, match="requires a 'url'"):
            await server.initialize()

    async def test_unknown_transport_type(self):
        """Unknown type raises ValueError listing supported types."""
        server = MCPServer("bad", {"type": "websocket"})
        with pytest.raises(ValueError, match="Unknown transport type"):
            await server.initialize()

    async def test_stdio_missing_command(self):
        """Stdio config where command resolves to None raises ValueError."""
        with patch("shutil.which", return_value=None):
            server = MCPServer("bad", {"command": "nonexistent"})
            with pytest.raises(ValueError, match="Invalid command"):
                await server.initialize()
