"""
Tests for MCPServer operations, MCPManager lifecycle, and serialize_tool_result.

Covers list_tools, execute_tool, cleanup, manager startup/shutdown/refresh,
tool lookup/execution routing, and result serialization.
Transport dispatch and helper functions are tested in test_mcp_transport.py.
"""
import json
import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock, mock_open

from open_responses_server.common.mcp_manager import (
    MCPServer,
    MCPManager,
    serialize_tool_result,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_tool(name, description="A tool", input_schema=None):
    """Create a mock tool object matching the MCP SDK tool shape."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.inputSchema = input_schema or {"type": "object"}
    return tool


# ---------------------------------------------------------------------------
# MCPServer operations
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestMCPServerOperations:
    """Tests for MCPServer.list_tools, execute_tool, and cleanup."""

    # -- list_tools ---------------------------------------------------------

    async def test_list_tools_parses_response(self):
        """list_tools returns structured dicts from session response tuples."""
        server = MCPServer("srv", {"command": "test"})
        mock_session = AsyncMock()

        tool_a = _make_mock_tool("tool_a", "Alpha tool", {"type": "object", "properties": {}})
        tool_b = _make_mock_tool("tool_b", "Beta tool")

        # session.list_tools() returns an iterable of tuples
        mock_session.list_tools = AsyncMock(
            return_value=[("tools", [tool_a, tool_b])]
        )
        server.session = mock_session

        tools = await server.list_tools()

        assert len(tools) == 2
        assert tools[0] == {
            "name": "tool_a",
            "description": "Alpha tool",
            "parameters": {"type": "object", "properties": {}},
        }
        assert tools[1]["name"] == "tool_b"
        assert tools[1]["description"] == "Beta tool"

    async def test_list_tools_skips_non_tools_tuples(self):
        """list_tools ignores tuples whose first element is not 'tools'."""
        server = MCPServer("srv", {"command": "test"})
        mock_session = AsyncMock()
        tool = _make_mock_tool("only_tool")
        mock_session.list_tools = AsyncMock(
            return_value=[
                ("cursor", "abc123"),
                ("tools", [tool]),
            ]
        )
        server.session = mock_session

        tools = await server.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "only_tool"

    async def test_list_tools_empty(self):
        """list_tools returns an empty list when server reports no tools."""
        server = MCPServer("srv", {"command": "test"})
        mock_session = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=[("tools", [])])
        server.session = mock_session

        tools = await server.list_tools()
        assert tools == []

    async def test_list_tools_no_session_raises(self):
        """list_tools raises RuntimeError when session is None."""
        server = MCPServer("srv", {"command": "test"})
        server.session = None

        with pytest.raises(RuntimeError, match="not initialized"):
            await server.list_tools()

    # -- execute_tool -------------------------------------------------------

    async def test_execute_tool_delegates_to_session(self):
        """execute_tool calls session.call_tool and returns its result."""
        server = MCPServer("srv", {"command": "test"})
        mock_session = AsyncMock()
        expected = MagicMock()
        mock_session.call_tool = AsyncMock(return_value=expected)
        server.session = mock_session

        result = await server.execute_tool("do_stuff", {"x": 1})

        mock_session.call_tool.assert_awaited_once_with("do_stuff", {"x": 1})
        assert result is expected

    async def test_execute_tool_no_session_raises(self):
        """execute_tool raises RuntimeError when session is None."""
        server = MCPServer("srv", {"command": "test"})
        server.session = None

        with pytest.raises(RuntimeError, match="not initialized"):
            await server.execute_tool("anything", {})

    async def test_execute_tool_propagates_exception(self):
        """execute_tool re-raises exceptions from session.call_tool."""
        server = MCPServer("srv", {"command": "test"})
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(side_effect=ConnectionError("lost"))
        server.session = mock_session

        with pytest.raises(ConnectionError, match="lost"):
            await server.execute_tool("broken", {})

    # -- cleanup ------------------------------------------------------------

    async def test_cleanup_closes_and_clears_session(self):
        """cleanup calls exit_stack.aclose and sets session to None."""
        server = MCPServer("srv", {"command": "test"})
        server.session = MagicMock()
        server.exit_stack = AsyncMock()
        server.exit_stack.aclose = AsyncMock()

        await server.cleanup()

        server.exit_stack.aclose.assert_awaited_once()
        assert server.session is None


# ---------------------------------------------------------------------------
# MCPManager lifecycle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestMCPManagerLifecycle:
    """Tests for MCPManager startup, shutdown, and refresh."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        MCPManager._instance = None
        yield
        MCPManager._instance = None

    # -- get_instance -------------------------------------------------------

    async def test_get_instance_returns_singleton(self):
        """get_instance returns the same object on subsequent calls."""
        a = MCPManager.get_instance()
        b = MCPManager.get_instance()
        assert a is b

    # -- startup_mcp_servers ------------------------------------------------

    async def test_startup_loads_config_and_initializes(self, tmp_path):
        """startup_mcp_servers reads JSON, initializes servers, caches tools."""
        config = {
            "mcpServers": {
                "s1": {"command": "cmd1"},
                "s2": {"command": "cmd2"},
            }
        }
        config_file = tmp_path / "servers.json"
        config_file.write_text(json.dumps(config))

        manager = MCPManager.get_instance()

        tool_s1 = _make_mock_tool("t1", "Tool 1")
        tool_s2 = _make_mock_tool("t2", "Tool 2")

        init_call_count = 0

        async def fake_initialize(self_server):
            nonlocal init_call_count
            init_call_count += 1

        async def fake_list_tools(self_server):
            if self_server.name == "s1":
                return [{"name": "t1", "description": "Tool 1", "parameters": {}}]
            return [{"name": "t2", "description": "Tool 2", "parameters": {}}]

        with patch(
            "open_responses_server.common.mcp_manager.MCP_SERVERS_CONFIG_PATH",
            str(config_file),
        ), patch.object(
            MCPServer, "initialize", fake_initialize
        ), patch.object(
            MCPServer, "list_tools", fake_list_tools
        ):
            await manager.startup_mcp_servers()

        assert len(manager.mcp_servers) == 2
        assert len(manager.mcp_functions_cache) == 2
        tool_names = {t["name"] for t in manager.mcp_functions_cache}
        assert tool_names == {"t1", "t2"}
        assert manager._server_tool_mapping["t1"] == "s1"
        assert manager._server_tool_mapping["t2"] == "s2"

        # Cancel the background refresh task created during startup
        if manager._refresh_task:
            manager._refresh_task.cancel()
            try:
                await manager._refresh_task
            except asyncio.CancelledError:
                pass

    async def test_startup_missing_config_logs_warning(self, tmp_path):
        """startup_mcp_servers with missing config file does not crash."""
        manager = MCPManager.get_instance()
        missing_path = str(tmp_path / "nonexistent.json")

        with patch(
            "open_responses_server.common.mcp_manager.MCP_SERVERS_CONFIG_PATH",
            missing_path,
        ):
            # Should not raise
            await manager.startup_mcp_servers()

        assert manager.mcp_servers == []
        assert manager.mcp_functions_cache == []

        if manager._refresh_task:
            manager._refresh_task.cancel()
            try:
                await manager._refresh_task
            except asyncio.CancelledError:
                pass

    async def test_startup_server_init_failure_continues(self, tmp_path):
        """If one server fails to initialize, others still succeed."""
        config = {
            "mcpServers": {
                "good": {"command": "ok"},
                "bad": {"command": "fail"},
            }
        }
        config_file = tmp_path / "servers.json"
        config_file.write_text(json.dumps(config))

        manager = MCPManager.get_instance()

        async def fake_initialize(self_server):
            if self_server.name == "bad":
                raise RuntimeError("init explosion")

        async def fake_list_tools(self_server):
            return [{"name": "good_tool", "description": "Works", "parameters": {}}]

        with patch(
            "open_responses_server.common.mcp_manager.MCP_SERVERS_CONFIG_PATH",
            str(config_file),
        ), patch.object(
            MCPServer, "initialize", fake_initialize
        ), patch.object(
            MCPServer, "list_tools", fake_list_tools
        ):
            await manager.startup_mcp_servers()

        # Only the "good" server should be present
        assert len(manager.mcp_servers) == 1
        assert manager.mcp_servers[0].name == "good"

        if manager._refresh_task:
            manager._refresh_task.cancel()
            try:
                await manager._refresh_task
            except asyncio.CancelledError:
                pass

    # -- shutdown_mcp_servers -----------------------------------------------

    async def test_shutdown_cancels_task_and_cleans_up(self):
        """shutdown cancels the refresh task and cleans up all servers."""
        manager = MCPManager.get_instance()

        mock_server_a = AsyncMock(spec=MCPServer)
        mock_server_a.name = "a"
        mock_server_b = AsyncMock(spec=MCPServer)
        mock_server_b.name = "b"
        manager.mcp_servers = [mock_server_a, mock_server_b]

        mock_task = MagicMock()
        mock_task.cancel = MagicMock()
        manager._refresh_task = mock_task

        await manager.shutdown_mcp_servers()

        mock_task.cancel.assert_called_once()
        mock_server_a.cleanup.assert_awaited_once()
        mock_server_b.cleanup.assert_awaited_once()

    async def test_shutdown_cleanup_failure_continues(self):
        """If one server's cleanup fails, the rest still get cleaned up."""
        manager = MCPManager.get_instance()

        failing = AsyncMock(spec=MCPServer)
        failing.name = "failing"
        failing.cleanup = AsyncMock(side_effect=OSError("cleanup boom"))

        ok = AsyncMock(spec=MCPServer)
        ok.name = "ok"

        manager.mcp_servers = [failing, ok]
        manager._refresh_task = None

        await manager.shutdown_mcp_servers()

        failing.cleanup.assert_awaited_once()
        ok.cleanup.assert_awaited_once()

    # -- _refresh_mcp_functions ---------------------------------------------

    async def test_refresh_updates_cache_and_mapping(self):
        """_refresh_mcp_functions fetches tools and rebuilds cache."""
        manager = MCPManager.get_instance()

        srv1 = AsyncMock(spec=MCPServer)
        srv1.name = "srv1"
        srv1.list_tools = AsyncMock(return_value=[
            {"name": "alpha", "description": "A", "parameters": {}},
        ])
        srv2 = AsyncMock(spec=MCPServer)
        srv2.name = "srv2"
        srv2.list_tools = AsyncMock(return_value=[
            {"name": "beta", "description": "B", "parameters": {"type": "object"}},
            {"name": "gamma", "description": "C", "parameters": {}},
        ])
        manager.mcp_servers = [srv1, srv2]

        await manager._refresh_mcp_functions()

        assert len(manager.mcp_functions_cache) == 3
        names = {t["name"] for t in manager.mcp_functions_cache}
        assert names == {"alpha", "beta", "gamma"}
        assert manager._server_tool_mapping["alpha"] == "srv1"
        assert manager._server_tool_mapping["beta"] == "srv2"
        assert manager._server_tool_mapping["gamma"] == "srv2"

    async def test_refresh_with_server_error_continues(self):
        """_refresh_mcp_functions skips erroring servers, keeps others."""
        manager = MCPManager.get_instance()

        broken = AsyncMock(spec=MCPServer)
        broken.name = "broken"
        broken.list_tools = AsyncMock(side_effect=ConnectionError("down"))

        working = AsyncMock(spec=MCPServer)
        working.name = "working"
        working.list_tools = AsyncMock(return_value=[
            {"name": "ok_tool", "description": "works", "parameters": {}},
        ])
        manager.mcp_servers = [broken, working]

        await manager._refresh_mcp_functions()

        assert len(manager.mcp_functions_cache) == 1
        assert manager.mcp_functions_cache[0]["name"] == "ok_tool"


# ---------------------------------------------------------------------------
# MCPManager tool operations
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestMCPManagerToolOps:
    """Tests for get_mcp_tools, is_mcp_tool, execute_mcp_tool."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        MCPManager._instance = None
        yield
        MCPManager._instance = None

    # -- get_mcp_tools ------------------------------------------------------

    async def test_get_mcp_tools_returns_cache(self):
        """get_mcp_tools returns the current mcp_functions_cache."""
        manager = MCPManager.get_instance()
        cache = [{"name": "x", "description": "X", "parameters": {}}]
        manager.mcp_functions_cache = cache

        assert manager.get_mcp_tools() is cache

    # -- is_mcp_tool --------------------------------------------------------

    async def test_is_mcp_tool_true(self):
        """is_mcp_tool returns True for a tool in the cache."""
        manager = MCPManager.get_instance()
        manager.mcp_functions_cache = [
            {"name": "known", "description": "", "parameters": {}},
        ]
        manager._server_tool_mapping = {"known": "srv"}

        assert manager.is_mcp_tool("known") is True

    async def test_is_mcp_tool_false(self):
        """is_mcp_tool returns False for unknown tools."""
        manager = MCPManager.get_instance()
        manager.mcp_functions_cache = []
        assert manager.is_mcp_tool("unknown") is False

    # -- execute_mcp_tool ---------------------------------------------------

    async def test_execute_mapped_path(self):
        """execute_mcp_tool uses _server_tool_mapping for direct dispatch."""
        manager = MCPManager.get_instance()

        mock_server = AsyncMock(spec=MCPServer)
        mock_server.name = "fast_server"
        expected = MagicMock()
        mock_server.execute_tool = AsyncMock(return_value=expected)

        manager.mcp_servers = [mock_server]
        manager._server_tool_mapping = {"mapped_tool": "fast_server"}

        result = await manager.execute_mcp_tool("mapped_tool", {"key": "val"})

        mock_server.execute_tool.assert_awaited_once_with("mapped_tool", {"key": "val"})
        assert result is expected

    async def test_execute_fallback_search(self):
        """execute_mcp_tool falls back to searching all servers."""
        manager = MCPManager.get_instance()

        srv = AsyncMock(spec=MCPServer)
        srv.name = "search_srv"
        srv.list_tools = AsyncMock(return_value=[
            {"name": "hidden_tool", "description": "Found by search"},
        ])
        expected = MagicMock()
        srv.execute_tool = AsyncMock(return_value=expected)

        manager.mcp_servers = [srv]
        manager._server_tool_mapping = {}  # not in mapping
        manager.mcp_functions_cache = []

        result = await manager.execute_mcp_tool("hidden_tool", {"a": 1})

        srv.list_tools.assert_awaited_once()
        srv.execute_tool.assert_awaited_once_with("hidden_tool", {"a": 1})
        assert result is expected
        # Mapping should be updated for future lookups
        assert manager._server_tool_mapping["hidden_tool"] == "search_srv"

    async def test_execute_tool_not_found_raises(self):
        """execute_mcp_tool raises RuntimeError when no server has the tool."""
        manager = MCPManager.get_instance()

        srv = AsyncMock(spec=MCPServer)
        srv.name = "empty"
        srv.list_tools = AsyncMock(return_value=[])
        manager.mcp_servers = [srv]
        manager._server_tool_mapping = {}
        manager.mcp_functions_cache = []

        with pytest.raises(RuntimeError, match="not found on any active MCP server"):
            await manager.execute_mcp_tool("ghost_tool", {})

    async def test_execute_mapped_server_fails_falls_through_to_search(self):
        """If mapped server fails, execute_mcp_tool searches other servers."""
        manager = MCPManager.get_instance()

        bad_srv = AsyncMock(spec=MCPServer)
        bad_srv.name = "bad"
        bad_srv.execute_tool = AsyncMock(side_effect=RuntimeError("exploded"))

        good_srv = AsyncMock(spec=MCPServer)
        good_srv.name = "good"
        good_srv.list_tools = AsyncMock(return_value=[
            {"name": "flaky_tool"},
        ])
        expected = MagicMock()
        good_srv.execute_tool = AsyncMock(return_value=expected)

        manager.mcp_servers = [bad_srv, good_srv]
        manager._server_tool_mapping = {"flaky_tool": "bad"}
        manager.mcp_functions_cache = []

        result = await manager.execute_mcp_tool("flaky_tool", {})

        # Bad server was tried first via mapping
        bad_srv.execute_tool.assert_awaited_once()
        # Good server found via fallback search
        good_srv.execute_tool.assert_awaited_once_with("flaky_tool", {})
        assert result is expected

    async def test_execute_fallback_search_skips_erroring_servers(self):
        """Fallback search continues past servers that error during listing."""
        manager = MCPManager.get_instance()

        err_srv = AsyncMock(spec=MCPServer)
        err_srv.name = "err"
        err_srv.list_tools = AsyncMock(side_effect=ConnectionError("offline"))

        ok_srv = AsyncMock(spec=MCPServer)
        ok_srv.name = "ok"
        ok_srv.list_tools = AsyncMock(return_value=[{"name": "target"}])
        expected = MagicMock()
        ok_srv.execute_tool = AsyncMock(return_value=expected)

        manager.mcp_servers = [err_srv, ok_srv]
        manager._server_tool_mapping = {}
        manager.mcp_functions_cache = []

        result = await manager.execute_mcp_tool("target", {})
        assert result is expected


# ---------------------------------------------------------------------------
# serialize_tool_result
# ---------------------------------------------------------------------------

class TestSerializeToolResult:
    """Tests for the serialize_tool_result helper."""

    def test_content_list_with_text(self):
        """Result with .content list of text items returns JSON text list."""
        item1 = MagicMock()
        item1.text = "hello"
        item2 = MagicMock()
        item2.text = "world"
        result = MagicMock()
        result.content = [item1, item2]

        serialized = serialize_tool_result(result)
        assert json.loads(serialized) == ["hello", "world"]

    def test_content_list_filters_non_text(self):
        """Items without .text attribute are excluded from output."""
        text_item = MagicMock()
        text_item.text = "included"
        no_text = MagicMock(spec=[])  # no attributes at all
        result = MagicMock()
        result.content = [text_item, no_text]

        serialized = serialize_tool_result(result)
        assert json.loads(serialized) == ["included"]

    def test_content_list_empty(self):
        """Empty content list results in empty JSON array."""
        result = MagicMock()
        result.content = []

        serialized = serialize_tool_result(result)
        assert json.loads(serialized) == []

    def test_plain_object_json_dumps(self):
        """Non-content objects are serialized via json.dumps."""
        data = {"key": "value", "num": 42}
        serialized = serialize_tool_result(data)
        assert json.loads(serialized) == data

    def test_plain_string(self):
        """A plain string is JSON-serialized."""
        serialized = serialize_tool_result("just a string")
        assert json.loads(serialized) == "just a string"

    def test_plain_list(self):
        """A plain list (no .content) is JSON-serialized."""
        serialized = serialize_tool_result([1, 2, 3])
        assert json.loads(serialized) == [1, 2, 3]
