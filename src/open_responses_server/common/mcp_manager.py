import os
import json
import asyncio
import logging
import shutil
import traceback
from pathlib import Path
from contextlib import AsyncExitStack
from typing import Dict, List, Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import MCP_TOOL_REFRESH_INTERVAL, logger

class MCPServer:
    """Wrapper for an MCP server session and tool execution."""
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.exit_stack = AsyncExitStack()
        self.session: ClientSession | None = None
        self._cleanup_lock = asyncio.Lock()

    async def initialize(self) -> None:
        command = shutil.which(self.config.get("command")) if self.config.get("command") != "npx" else shutil.which("npx")
        if not command:
            raise ValueError(f"Invalid command for MCP server {self.name}")
        params = StdioServerParameters(
            command=command,
            args=self.config.get("args", []),
            env={**os.environ, **self.config.get("env", {})} if self.config.get("env") else None,
        )
        transport = await self.exit_stack.enter_async_context(stdio_client(params))
        read, write = transport
        session = await self.exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self.session = session

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools with metadata from the server."""
        if not self.session:
            raise RuntimeError(f"MCP server {self.name} not initialized")
        resp = await self.session.list_tools()
        tools: List[Dict[str, Any]] = []
        for item in resp:
            if isinstance(item, tuple) and item[0] == "tools":
                for tool in item[1]:
                    tools.append({
                        "name": tool.name,
                        "description": getattr(tool, "description", None),
                        "parameters": getattr(tool, "inputSchema", {}),
                    })
        return tools

    async def execute_tool(self, tool_name: str, arguments: dict) -> Any:
        if not self.session:
            raise RuntimeError(f"MCP server {self.name} not initialized")
        return await self.session.call_tool(tool_name, arguments)

    async def cleanup(self) -> None:
        async with self._cleanup_lock:
            await self.exit_stack.aclose()
            self.session = None

class MCPManager:
    """Manages MCP servers, tool caching, and execution."""
    _instance = None

    def __init__(self):
        self.mcp_servers: List[MCPServer] = []
        self.mcp_functions_cache: List[Dict[str, Any]] = []
        self._refresh_task: asyncio.Task | None = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def startup_mcp_servers(self):
        """Initialize all MCP servers defined in servers_config.json on startup."""
        config_path = Path(__file__).parent.parent / "servers_config.json"
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            for name, srv in cfg.get("mcpServers", {}).items():
                server = MCPServer(name, srv)
                try:
                    await server.initialize()
                    self.mcp_servers.append(server)
                    logger.info(f"Initialized MCP server: {name}")
                    try:
                        tools = await server.list_tools()
                        logger.info(f"Initial tools for {name}: {[t['name'] for t in tools]}")
                    except Exception as e:
                        logger.warning(f"Could not list tools for {name} on startup: {e}")
                except Exception as e:
                    logger.error(f"Error initializing MCP server {name}: {e}")
        except FileNotFoundError:
            logger.warning(f"{config_path} not found. No MCP servers will be available.")
        
        await self._refresh_mcp_functions() # Initial refresh
        self._refresh_task = asyncio.create_task(self._mcp_refresh_loop())


    async def shutdown_mcp_servers(self):
        """Clean up all MCP servers on shutdown."""
        if self._refresh_task:
            self._refresh_task.cancel()
        for server in self.mcp_servers:
            try:
                await server.cleanup()
                logger.info(f"Cleaned up MCP server: {server.name}")
            except Exception as e:
                logger.error(f"Error cleaning up MCP server {server.name}: {e}")

    async def _refresh_mcp_functions(self) -> None:
        """Fetch tools from all MCP servers and update the cache."""
        new_cache: List[Dict[str, Any]] = []
        for server in self.mcp_servers:
            try:
                tools = await server.list_tools()
                tool_entries = [
                    {"name": t["name"], "description": t.get("description"), "parameters": t.get("parameters", {})}
                    for t in tools
                ]
                new_cache.extend(tool_entries)
            except Exception as e:
                logger.warning(f"Error refreshing tools from {server.name}: {e}")
        self.mcp_functions_cache = new_cache
        # logger.info(f"MCP tool cache refreshed. Found {len(self.mcp_functions_cache)} tools.")


    async def _mcp_refresh_loop(self) -> None:
        """Background task: periodically refresh MCP tool cache."""
        while True:
            await asyncio.sleep(MCP_TOOL_REFRESH_INTERVAL)
            await self._refresh_mcp_functions()

    def get_mcp_tools(self) -> List[Dict[str, Any]]:
        """Returns the cached list of MCP tools."""
        return self.mcp_functions_cache

    def is_mcp_tool(self, tool_name: str) -> bool:
        """Determines if the given tool name belongs to an MCP tool."""
        return any(func.get("name") == tool_name for func in self.mcp_functions_cache)

    async def execute_mcp_tool(self, tool_name: str, arguments: dict) -> Any:
        """Finds the appropriate MCP server hosting the tool and executes it."""
        for server in self.mcp_servers:
            try:
                # No need to list tools again, just try to execute
                # A better approach would be to cache which server has which tool
                # For now, we'll keep the logic of checking before calling
                tools = await server.list_tools()
                if any(tool.get("name") == tool_name for tool in tools):
                    logger.info(f"Executing tool {tool_name} on server {server.name}")
                    return await server.execute_tool(tool_name, arguments)
            except Exception:
                logger.warning(f"Error checking/executing tool on server {server.name}: {traceback.format_exc()}")
                continue
        raise RuntimeError(f"Tool '{tool_name}' not found on any active MCP server")

# Singleton instance
mcp_manager = MCPManager.get_instance() 