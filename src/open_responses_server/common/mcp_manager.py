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

from .config import MCP_TOOL_REFRESH_INTERVAL, MCP_SERVERS_CONFIG_PATH, logger

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
        logger.info(f"[MCP-INIT] Server '{self.name}': command='{command}', args={self.config.get('args', [])}")
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
        logger.info(f"[MCP-INIT] Server '{self.name}' session initialized successfully")

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools with metadata from the server."""
        if not self.session:
            raise RuntimeError(f"MCP server {self.name} not initialized")
        resp = await self.session.list_tools()
        tools: List[Dict[str, Any]] = []
        for item in resp:
            if isinstance(item, tuple) and item[0] == "tools":
                for tool in item[1]:
                    tool_data = {
                        "name": tool.name,
                        "description": getattr(tool, "description", None),
                        "parameters": getattr(tool, "inputSchema", {}),
                    }
                    tools.append(tool_data)
                    logger.debug(f"[MCP-TOOL-LIST] Server '{self.name}': found tool '{tool.name}' with description: '{tool_data['description']}'")
        logger.info(f"[MCP-TOOL-LIST] Server '{self.name}': listed {len(tools)} tools: {[t['name'] for t in tools]}")
        return tools

    async def execute_tool(self, tool_name: str, arguments: dict) -> Any:
        if not self.session:
            raise RuntimeError(f"MCP server {self.name} not initialized")
        logger.info(f"[MCP-EXEC] Server '{self.name}': executing tool '{tool_name}' with args {arguments}")
        try:
            result = await self.session.call_tool(tool_name, arguments)
            logger.info(f"[MCP-EXEC] Server '{self.name}': tool '{tool_name}' executed successfully")
            logger.debug(f"[MCP-EXEC] Server '{self.name}': tool '{tool_name}' result: {result}")
            return result
        except Exception as e:
            logger.error(f"[MCP-EXEC] Server '{self.name}': tool '{tool_name}' failed: {e}")
            raise

    async def cleanup(self) -> None:
        async with self._cleanup_lock:
            await self.exit_stack.aclose()
            self.session = None
            logger.info(f"[MCP-CLEANUP] Server '{self.name}' cleaned up")

class MCPManager:
    """Manages MCP servers, tool caching, and execution."""
    _instance = None

    def __init__(self):
        self.mcp_servers: List[MCPServer] = []
        self.mcp_functions_cache: List[Dict[str, Any]] = []
        self._refresh_task: asyncio.Task | None = None
        self._server_tool_mapping: Dict[str, str] = {}  # tool_name -> server_name mapping

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def startup_mcp_servers(self):
        """Initialize all MCP servers defined in servers_config.json on startup."""
        config_path = Path(MCP_SERVERS_CONFIG_PATH)
        logger.info(f"[MCP-STARTUP] Loading MCP server configuration from: {config_path}")
        
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            
            server_configs = cfg.get("mcpServers", {})
            logger.info(f"[MCP-STARTUP] Found {len(server_configs)} server configurations: {list(server_configs.keys())}")
            
            for name, srv in server_configs.items():
                logger.info(f"[MCP-STARTUP] Initializing server '{name}' with config: {srv}")
                server = MCPServer(name, srv)
                try:
                    await server.initialize()
                    self.mcp_servers.append(server)
                    logger.info(f"[MCP-STARTUP] ✓ Server '{name}' initialized successfully")
                    
                    # Get and log initial tools
                    try:
                        tools = await server.list_tools()
                        tool_names = [t['name'] for t in tools]
                        logger.info(f"[MCP-STARTUP] Server '{name}' provides {len(tools)} tools: {tool_names}")
                        
                        # Update server-tool mapping
                        for tool_name in tool_names:
                            self._server_tool_mapping[tool_name] = name
                            
                    except Exception as e:
                        logger.warning(f"[MCP-STARTUP] Could not list tools for '{name}' on startup: {e}")
                except Exception as e:
                    logger.error(f"[MCP-STARTUP] ✗ Error initializing server '{name}': {e}")
                    
        except FileNotFoundError:
            logger.warning(f"[MCP-STARTUP] {config_path} not found. No MCP servers will be available.")
        except Exception as e:
            logger.error(f"[MCP-STARTUP] Error reading configuration file: {e}")
        
        logger.info(f"[MCP-STARTUP] Successfully initialized {len(self.mcp_servers)} MCP servers")
        
        # Initial tool cache refresh
        await self._refresh_mcp_functions()
        
        # Start background refresh task
        self._refresh_task = asyncio.create_task(self._mcp_refresh_loop())
        logger.info(f"[MCP-STARTUP] Started background tool refresh task (interval: {MCP_TOOL_REFRESH_INTERVAL}s)")

    async def shutdown_mcp_servers(self):
        """Clean up all MCP servers on shutdown."""
        logger.info(f"[MCP-SHUTDOWN] Shutting down {len(self.mcp_servers)} MCP servers")
        
        if self._refresh_task:
            self._refresh_task.cancel()
            logger.info("[MCP-SHUTDOWN] Cancelled background refresh task")
            
        for server in self.mcp_servers:
            try:
                await server.cleanup()
                logger.info(f"[MCP-SHUTDOWN] ✓ Server '{server.name}' cleaned up")
            except Exception as e:
                logger.error(f"[MCP-SHUTDOWN] ✗ Error cleaning up server '{server.name}': {e}")
        
        logger.info("[MCP-SHUTDOWN] MCP server shutdown complete")

    async def _refresh_mcp_functions(self) -> None:
        """Fetch tools from all MCP servers and update the cache."""
        logger.debug("[MCP-REFRESH] Starting tool cache refresh")
        new_cache: List[Dict[str, Any]] = []
        server_tool_counts = {}
        
        for server in self.mcp_servers:
            try:
                tools = await server.list_tools()
                tool_entries = [
                    {"name": t["name"], "description": t.get("description"), "parameters": t.get("parameters", {})}
                    for t in tools
                ]
                new_cache.extend(tool_entries)
                server_tool_counts[server.name] = len(tool_entries)
                
                # Update server-tool mapping
                for tool in tool_entries:
                    self._server_tool_mapping[tool["name"]] = server.name
                    
                logger.debug(f"[MCP-REFRESH] Server '{server.name}': added {len(tool_entries)} tools to cache")
                
            except Exception as e:
                logger.warning(f"[MCP-REFRESH] Error refreshing tools from server '{server.name}': {e}")
                server_tool_counts[server.name] = 0
        
        # Update cache
        old_count = len(self.mcp_functions_cache)
        self.mcp_functions_cache = new_cache
        new_count = len(self.mcp_functions_cache)
        
        # Log detailed refresh results
        all_tool_names = [tool["name"] for tool in self.mcp_functions_cache]
        logger.info(f"[MCP-REFRESH] Tool cache updated: {old_count} -> {new_count} tools")
        logger.info(f"[MCP-REFRESH] Server tool counts: {server_tool_counts}")
        logger.info(f"[MCP-REFRESH] Available tools: {all_tool_names}")

    async def _mcp_refresh_loop(self) -> None:
        """Background task: periodically refresh MCP tool cache."""
        while True:
            try:
                await asyncio.sleep(MCP_TOOL_REFRESH_INTERVAL)
                await self._refresh_mcp_functions()
            except asyncio.CancelledError:
                logger.info("[MCP-REFRESH] Background refresh task cancelled")
                break
            except Exception as e:
                logger.error(f"[MCP-REFRESH] Error in background refresh: {e}")

    def get_mcp_tools(self) -> List[Dict[str, Any]]:
        """Returns the cached list of MCP tools."""
        tool_count = len(self.mcp_functions_cache)
        tool_names = [tool["name"] for tool in self.mcp_functions_cache]
        logger.debug(f"[MCP-GET-TOOLS] Returning {tool_count} cached MCP tools: {tool_names}")
        return self.mcp_functions_cache

    def is_mcp_tool(self, tool_name: str) -> bool:
        """Determines if the given tool name belongs to an MCP tool."""
        is_mcp = any(func.get("name") == tool_name for func in self.mcp_functions_cache)
        server_name = self._server_tool_mapping.get(tool_name, "unknown")
        logger.debug(f"[MCP-CHECK] Tool '{tool_name}': is_mcp={is_mcp}, server='{server_name}'")
        return is_mcp

    async def execute_mcp_tool(self, tool_name: str, arguments: dict) -> Any:
        """Finds the appropriate MCP server hosting the tool and executes it."""
        logger.info(f"[MCP-EXECUTE] Attempting to execute tool '{tool_name}' with args: {arguments}")
        
        # First check if we know which server has this tool
        if tool_name in self._server_tool_mapping:
            server_name = self._server_tool_mapping[tool_name]
            logger.info(f"[MCP-EXECUTE] Tool '{tool_name}' mapped to server '{server_name}'")
            
            # Find the server by name
            for server in self.mcp_servers:
                if server.name == server_name:
                    try:
                        result = await server.execute_tool(tool_name, arguments)
                        logger.info(f"[MCP-EXECUTE] ✓ Tool '{tool_name}' executed successfully on server '{server_name}'")
                        return result
                    except Exception as e:
                        logger.error(f"[MCP-EXECUTE] ✗ Tool '{tool_name}' failed on server '{server_name}': {e}")
                        break
        
        # Fallback: search all servers
        logger.info(f"[MCP-EXECUTE] Searching all {len(self.mcp_servers)} servers for tool '{tool_name}'")
        for server in self.mcp_servers:
            try:
                tools = await server.list_tools()
                if any(tool.get("name") == tool_name for tool in tools):
                    logger.info(f"[MCP-EXECUTE] Found tool '{tool_name}' on server '{server.name}', executing...")
                    result = await server.execute_tool(tool_name, arguments)
                    logger.info(f"[MCP-EXECUTE] ✓ Tool '{tool_name}' executed successfully on server '{server.name}'")
                    
                    # Update mapping for future use
                    self._server_tool_mapping[tool_name] = server.name
                    return result
            except Exception as e:
                logger.warning(f"[MCP-EXECUTE] Error checking/executing tool '{tool_name}' on server '{server.name}': {e}")
                continue
        
        error_msg = f"Tool '{tool_name}' not found on any active MCP server"
        logger.error(f"[MCP-EXECUTE] ✗ {error_msg}")
        logger.error(f"[MCP-EXECUTE] Available servers: {[s.name for s in self.mcp_servers]}")
        logger.error(f"[MCP-EXECUTE] Available tools: {[t['name'] for t in self.mcp_functions_cache]}")
        raise RuntimeError(error_msg)

        
def serialize_tool_result(result):
    if hasattr(result, 'content') and isinstance(result.content, list):
        content_list = [content.text for content in result.content if hasattr(content, 'text')]
        tool_content = json.dumps(content_list)
    else:
        tool_content = json.dumps(result)
    return tool_content

# Singleton instance
mcp_manager = MCPManager.get_instance() 