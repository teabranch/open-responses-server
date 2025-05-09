"""
Model Context Protocol (MCP) Client Implementation

This module provides the client implementation for interacting with MCP servers,
which expose tools that can be executed remotely.
"""

import asyncio
import json
import logging
import os
import shutil
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger("mcp_client")

class MCPTool:
    """Represents a tool available from an MCP server."""

    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: Dict[str, Any] = input_schema

    def to_openai_tool(self) -> Dict[str, Any]:
        """Convert MCP tool to OpenAI tool format.

        Returns:
            Dictionary in OpenAI tool format.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema
            }
        }


class MCPServer:
    """Manages a connection to an MCP server and its tools."""

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        """Initialize an MCP server connection manager.

        Args:
            name: The server name identifier
            config: The server configuration containing command, args, env variables
        """
        self.name: str = name
        self.config: Dict[str, Any] = config
        self.session: Optional[ClientSession] = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self.tools_cache: List[MCPTool] = []

    async def initialize(self) -> None:
        """Initialize the connection to the MCP server.

        Raises:
            ValueError: If the command is invalid
            RuntimeError: If initialization fails
        """
        command = (
            shutil.which("npx")
            if self.config["command"] == "npx"
            else self.config["command"]
        )
        if command is None:
            raise ValueError(f"Invalid command for MCP server {self.name}: {self.config['command']}")

        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env={**os.environ, **self.config.get("env", {})}
        )
        
        try:
            logger.info(f"Initializing MCP server {self.name} with command: {command} {' '.join(self.config['args'])}")
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            logger.info(f"MCP server {self.name} initialized successfully")
            self.session = session
            
            # Cache the tools
            await self.list_tools()
        except Exception as e:
            logger.error(f"Error initializing MCP server {self.name}: {str(e)}")
            await self.cleanup()
            raise RuntimeError(f"Failed to initialize MCP server {self.name}: {str(e)}")

    async def list_tools(self) -> List[MCPTool]:
        """List all available tools from the MCP server.
        
        Returns:
            List of MCPTool objects representing available tools.
            
        Raises:
            RuntimeError: If server is not initialized
        """
        if not self.session:
            raise RuntimeError(f"MCP server {self.name} not initialized")

        # If we already have cached tools, return them
        if self.tools_cache:
            return self.tools_cache
            
        tools = []
        try:
            tools_response = await self.session.list_tools()
            logger.info(f"Got tools response from MCP server {self.name}: {tools_response}")
            
            for item in tools_response:
                if isinstance(item, tuple) and item[0] == "tools":
                    tools.extend(
                        MCPTool(
                            tool.name,
                            tool.description,
                            tool.inputSchema
                        )
                        for tool in item[1]
                    )
            
            # Cache the tools for future use
            self.tools_cache = tools
            logger.info(f"Listed {len(tools)} tools from MCP server {self.name}")
            return tools
        except Exception as e:
            logger.error(f"Error listing tools from MCP server {self.name}: {str(e)}")
            raise

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            retries: Number of retry attempts
            delay: Delay between retries in seconds

        Returns:
            The tool execution result

        Raises:
            RuntimeError: If server is not initialized
            Exception: If tool execution fails after all retries
        """
        if not self.session:
            raise RuntimeError(f"MCP server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                logger.info(f"Executing tool {tool_name} on MCP server {self.name} with arguments: {arguments}")
                result = await self.session.call_tool(tool_name, arguments)
                logger.info(f"Tool {tool_name} execution successful")
                return result
            except Exception as e:
                attempt += 1
                logger.warning(
                    f"Error executing tool {tool_name}: {str(e)}. Attempt {attempt} of {retries}."
                )
                if attempt < retries:
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Max retries reached for tool {tool_name}. Failing.")
                    raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                logger.info(f"Cleaning up MCP server {self.name}")
                await self.exit_stack.aclose()
                self.session = None
                self.tools_cache = []
                logger.info(f"MCP server {self.name} cleanup completed")
            except Exception as e:
                logger.error(f"Error during cleanup of MCP server {self.name}: {str(e)}")


class MCPManager:
    """Manages multiple MCP servers and their tools."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the MCP Manager.
        
        Args:
            config_path: Path to the MCP servers configuration file
        """
        self.servers: Dict[str, MCPServer] = {}
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "servers_config.json"
        )
        self._initialized = False

    def load_config(self) -> Dict[str, Any]:
        """Load the MCP servers configuration from file.
        
        Returns:
            Dictionary containing server configurations
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            json.JSONDecodeError: If configuration file is invalid JSON
        """
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"MCP configuration file not found at {self.config_path}")
            # Return empty default configuration
            return {"mcpServers": {}}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in MCP configuration file: {str(e)}")
            raise

    async def initialize(self) -> None:
        """Initialize all MCP servers defined in the configuration."""
        if self._initialized:
            logger.info("MCP Manager already initialized")
            return
            
        try:
            config = self.load_config()
            if "mcpServers" not in config or not config["mcpServers"]:
                logger.info("No MCP servers defined in configuration")
                self._initialized = True
                return
                
            logger.info(f"Initializing {len(config['mcpServers'])} MCP servers")
            
            initialization_tasks = []
            # Create server instances
            for name, server_config in config["mcpServers"].items():
                self.servers[name] = MCPServer(name, server_config)
                initialization_tasks.append(self.servers[name].initialize())
            
            # Initialize all servers concurrently
            if initialization_tasks:
                await asyncio.gather(*initialization_tasks, return_exceptions=True)
                
            logger.info("All MCP servers initialized")
            self._initialized = True
        except Exception as e:
            logger.error(f"Error initializing MCP servers: {str(e)}")
            await self.cleanup()
            raise

    async def get_all_tools(self) -> List[Dict[str, Any]]:
        """Get all tools from all initialized MCP servers in OpenAI format.
        
        Returns:
            List of tools in OpenAI format
        """
        if not self._initialized:
            await self.initialize()
            
        all_tools = []
        for server in self.servers.values():
            try:
                tools = await server.list_tools()
                all_tools.extend([tool.to_openai_tool() for tool in tools])
            except Exception as e:
                logger.error(f"Error getting tools from server {server.name}: {str(e)}")
        
        logger.info(f"Retrieved {len(all_tools)} tools from all MCP servers")
        return all_tools

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool on the appropriate server.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            
        Returns:
            The tool execution result
            
        Raises:
            ValueError: If no server has the specified tool
        """
        if not self._initialized:
            await self.initialize()
            
        # Find the server that has this tool
        for server in self.servers.values():
            try:
                tools = await server.list_tools()
                if any(tool.name == tool_name for tool in tools):
                    return await server.execute_tool(tool_name, arguments)
            except Exception as e:
                logger.error(f"Error while checking server {server.name} for tool {tool_name}: {str(e)}")
                
        raise ValueError(f"No MCP server found with tool: {tool_name}")

    async def cleanup(self) -> None:
        """Clean up all server resources."""
        cleanup_tasks = []
        for server in self.servers.values():
            cleanup_tasks.append(server.cleanup())
            
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
        self.servers = {}
        self._initialized = False
        logger.info("All MCP servers cleaned up")
