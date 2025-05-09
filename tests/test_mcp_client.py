"""
Tests for the MCP client module
"""
import json
import pytest
import asyncio
import os
from unittest.mock import patch, MagicMock, AsyncMock
from openai_responses_server.mcp_client import MCPServer, MCPManager, MCPTool

@pytest.mark.asyncio
class TestMCPClient:
    """Tests for the MCP client module"""
    
    @pytest.fixture
    def test_config_path(self):
        """Test config path fixture"""
        return os.path.join(os.path.dirname(__file__), "test_mcp_config.json")
    
    @pytest.fixture
    def mock_session(self):
        """Mock session fixture"""
        mock = MagicMock()
        mock.initialize = AsyncMock()
        mock.list_tools = AsyncMock(return_value=[
            ("tools", [
                MagicMock(
                    name="test_tool",
                    description="A test tool",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "test": {
                                "type": "string",
                                "description": "Test parameter"
                            }
                        }
                    }
                )
            ])
        ])
        mock.call_tool = AsyncMock(return_value={"result": "success"})
        return mock
    
    async def test_mcp_tool_conversion(self):
        """Test converting MCP tool to OpenAI tool format"""
        tool = MCPTool(
            name="test_tool",
            description="A test tool",
            input_schema={
                "type": "object",
                "properties": {
                    "test": {
                        "type": "string",
                        "description": "Test parameter"
                    }
                }
            }
        )
        
        openai_tool = tool.to_openai_tool()
        
        assert openai_tool["type"] == "function"
        assert openai_tool["function"]["name"] == "test_tool"
        assert openai_tool["function"]["description"] == "A test tool"
        assert "parameters" in openai_tool["function"]
    
    @patch("openai_responses_server.mcp_client.ClientSession")
    @patch("openai_responses_server.mcp_client.stdio_client")
    async def test_mcp_server_initialize(self, mock_stdio_client, mock_client_session, mock_session):
        """Test initializing an MCP server"""
        # Setup mocks
        mock_stdio_client.return_value.__aenter__.return_value = (MagicMock(), MagicMock())
        mock_client_session.return_value.__aenter__.return_value = mock_session
        
        # Create a proper tool mock with actual string values
        tool_mock = MagicMock()
        tool_mock.name = "test_tool"
        tool_mock.description = "A test tool"
        tool_mock.inputSchema = {
            "type": "object",
            "properties": {
                "test": {
                    "type": "string", 
                    "description": "Test parameter"
                }
            }
        }
        mock_session.list_tools.return_value = [("tools", [tool_mock])]
        
        # Create server and initialize
        server = MCPServer("test", {
            "command": "python",
            "args": ["-m", "test_server"],
            "env": {"TEST": "value"}
        })
        
        await server.initialize()
        
        # Check that the right calls were made
        mock_stdio_client.assert_called_once()
        mock_client_session.assert_called_once()
        mock_session.initialize.assert_called_once()
        
        # Test listing tools
        tools = await server.list_tools()
        mock_session.list_tools.assert_called_once()
        assert len(tools) == 1
        assert tools[0].name == "test_tool"
        
        # Test executing a tool
        result = await server.execute_tool("test_tool", {"test": "value"})
        mock_session.call_tool.assert_called_once_with("test_tool", {"test": "value"})
        assert result == {"result": "success"}
        
        # Test cleanup
        await server.cleanup()
    
    @patch("openai_responses_server.mcp_client.MCPServer")
    async def test_mcp_manager(self, mock_server_class, test_config_path):
        """Test the MCP manager"""
        # Setup mock servers
        mock_server = MagicMock()
        mock_server.initialize = AsyncMock()
        mock_server.list_tools = AsyncMock(return_value=[
            MCPTool("test_tool", "A test tool", {"type": "object"})
        ])
        mock_server.execute_tool = AsyncMock(return_value={"result": "success"})
        mock_server.cleanup = AsyncMock()
        mock_server_class.return_value = mock_server
        
        # Create manager and initialize
        manager = MCPManager(test_config_path)
        await manager.initialize()
        
        # Check that servers were initialized
        mock_server_class.assert_called_once()
        mock_server.initialize.assert_called_once()
        
        # Test getting tools
        tools = await manager.get_all_tools()
        mock_server.list_tools.assert_called_once()
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "test_tool"
        
        # Test executing a tool
        result = await manager.execute_tool("test_tool", {"test": "value"})
        mock_server.execute_tool.assert_called_once_with("test_tool", {"test": "value"})
        assert result == {"result": "success"}
        
        # Test cleanup
        await manager.cleanup()
        mock_server.cleanup.assert_called_once()
