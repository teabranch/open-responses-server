#!/usr/bin/env python3
import asyncio
import json
import logging
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

async def test_random_mcp_server():
    """Test the random number MCP server"""
    # Define server parameters for our random number MCP server
    server_params = StdioServerParameters(
        command="python",
        args=["test_tools/random_number_mcp_server.py"],
    )
    
    async with AsyncExitStack() as exit_stack:
        try:
            # Start the MCP server process and establish connection
            logging.info("Starting random number MCP server...")
            stdio_transport = await exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read_stream, write_stream = stdio_transport
            
            # Create and initialize client session
            session = await exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()
            logging.info("MCP session initialized successfully")
            
            # List available tools
            logging.info("Listing available tools...")
            tools_response = await session.list_tools()
            logging.info(f"Available tools: {json.dumps(tools_response, indent=2)}")
            
            # Test generate_random_number
            logging.info("\nTesting generate_random_number...")
            result1 = await session.call_tool("generate_random_number", {"min": 1, "max": 100})
            logging.info(f"Random number between 1 and 100: {result1}")
            
            # Test generate_random_numbers
            logging.info("\nTesting generate_random_numbers...")
            result2 = await session.call_tool(
                "generate_random_numbers", 
                {"min": 1, "max": 10, "count": 5}
            )
            logging.info(f"5 random numbers between 1 and 10: {result2}")
            
            # Test flip_coin
            logging.info("\nTesting flip_coin...")
            result3 = await session.call_tool("flip_coin", {})
            logging.info(f"Coin flip result: {result3}")
            
            # Test roll_dice
            logging.info("\nTesting roll_dice...")
            result4 = await session.call_tool("roll_dice", {"sides": 20})
            logging.info(f"20-sided dice roll: {result4}")
            
            # Test roll_dice with default value
            logging.info("\nTesting roll_dice with default sides...")
            result5 = await session.call_tool("roll_dice", {})
            logging.info(f"Default dice roll (6-sided): {result5}")
            
            logging.info("\nAll tests completed successfully!")
            
        except Exception as e:
            logging.error(f"Error while testing MCP server: {e}")
            raise

if __name__ == "__main__":
    try:
        asyncio.run(test_random_mcp_server())
    except KeyboardInterrupt:
        logging.info("Test interrupted by user")
    except Exception as e:
        logging.error(f"Test failed: {e}")