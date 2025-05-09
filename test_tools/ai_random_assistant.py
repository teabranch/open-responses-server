#!/usr/bin/env python3
import asyncio
import json
import logging
import os
from contextlib import AsyncExitStack
from typing import Any, Dict, List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class RandomToolServer:
    """Manages the random number MCP server connection"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.server_config = next(iter(self.config["mcpServers"].values()))
        self.exit_stack = AsyncExitStack()
        self.session = None
    
    async def initialize(self):
        """Initialize connection to the MCP server"""
        try:
            server_params = StdioServerParameters(
                command=self.server_config["command"],
                args=self.server_config["args"],
                env=self.server_config.get("env", {})
            )
            
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read_stream, write_stream = stdio_transport
            
            session = await self.exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()
            self.session = session
            
            logging.info("MCP random number server initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error initializing MCP server: {e}")
            await self.cleanup()
            return False
    
    async def list_tools(self):
        """List available tools from the MCP server"""
        if not self.session:
            raise RuntimeError("MCP server not initialized")
        
        tools_response = await self.session.list_tools()
        return tools_response
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]):
        """Execute a tool on the MCP server"""
        if not self.session:
            raise RuntimeError("MCP server not initialized")
        
        return await self.session.call_tool(tool_name, arguments)
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()
        self.session = None


class AIRandomAssistant:
    """AI assistant that uses random number tools"""
    
    def __init__(self):
        self.server = RandomToolServer("test_tools/random_mcp_config.json")
        self.available_tools = []
    
    async def initialize(self):
        """Initialize the assistant"""
        if not await self.server.initialize():
            return False
        
        # Get available tools
        tools_response = await self.server.list_tools()
        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                self.available_tools = item[1]
                break
        
        logging.info(f"Available tools: {len(self.available_tools)}")
        return True
    
    async def process_user_request(self, request: str):
        """Process a user request by calling appropriate random tools"""
        request = request.lower()
        
        if "random number" in request:
            if "between" in request:
                # Try to extract min and max values
                try:
                    import re
                    numbers = re.findall(r'\d+', request)
                    if len(numbers) >= 2:
                        min_val, max_val = int(numbers[0]), int(numbers[1])
                    else:
                        min_val, max_val = 1, 100  # Default range
                except:
                    min_val, max_val = 1, 100  # Default range
                
                result = await self.server.execute_tool(
                    "generate_random_number", 
                    {"min": min_val, "max": max_val}
                )
                return f"Here's a random number between {min_val} and {max_val}: {result}"
            
        elif "random numbers" in request or "multiple" in request:
            # Generate multiple random numbers
            try:
                import re
                count = 5  # Default count
                min_val, max_val = 1, 100  # Default range
                
                # Look for count
                count_match = re.search(r'(\d+)\s+(?:random\s+)?numbers', request)
                if count_match:
                    count = int(count_match.group(1))
                
                # Look for range
                range_match = re.search(r'between\s+(\d+)\s+and\s+(\d+)', request)
                if range_match:
                    min_val = int(range_match.group(1))
                    max_val = int(range_match.group(2))
            except:
                count = 5
                min_val, max_val = 1, 100
            
            result = await self.server.execute_tool(
                "generate_random_numbers", 
                {"min": min_val, "max": max_val, "count": count}
            )
            return f"Here are {count} random numbers between {min_val} and {max_val}: {result}"
            
        elif "flip" in request and ("coin" in request or "toss" in request):
            result = await self.server.execute_tool("flip_coin", {})
            return f"I flipped a coin and got: {result.upper()}!"
            
        elif "roll" in request and ("dice" in request or "die" in request):
            sides = 6  # Default is a standard 6-sided die
            
            # Check for D&D style dice notation (d20, d6, etc.)
            if "d20" in request:
                sides = 20
            elif "d12" in request:
                sides = 12
            elif "d10" in request:
                sides = 10
            elif "d8" in request:
                sides = 8
            elif "d4" in request:
                sides = 4
                
            result = await self.server.execute_tool("roll_dice", {"sides": sides})
            return f"I rolled a {sides}-sided die and got: {result}!"
            
        else:
            # Help message if input doesn't match any patterns
            return (
                "I can help with random numbers! Try asking me to:\n"
                "- Generate a random number (between X and Y)\n"
                "- Generate multiple random numbers\n"
                "- Flip a coin\n"
                "- Roll a dice (standard 6-sided or specify d20, d12, etc.)"
            )
    
    async def run_interactive_session(self):
        """Run an interactive session with the assistant"""
        print("\n===== Random Number AI Assistant =====")
        print("(Type 'exit' or 'quit' to end the session)")
        
        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("Goodbye!")
                    break
                
                # Process request and get response
                response = await self.process_user_request(user_input)
                print(f"\nAI: {response}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.server.cleanup()


async def main():
    """Main function"""
    assistant = AIRandomAssistant()
    
    if await assistant.initialize():
        try:
            await assistant.run_interactive_session()
        finally:
            await assistant.cleanup()
    else:
        logging.error("Failed to initialize the AI assistant")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass