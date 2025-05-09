#!/usr/bin/env python3
import json
import random
import sys
import asyncio
from typing import Dict, Any, List, Tuple, Optional

from mcp import ServerSession
from mcp.server.stdio import stdio_server


class RandomNumberServer:
    """MCP server that provides tools for generating random numbers"""
    
    def __init__(self):
        self.tools = [
            {
                "name": "generate_random_number",
                "description": "Generate a random number between min and max values",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "min": {
                            "type": "integer",
                            "description": "Minimum value (inclusive)"
                        },
                        "max": {
                            "type": "integer",
                            "description": "Maximum value (inclusive)"
                        }
                    },
                    "required": ["min", "max"]
                }
            },
            {
                "name": "generate_random_numbers",
                "description": "Generate a list of random numbers",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "min": {
                            "type": "integer",
                            "description": "Minimum value (inclusive)"
                        },
                        "max": {
                            "type": "integer",
                            "description": "Maximum value (inclusive)"
                        },
                        "count": {
                            "type": "integer",
                            "description": "Number of random numbers to generate"
                        }
                    },
                    "required": ["min", "max", "count"]
                }
            },
            {
                "name": "flip_coin",
                "description": "Simulate a coin flip, returns 'heads' or 'tails'",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "roll_dice",
                "description": "Roll a dice with specified number of sides",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "sides": {
                            "type": "integer",
                            "description": "Number of sides on the dice",
                            "default": 6
                        }
                    },
                    "required": []
                }
            }
        ]
    
    async def generate_random_number(self, args: Dict[str, Any]) -> int:
        """Generate a random integer between min and max (inclusive)"""
        min_val = args["min"]
        max_val = args["max"]
        return random.randint(min_val, max_val)
    
    async def generate_random_numbers(self, args: Dict[str, Any]) -> List[int]:
        """Generate a list of random integers"""
        min_val = args["min"]
        max_val = args["max"]
        count = args["count"]
        return [random.randint(min_val, max_val) for _ in range(count)]
    
    async def flip_coin(self, _: Dict[str, Any]) -> str:
        """Simulate a coin flip"""
        return random.choice(["heads", "tails"])
    
    async def roll_dice(self, args: Dict[str, Any]) -> int:
        """Roll a dice with specified number of sides"""
        sides = args.get("sides", 6)
        return random.randint(1, sides)
    
    async def handle_call(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Route tool calls to the appropriate method"""
        if tool_name == "generate_random_number":
            return await self.generate_random_number(args)
        elif tool_name == "generate_random_numbers":
            return await self.generate_random_numbers(args)
        elif tool_name == "flip_coin":
            return await self.flip_coin(args)
        elif tool_name == "roll_dice":
            return await self.roll_dice(args)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")


async def main():
    """Main function to start the MCP server"""
    server = RandomNumberServer()
    
    async def on_list_tools() -> List[Tuple[str, List[Dict[str, Any]]]]:
        return [("tools", server.tools)]
    
    async def on_call_tool(tool_name: str, args: Dict[str, Any]) -> Any:
        return await server.handle_call(tool_name, args)
    
    session = ServerSession(
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool
    )
    
    async with stdio_server(session) as _:
        # The server will run until the connection is closed
        while True:
            await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)