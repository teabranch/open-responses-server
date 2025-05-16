"""
This function checks if a given tool name is present in the MCP cache.
"""
def is_mcp_tool(tool_name: str) -> bool:
    """
    Determines if the given tool name belongs to an MCP tool.
    
    Args:
        tool_name: The name of the tool to check
        
    Returns:
        bool: True if it's an MCP tool, False otherwise
    """
    for func in mcp_functions_cache:
        if func.get("name") == tool_name:
            return True
    return False
