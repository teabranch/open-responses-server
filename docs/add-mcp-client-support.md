This Python script demonstrates how to create a client application that interacts with MCP (MetaCall Protocol) servers and a Large Language Model (LLM). The MCP client part is primarily managed by the `Server` class, which utilizes the `mcp` library.

Here's a breakdown of how the MCP client functionality is implemented and how you can understand it to write your own:

**Core Idea: MCP Client Interaction**

The script aims to:

1.  **Connect to MCP Servers:** These servers expose "tools" (functions or capabilities) that can be executed remotely. The connection is established via standard input/output (stdio).
2.  **List Available Tools:** Once connected, the client can query the MCP server to discover the tools it offers.
3.  **Execute Tools:** The client can then request the execution of a specific tool on the MCP server, providing necessary arguments.
4.  **Integrate with an LLM:** The LLM is used to understand user input and decide whether to use one of the available tools or respond directly. If a tool is needed, the LLM formats a request that the MCP client can understand and execute.

**Key Components for Writing an MCP Client (as seen in the script):**

1.  **Import Necessary MCP Libraries:**
    * `mcp.ClientSession`: This is the primary class for managing a session with an MCP server. It handles sending requests (like listing or calling tools) and receiving responses.
    * `mcp.StdioServerParameters`: This class is used to define how to start and communicate with a local MCP server that uses stdio for communication.
    * `mcp.client.stdio.stdio_client`: This is an asynchronous context manager that establishes the stdio connection to the MCP server.

2.  **`Server` Class - The MCP Client Wrapper:**
    This class encapsulates the logic for managing a connection to a single MCP server and interacting with its tools.

    * **`__init__(self, name: str, config: dict[str, Any])`:**
        * Stores the server's `name` and its `config` (loaded from `servers_config.json`). This config likely contains details on how to run the MCP server (e.g., command, arguments, environment variables).
        * Initializes `self.session: ClientSession | None = None` which will hold the active MCP session.
        * `self.exit_stack: AsyncExitStack = AsyncExitStack()`: This is crucial for managing asynchronous resources. It ensures that connections are properly closed even if errors occur.

    * **`async def initialize(self) -> None` (Key MCP Client Setup):**
        * **Determine Server Command:** It figures out the command needed to start the MCP server. It can be a direct command or `npx` (Node Package Execute) if specified.
        * **`StdioServerParameters`:** It creates an instance of `StdioServerParameters`. This tells the `mcp` library:
            * `command`: The executable to run for the MCP server.
            * `args`: Any arguments to pass to the server command.
            * `env`: Environment variables for the server process. It cleverly merges the current environment with any server-specific ones.
        * **`stdio_client(server_params)`:** This is the function that actually attempts to start the MCP server process and establish a stdio-based transport (read and write streams) with it. It's used as an async context manager (`await self.exit_stack.enter_async_context(...)`) to ensure proper cleanup.
        * **`ClientSession(read, write)`:** Once the transport (read/write streams) is established by `stdio_client`, these streams are passed to the `ClientSession` constructor. This session object will be used for all further communication with this MCP server. This is also managed by the `exit_stack`.
        * **`await session.initialize()`:** After creating the `ClientSession`, this crucial call performs the initial handshake or setup with the MCP server. This makes the session ready to use.
        * `self.session = session`: The active session is stored.
        * **Error Handling & Cleanup:** If any part of this initialization fails, it logs the error and calls `self.cleanup()` to release any acquired resources.

    * **`async def list_tools(self) -> list[Any]`:**
        * Checks if the server is initialized (`self.session` is not None).
        * **`await self.session.list_tools()`:** This is a direct call to the `mcp` library's `ClientSession` object. It sends a request to the MCP server to list its available tools.
        * The response is then parsed to create a list of `Tool` objects (a custom class in this script for representing tool details).

    * **`async def execute_tool(...)`:**
        * Checks if the server is initialized.
        * **`await self.session.call_tool(tool_name, arguments)`:** This is another direct call to the `ClientSession`. It tells the MCP server to execute the tool specified by `tool_name` with the provided `arguments`.
        * Includes a retry mechanism for transient errors.

    * **`async def cleanup(self) -> None`:**
        * Uses `self.exit_stack.aclose()` to asynchronously clean up all resources that were registered with it (like the `stdio_client` and `ClientSession`). This is very important for preventing resource leaks, especially with subprocesses and network connections.
        * Resets `self.session` and `self.stdio_context`.

3.  **Configuration (`servers_config.json` - not shown but implied):**
    You would need a JSON file (e.g., `servers_config.json`) that defines the MCP servers you want to connect to. The `main` function loads this. An example structure might be:

    ```json
    {
      "mcpServers": {
        "myPythonServer": {
          "command": "python",
          "args": ["path/to/your/mcp_server_script.py"],
          "env": {}
        },
        "myNodeServer": {
          "command": "npx",
          "args": ["your-mcp-package", "--port", "XXXX"],
          "env": {"NODE_ENV": "development"}
        }
      }
    }
    ```

4.  **Running the Client (`main` function and `asyncio.run`):**
    * The `main` async function orchestrates:
        * Loading configuration.
        * Creating `Server` instances for each configured MCP server.
        * Creating an `LLMClient`.
        * Initializing a `ChatSession` with the servers and LLM client.
        * Starting the chat session, which involves initializing the MCP servers (`await server.initialize()`).
    * `asyncio.run(main())` is the entry point for the asynchronous application.

**Steps to Write Your Own Basic MCP Client (based on this script):**

1.  **Install the `mcp` library:**
    ```bash
    pip install mcp
    ```
    (Or however the specific MCP library you intend to use is installed. The script uses a library also named `mcp`).

2.  **Define Server Parameters:**
    Decide how your MCP server will be started. For a stdio-based server (like in the example), you'll need the command and any arguments.
    ```python
    from mcp import StdioServerParameters

    server_params = StdioServerParameters(
        command="python",  # Or "node", "npx your-package", etc.
        args=["path/to/your/mcp_server_executable.py"],
        env={"SOME_ENV_VAR": "value"} # Optional
    )
    ```

3.  **Establish Connection and Session (Asynchronously):**
    Use `stdio_client` and `ClientSession` within an `AsyncExitStack` for robust resource management.
    ```python
    import asyncio
    from contextlib import AsyncExitStack
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client # Assuming this path

    async def connect_to_mcp():
        async with AsyncExitStack() as exit_stack:
            try:
                # Define server_params as above
                stdio_transport = await exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                read_stream, write_stream = stdio_transport

                session = await exit_stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
                await session.initialize()
                print("Successfully connected and initialized MCP session!")
                return session # You can now use this session
            except Exception as e:
                print(f"Error connecting to MCP server: {e}")
                return None
    ```

4.  **Interact with the Server:**
    Once you have an initialized `session` object:
    * **List Tools:**
        ```python
        # Assuming 'session' is your initialized ClientSession
        tools_response = await session.list_tools()
        # Process tools_response (structure depends on MCP server implementation)
        # The example script expects a list of tuples, e.g., [("tools", [ToolObject1, ...])]
        print("Available tools:", tools_response)
        ```
    * **Call a Tool:**
        ```python
        # Assuming 'session' is your initialized ClientSession
        tool_name_to_call = "example_tool"
        arguments_for_tool = {"param1": "value1", "count": 10}
        try:
            result = await session.call_tool(tool_name_to_call, arguments_for_tool)
            print(f"Result of '{tool_name_to_call}':", result)
        except Exception as e:
            print(f"Error calling tool '{tool_name_to_call}': {e}")
        ```

5.  **Cleanup:**
    The `AsyncExitStack` handles most of the cleanup automatically when the `async with` block exits. The `Server.cleanup()` method in the example script explicitly calls `exit_stack.aclose()`.

**In summary, the script provides a robust, asynchronous framework for an MCP client. The `Server` class is the heart of the MCP client logic, showing how to:**

* **Configure and start** an MCP server process using `StdioServerParameters`.
* **Establish communication** using `stdio_client`.
* **Create and initialize** an `ClientSession`.
* **Use the session** to `list_tools` and `call_tool`.
* **Manage resources** carefully with `AsyncExitStack`.

By studying the `Server` class and its `initialize`, `list_tools`, and `execute_tool` methods, you can adapt this pattern to connect to and interact with various MCP-compliant servers. The rest of the script (Configuration, LLMClient, ChatSession, Tool) builds upon this core MCP client functionality to create a more complete application.


# Sources

Link: https://github.com/modelcontextprotocol/python-sdk/blob/main/examples/clients/simple-chatbot/mcp_simple_chatbot/

## Requirements
```
python-dotenv>=1.0.0
requests>=2.31.0
mcp>=1.0.0
uvicorn>=0.32.1
```

### Code
File: `main.py`

```python
import asyncio
import json
import logging
import os
import shutil
from contextlib import AsyncExitStack
from typing import Any

import httpx
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.api_key = os.getenv("LLM_API_KEY")

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        with open(file_path, "r") as f:
            return json.load(f)

    @property
    def llm_api_key(self) -> str:
        """Get the LLM API key.

        Returns:
            The API key as a string.

        Raises:
            ValueError: If the API key is not found in environment variables.
        """
        if not self.api_key:
            raise ValueError("LLM_API_KEY not found in environment variables")
        return self.api_key


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.stdio_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the server connection."""
        command = (
            shutil.which("npx")
            if self.config["command"] == "npx"
            else self.config["command"]
        )
        if command is None:
            raise ValueError("The command must be a valid string and cannot be None.")

        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env={**os.environ, **self.config["env"]}
            if self.config.get("env")
            else None,
        )
        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> list[Any]:
        """List available tools from the server.

        Returns:
            A list of available tools.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        tools_response = await self.session.list_tools()
        tools = []

        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                tools.extend(
                    Tool(tool.name, tool.description, tool.inputSchema)
                    for tool in item[1]
                )

        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                logging.info(f"Executing {tool_name}...")
                result = await self.session.call_tool(tool_name, arguments)

                return result

            except Exception as e:
                attempt += 1
                logging.warning(
                    f"Error executing tool: {e}. Attempt {attempt} of {retries}."
                )
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
                    raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                self.stdio_context = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")


class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(
        self, name: str, description: str, input_schema: dict[str, Any]
    ) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: dict[str, Any] = input_schema

    def format_for_llm(self) -> str:
        """Format tool information for LLM.

        Returns:
            A formatted string describing the tool.
        """
        args_desc = []
        if "properties" in self.input_schema:
            for param_name, param_info in self.input_schema["properties"].items():
                arg_desc = (
                    f"- {param_name}: {param_info.get('description', 'No description')}"
                )
                if param_name in self.input_schema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""


class LLMClient:
    """Manages communication with the LLM provider."""

    def __init__(self, api_key: str) -> None:
        self.api_key: str = api_key

    def get_response(self, messages: list[dict[str, str]]) -> str:
        """Get a response from the LLM.

        Args:
            messages: A list of message dictionaries.

        Returns:
            The LLM's response as a string.

        Raises:
            httpx.RequestError: If the request to the LLM fails.
        """
        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "messages": messages,
            "model": "llama-3.2-90b-vision-preview",
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 1,
            "stream": False,
            "stop": None,
        }

        try:
            with httpx.Client() as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]

        except httpx.RequestError as e:
            error_message = f"Error getting LLM response: {str(e)}"
            logging.error(error_message)

            if isinstance(e, httpx.HTTPStatusError):
                status_code = e.response.status_code
                logging.error(f"Status code: {status_code}")
                logging.error(f"Response details: {e.response.text}")

            return (
                f"I encountered an error: {error_message}. "
                "Please try again or rephrase your request."
            )


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: list[Server], llm_client: LLMClient) -> None:
        self.servers: list[Server] = servers
        self.llm_client: LLMClient = llm_client

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        cleanup_tasks = [
            asyncio.create_task(server.cleanup()) for server in self.servers
        ]
        if cleanup_tasks:
            try:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            except Exception as e:
                logging.warning(f"Warning during final cleanup: {e}")

    async def process_llm_response(self, llm_response: str) -> str:
        """Process the LLM response and execute tools if needed.

        Args:
            llm_response: The response from the LLM.

        Returns:
            The result of tool execution or the original response.
        """
        import json

        try:
            tool_call = json.loads(llm_response)
            if "tool" in tool_call and "arguments" in tool_call:
                logging.info(f"Executing tool: {tool_call['tool']}")
                logging.info(f"With arguments: {tool_call['arguments']}")

                for server in self.servers:
                    tools = await server.list_tools()
                    if any(tool.name == tool_call["tool"] for tool in tools):
                        try:
                            result = await server.execute_tool(
                                tool_call["tool"], tool_call["arguments"]
                            )

                            if isinstance(result, dict) and "progress" in result:
                                progress = result["progress"]
                                total = result["total"]
                                percentage = (progress / total) * 100
                                logging.info(
                                    f"Progress: {progress}/{total} ({percentage:.1f}%)"
                                )

                            return f"Tool execution result: {result}"
                        except Exception as e:
                            error_msg = f"Error executing tool: {str(e)}"
                            logging.error(error_msg)
                            return error_msg

                return f"No server found with tool: {tool_call['tool']}"
            return llm_response
        except json.JSONDecodeError:
            return llm_response

    async def start(self) -> None:
        """Main chat session handler."""
        try:
            for server in self.servers:
                try:
                    await server.initialize()
                except Exception as e:
                    logging.error(f"Failed to initialize server: {e}")
                    await self.cleanup_servers()
                    return

            all_tools = []
            for server in self.servers:
                tools = await server.list_tools()
                all_tools.extend(tools)

            tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])

            system_message = (
                "You are a helpful assistant with access to these tools:\n\n"
                f"{tools_description}\n"
                "Choose the appropriate tool based on the user's question. "
                "If no tool is needed, reply directly.\n\n"
                "IMPORTANT: When you need to use a tool, you must ONLY respond with "
                "the exact JSON object format below, nothing else:\n"
                "{\n"
                '    "tool": "tool-name",\n'
                '    "arguments": {\n'
                '        "argument-name": "value"\n'
                "    }\n"
                "}\n\n"
                "After receiving a tool's response:\n"
                "1. Transform the raw data into a natural, conversational response\n"
                "2. Keep responses concise but informative\n"
                "3. Focus on the most relevant information\n"
                "4. Use appropriate context from the user's question\n"
                "5. Avoid simply repeating the raw data\n\n"
                "Please use only the tools that are explicitly defined above."
            )

            messages = [{"role": "system", "content": system_message}]

            while True:
                try:
                    user_input = input("You: ").strip().lower()
                    if user_input in ["quit", "exit"]:
                        logging.info("\nExiting...")
                        break

                    messages.append({"role": "user", "content": user_input})

                    llm_response = self.llm_client.get_response(messages)
                    logging.info("\nAssistant: %s", llm_response)

                    result = await self.process_llm_response(llm_response)

                    if result != llm_response:
                        messages.append({"role": "assistant", "content": llm_response})
                        messages.append({"role": "system", "content": result})

                        final_response = self.llm_client.get_response(messages)
                        logging.info("\nFinal response: %s", final_response)
                        messages.append(
                            {"role": "assistant", "content": final_response}
                        )
                    else:
                        messages.append({"role": "assistant", "content": llm_response})

                except KeyboardInterrupt:
                    logging.info("\nExiting...")
                    break

        finally:
            await self.cleanup_servers()


async def main() -> None:
    """Initialize and run the chat session."""
    config = Configuration()
    server_config = config.load_config("servers_config.json")
    servers = [
        Server(name, srv_config)
        for name, srv_config in server_config["mcpServers"].items()
    ]
    llm_client = LLMClient(config.llm_api_key)
    chat_session = ChatSession(servers, llm_client)
    await chat_session.start()


if __name__ == "__main__":
    asyncio.run(main())
```
