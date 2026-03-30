import os
import sys
import subprocess
import pytest
import asyncio
import socket
import time
from pathlib import Path

# Helper function to find an available port
def find_available_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

# Helper to detect Python version and set up path
@pytest.fixture(scope="session")
def python_executable():
    """Detect Python executable and ensure it's in PATH."""
    # Get Python version from sys.version_info
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    # Log Python version info
    print(f"Running tests with Python {python_version}")
    
    # Return the current Python executable path
    return sys.executable

# Fixture for uv installation check
@pytest.fixture(scope="session")
def ensure_uv():
    """Ensure uv is installed and available."""
    try:
        # Check if uv is installed
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
        print("uv package manager detected")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.fail("uv package manager not found. Please install it with 'pip install uv'")
    
    return True

# Fixture to create a temporary .env file for testing
@pytest.fixture(scope="function")
def temp_env_file(tmp_path):
    """Create a temporary .env file for testing."""
    env_file = tmp_path / ".env"
    
    # Define test environment variables
    test_env_vars = {
        "API_ADAPTER_HOST": "127.0.0.1",
        "API_ADAPTER_PORT": str(find_available_port()),
        "OPENAI_BASE_URL_INTERNAL": "http://localhost:8000",
        "OPENAI_BASE_URL": "http://localhost:8080",
        "OPENAI_API_KEY": "test-api-key"
    }
    
    # Write environment variables to file
    with open(env_file, "w") as f:
        for key, value in test_env_vars.items():
            f.write(f"{key}={value}\n")
    
    # Return the file path and the environment variables
    return env_file, test_env_vars

# Fixture to start the server for integration tests
@pytest.fixture(scope="function")
async def server_process(python_executable, temp_env_file):
    """Start the server as a separate process for testing."""
    env_file, env_vars = temp_env_file
    
    # Set environment variables from temp env file
    test_env = os.environ.copy()
    for key, value in env_vars.items():
        test_env[key] = value
    
    # Start server
    server_port = int(env_vars["API_ADAPTER_PORT"])
    server_cmd = [
        python_executable, 
        "-m", "uvicorn", 
        "open_responses_server.server_entrypoint:app", 
        "--host", "127.0.0.1", 
        "--port", str(server_port)
    ]
    
    # Log the command being run
    print(f"Starting server with command: {' '.join(server_cmd)}")
    
    process = subprocess.Popen(
        server_cmd,
        env=test_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    server_ready = False
    start_time = time.time()
    while not server_ready and time.time() - start_time < 10:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect(('127.0.0.1', server_port))
                server_ready = True
        except ConnectionRefusedError:
            await asyncio.sleep(0.5)
    
    if not server_ready:
        process.kill()
        stdout, stderr = process.communicate()
        print(f"Server stdout: {stdout.decode()}")
        print(f"Server stderr: {stderr.decode()}")
        pytest.fail("Server failed to start within the expected time")
    
    # Wait a bit more to ensure the server is fully operational
    await asyncio.sleep(1)
    
    # Yield the process and base URL as a tuple
    yield process, f"http://127.0.0.1:{server_port}"
    
    # Clean up after test
    process.kill()
    process.wait()

# Mock dependencies for unit tests
@pytest.fixture
def mock_httpx_client(monkeypatch):
    """Mock the httpx client for unit tests."""
    class MockAsyncClient:
        base_url = "http://mock-llm:8000"

        async def post(self, url, **kwargs):
            class MockResponse:
                status_code = 200
                
                async def aread(self):
                    if "chat/completions" in url:
                        return b'{"id": "mock-id", "model": "test-model", "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": "This is a test response"}}]}'
                    return b'{}'
                
                async def aiter_bytes(self):
                    yield b'data: {"id": "mock-id", "model": "test-model", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "This"}}]}\n\n'
                    yield b'data: {"id": "mock-id", "model": "test-model", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": " is"}}]}\n\n'
                    yield b'data: {"id": "mock-id", "model": "test-model", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": " a"}}]}\n\n'
                    yield b'data: {"id": "mock-id", "model": "test-model", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": " test"}}]}\n\n'
                    yield b'data: {"id": "mock-id", "model": "test-model", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": " response"}}]}\n\n'
                    yield b'data: [DONE]\n\n'
                
                async def json(self):
                    if "chat/completions" in url:
                        return {
                            "id": "mock-id",
                            "model": "test-model",
                            "object": "chat.completion",
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "This is a test response"
                                    }
                                }
                            ]
                        }
                    return {}
                
                async def __aenter__(self):
                    return self
                
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    pass
            
            return MockResponse()
        
        async def get(self, url, **kwargs):
            class MockResponse:
                status_code = 200
                content = b'{"status": "ok", "adapter": "running"}'
                
                async def json(self):
                    if "/health" in url:
                        return {"status": "ok", "adapter": "running"}
                    return {"status": "ok"}
                
                async def __aenter__(self):
                    return self
                
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    pass
            
            return MockResponse()
        
        async def request(self, method, url, **kwargs):
            """Mock for generic requests"""
            class MockResponse:
                status_code = 200
                content = b'{"status": "ok"}'
                headers = {}
                
                async def json(self):
                    return {"status": "ok"}
                
                async def __aenter__(self):
                    return self
                
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    pass
            
            return MockResponse()
            
        async def stream(self, method, url, **kwargs):
            """Mock for streaming requests"""
            class MockStreamResponse:
                status_code = 200
                
                async def aenter(self):
                    return self
                
                async def __aenter__(self):
                    return self
                
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    pass
                
                async def aread(self):
                    return b'{}'
                
                async def aiter_bytes(self):
                    if "chat/completions" in url:
                        yield b'data: {"id": "mock-id", "model": "test-model", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "This is a test response"}}]}\n\n'
                        yield b'data: [DONE]\n\n'
                    else:
                        yield b'{}'
            
            return MockStreamResponse()
        
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
    
    monkeypatch.setattr("httpx.AsyncClient", MockAsyncClient)
    return MockAsyncClient()


class MockStreamResponse:
    """Mock streaming response with aiter_lines() for testing process_chat_completions_stream."""
    def __init__(self, lines):
        self._lines = lines
        self.status_code = 200

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self):
        return b'{}'

    async def aiter_bytes(self):
        for line in self._lines:
            yield (line + "\n\n").encode()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


@pytest.fixture
def mock_stream_response():
    """Factory fixture to create MockStreamResponse instances."""
    def _make(lines):
        return MockStreamResponse(lines)
    return _make


@pytest.fixture(autouse=True)
def clean_conversation_history():
    """Clear conversation history between tests."""
    from open_responses_server.responses_service import conversation_history
    conversation_history.clear()
    yield
    conversation_history.clear()


@pytest.fixture
def mock_mcp_manager_fixture(monkeypatch):
    """Patch the mcp_manager singleton with configurable mocks."""
    from unittest.mock import MagicMock, AsyncMock
    from open_responses_server.common.mcp_manager import MCPManager

    mock_mgr = MagicMock(spec=MCPManager)
    mock_mgr.mcp_functions_cache = []
    mock_mgr.mcp_servers = []
    mock_mgr._server_tool_mapping = {}
    mock_mgr.is_mcp_tool = MagicMock(return_value=False)
    mock_mgr.execute_mcp_tool = AsyncMock(return_value=None)
    mock_mgr.get_mcp_tools = MagicMock(return_value=[])
    mock_mgr.startup_mcp_servers = AsyncMock()
    mock_mgr.shutdown_mcp_servers = AsyncMock()

    # Patch at all import points
    monkeypatch.setattr("open_responses_server.api_controller.mcp_manager", mock_mgr)
    monkeypatch.setattr("open_responses_server.responses_service.mcp_manager", mock_mgr)
    monkeypatch.setattr("open_responses_server.chat_completions_service.mcp_manager", mock_mgr)

    return mock_mgr


@pytest.fixture
def mock_llm_client_fixture(monkeypatch):
    """Patch LLMClient.get_client to return a configurable mock async client."""
    from unittest.mock import AsyncMock, MagicMock

    mock_client = MagicMock()
    mock_client.base_url = "http://mock-llm:8000"

    async def _get_client():
        return mock_client

    monkeypatch.setattr(
        "open_responses_server.common.llm_client.LLMClient.get_client",
        _get_client
    )
    monkeypatch.setattr(
        "open_responses_server.api_controller.LLMClient.get_client",
        _get_client
    )
    monkeypatch.setattr(
        "open_responses_server.chat_completions_service.LLMClient.get_client",
        _get_client
    )

    return mock_client 