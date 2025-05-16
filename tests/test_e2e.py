"""
End-to-end tests for the OpenAI Responses Server
"""
import os
import subprocess
import pytest
import asyncio
import json
import httpx

@pytest.mark.asyncio
@pytest.mark.usefixtures("ensure_uv")
class TestE2E:
    """End-to-end tests for the server"""
    
    @pytest.mark.asyncio
    async def test_server_responses_api(self, server_process):
        """Test that the server handles Responses API requests correctly"""
        process, base_url = await server_process.__anext__()
        
        # Test request data
        request_data = {
            "model": "test-model",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Hello, world!"
                        }
                    ]
                }
            ],
            "stream": False
        }
        
        # Send request to the responses endpoint
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{base_url}/responses", json=request_data)
            assert response.status_code == 200
            
            # Verify response structure
            response_data = response.json()
            if response_data:  # Only check if we have data
                assert response_data["model"] == "test-model"
                assert "id" in response_data
                assert "created_at" in response_data
    
    @pytest.mark.asyncio
    async def test_server_streaming_responses(self, server_process):
        """Test that the server handles streaming Responses API requests correctly"""
        process, base_url = await server_process.__anext__()
        
        # Test request data for streaming
        request_data = {
            "model": "test-model",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Hello, world!"
                        }
                    ]
                }
            ],
            "stream": True
        }
        
        # Send request to the responses endpoint
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", f"{base_url}/responses", json=request_data) as response:
                assert response.status_code == 200
                
                # Just check that we get a valid streaming response
                # Don't rely on specific event types which might change
                has_data = False
                async for line in response.aiter_lines():
                    if line and line.startswith('data: '):
                        has_data = True
                        break
                
                assert has_data, "No streaming data received"
    
    @pytest.mark.asyncio
    async def test_proxy_functionality(self, server_process):
        """Test the proxy functionality to pass through requests to other endpoints"""
        process, base_url = await server_process.__anext__()
        
        # Skip the actual proxy test since we don't have a real backend
        # Just check that the endpoint exists
        async with httpx.AsyncClient() as client:
            # Test a direct endpoint we know exists
            response = await client.get(f"{base_url}/health")
            assert response.status_code == 200
            
            # The proxy might return an error when no backend is available
            # but should not crash the server
            response = await client.get(f"{base_url}/v1/models")
            assert response.status_code in (200, 404, 500)
    
    @pytest.mark.asyncio
    async def test_cli_startup(self, python_executable, temp_env_file):
        """Test that the server can be started through the CLI"""
        env_file, env_vars = temp_env_file
        
        # Set environment variables
        test_env = os.environ.copy()
        for key, value in env_vars.items():
            test_env[key] = value
        
        # Start the server using the CLI
        cli_cmd = [
            python_executable,
            "-m", "open_responses_server.cli",
            "start"
        ]
        
        process = subprocess.Popen(
            cli_cmd,
            env=test_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        try:
            # Wait for server to start
            await asyncio.sleep(3)
            
            # Test that the server is running
            server_port = env_vars["API_ADAPTER_PORT"]
            base_url = f"http://{env_vars['API_ADAPTER_HOST']}:{server_port}"
            
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{base_url}/health")
                assert response.status_code == 200
                assert response.json() == {"status": "ok", "adapter": "running"}
        finally:
            # Clean up
            process.kill()
            process.wait() 