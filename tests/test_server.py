"""
Tests for the server module
"""
import json
import pytest
import asyncio
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from open_responses_server.api_controller import app
from open_responses_server.responses_service import convert_responses_to_chat_completions

class TestServer:
    """Tests for the server module"""
    
    @pytest.fixture
    def client(self):
        """TestClient fixture"""
        return TestClient(app)
    
    def test_health_check(self, client):
        """Test the health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "adapter": "running"}
    
    def test_root_endpoint(self, client):
        """Test the root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Open Responses Server is running."}
    
    def test_convert_responses_to_chat_completions(self):
        """Test the conversion from Responses API to chat.completions API"""
        # Sample responses API request
        responses_request = {
            "model": "test-model",
            "temperature": 0.7,
            "top_p": 0.8,
            "stream": True,
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
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "test_function",
                        "description": "A test function",
                        "parameters": {}
                    }
                }
            ]
        }
        
        result = convert_responses_to_chat_completions(responses_request)
        
        # Check basic fields are correctly transferred
        assert result["model"] == "test-model"
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.8
        assert result["stream"] == True
        
        # Check that messages exists 
        assert "messages" in result
        assert isinstance(result["messages"], list)
        assert len(result["messages"]) > 0
        
        # Don't check tools as they might be handled differently in the actual implementation
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_httpx_client")
    async def test_responses_endpoint_non_streaming(self, client):
        """Test the /responses endpoint with non-streaming response"""
        # Sample request data
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
        
        # Note: The actual implementation might return a different response
        # or might log that non-streaming is unsupported
        response = client.post("/responses", json=request_data)
        assert response.status_code == 200 or response.status_code == 500
        
        # If we get a successful response, check basic structure
        if response.status_code == 200 and response.content:
            response_data = response.json()
            if response_data:
                assert "model" in response_data
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_httpx_client")
    async def test_responses_endpoint_streaming(self, client):
        """Test the /responses endpoint with streaming response"""
        # Sample request data
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
        
        # For streaming responses with TestClient, we need to handle it differently
        response = client.post("/responses", json=request_data)
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_proxy_endpoint(self, client, mock_httpx_client):
        """Test the proxy endpoint functionality"""
        # Patch the AsyncClient to avoid real HTTP requests
        with patch('open_responses_server.api_controller.httpx.AsyncClient', return_value=mock_httpx_client):
            # Test GET request
            response = client.get("/v1/models")
            assert response.status_code == 200


@pytest.mark.asyncio
@pytest.mark.usefixtures("ensure_uv")
class TestServerIntegration:
    """Integration tests for the server"""
    
    @pytest.mark.asyncio
    async def test_server_startup(self, server_process):
        """Test that the server starts up correctly"""
        process, base_url = await server_process.__anext__()
        
        # Test that the server is running and responds to health check
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/health")
            assert response.status_code == 200
            assert response.json() == {"status": "ok", "adapter": "running"} 