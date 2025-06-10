import httpx
import logging
import json
from typing import Dict, Any, AsyncGenerator

from .config import OPENAI_BASE_URL_INTERNAL, OPENAI_API_KEY

logger = logging.getLogger(__name__)

class LLMClient:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LLMClient, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'): # Ensure __init__ runs only once
            self.base_url = OPENAI_BASE_URL_INTERNAL
            self.headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            self.http_client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self.headers,
                timeout=httpx.Timeout(120.0) # Default timeout, can be overridden
            )
            self._initialized = True
            logger.info(f"LLMClient initialized with base_url: {self.base_url}")

    async def close(self):
        await self.http_client.aclose()
        logger.info("LLMClient closed.")

    async def post_chat_completions_stream(
        self, 
        payload: Dict[str, Any],
        request_headers: Dict[str, str] = None
    ) -> AsyncGenerator[bytes, None]:
        """Makes a streaming POST request to /chat/completions."""
        url = "/chat/completions" # Assuming base_url already includes /v1 or similar prefix if needed
        
        # Prepare headers for the internal request, prioritizing specific request_headers if provided
        final_headers = self.headers.copy()
        if request_headers:
            # Filter out headers that httpx should manage or are sensitive
            filtered_request_headers = { 
                k: v for k, v in request_headers.items() 
                if k.lower() not in ["host", "content-length", "accept-encoding", "authorization"]
            }
            final_headers.update(filtered_request_headers)
        final_headers["Authorization"] = f"Bearer {OPENAI_API_KEY}" # Ensure our key is used

        logger.debug(f"Streaming POST to {self.base_url}{url} with payload: {json.dumps(payload)}")
        try:
            async with self.http_client.stream(
                "POST",
                url,
                json=payload,
                headers=final_headers,
                timeout=None # Streaming requests often need longer timeouts
            ) as response:
                response.raise_for_status() # Raise HTTPStatusError for bad responses (4xx or 5xx)
                async for chunk in response.aiter_bytes():
                    yield chunk
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during streaming chat completions: {e.response.status_code} - {e.response.text}")
            # Yield an error structure if desired, or re-raise
            error_payload = json.dumps({"error": {"message": e.response.text, "type": "llm_request_failed", "code": e.response.status_code}})
            yield f"data: {error_payload}\n\n".encode('utf-8')
            yield b"data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Error during streaming chat completions: {e}")
            error_payload = json.dumps({"error": {"message": str(e), "type": "stream_error"}})
            yield f"data: {error_payload}\n\n".encode('utf-8')
            yield b"data: [DONE]\n\n"

    async def post_chat_completions_non_stream(
        self, 
        payload: Dict[str, Any],
        request_headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """Makes a non-streaming POST request to /chat/completions."""
        url = "/chat/completions"
        final_headers = self.headers.copy()
        if request_headers:
            filtered_request_headers = { 
                k: v for k, v in request_headers.items() 
                if k.lower() not in ["host", "content-length", "accept-encoding", "authorization"]
            }
            final_headers.update(filtered_request_headers)
        final_headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"

        logger.debug(f"Non-streaming POST to {self.base_url}{url} with payload: {json.dumps(payload)}")
        try:
            response = await self.http_client.post(
                url,
                json=payload,
                headers=final_headers,
                timeout=120.0 
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during non-streaming chat completions: {e.response.status_code} - {e.response.text}")
            return {"error": {"message": e.response.text, "type": "llm_request_failed", "code": e.response.status_code}}
        except Exception as e:
            logger.error(f"Error during non-streaming chat completions: {e}")
            return {"error": {"message": str(e), "type": "request_error"}}

    async def proxy_request(
        self, 
        method: str, 
        path: str, 
        query_params: bytes, 
        headers: Dict[str, str],
        content: bytes,
        stream: bool = False
    ) -> httpx.Response:
        """Proxies a generic request to the LLM API."""
        # Construct the target URL
        # Ensure path doesn't have leading slash if base_url ends with one, or vice-versa
        # Assuming OPENAI_BASE_URL_INTERNAL is like 'http://host:port/v1'
        # and path is like 'models' or 'embeddings'
        url = httpx.URL(path=path, query=query_params)

        # Prepare headers for the proxy request
        proxy_headers = {k: v for k, v in headers.items() if k.lower() not in ['host', 'content-length']}
        proxy_headers["Authorization"] = f"Bearer {OPENAI_API_KEY}" # Ensure our API key is used

        logger.info(f"Proxying {method} to {self.base_url}{url}")
        
        rp_req = self.http_client.build_request(
            method, url, headers=proxy_headers, content=content, timeout=None
        )
        
        try:
            rp_resp = await self.http_client.send(rp_req, stream=stream)
            # For streaming responses, the caller is responsible for handling the stream
            # For non-streaming, the caller can use rp_resp.read() or rp_resp.json()
            return rp_resp
        except httpx.HTTPStatusError as e:
            logger.error(f"Proxy HTTP error: {e.response.status_code} - {e.response.text}")
            # Return a custom error response or re-raise
            # For simplicity, re-raising here, but a Response object might be better
            raise
        except Exception as e:
            logger.error(f"Error during proxy request: {e}")
            raise

# Singleton instance
llm_client = LLMClient()
