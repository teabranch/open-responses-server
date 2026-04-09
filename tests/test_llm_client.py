"""
Tests for the LLM client module.
"""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import httpx

from open_responses_server.common.config import STREAM_TIMEOUT
from open_responses_server.common.llm_client import (
    LLMClient,
    BACKEND_CONNECT_TIMEOUT,
    get_backend_timeout,
    startup_llm_client,
    shutdown_llm_client,
)


@pytest.fixture(autouse=True)
def reset_llm_client():
    """Reset LLMClient singleton between tests."""
    LLMClient._client = None
    yield
    LLMClient._client = None


class TestLLMClient:
    def test_backend_timeout_uses_long_read_and_short_connect(self):
        """Backend timeout should keep connect short while allowing slow reads."""
        timeout = get_backend_timeout()
        assert timeout.connect == BACKEND_CONNECT_TIMEOUT
        assert timeout.pool == BACKEND_CONNECT_TIMEOUT
        assert timeout.read == STREAM_TIMEOUT
        assert timeout.write == STREAM_TIMEOUT

    @pytest.mark.asyncio
    async def test_get_client_creates_instance(self):
        """get_client creates an httpx.AsyncClient on first call."""
        client = await LLMClient.get_client()
        assert client is not None
        assert isinstance(client, httpx.AsyncClient)
        assert client.timeout.connect == BACKEND_CONNECT_TIMEOUT
        assert client.timeout.read == STREAM_TIMEOUT
        await client.aclose()

    @pytest.mark.asyncio
    async def test_get_client_returns_singleton(self):
        """get_client returns the same instance on repeated calls."""
        client1 = await LLMClient.get_client()
        client2 = await LLMClient.get_client()
        assert client1 is client2
        await client1.aclose()

    @pytest.mark.asyncio
    async def test_close_client(self):
        """close_client closes the client and resets to None."""
        await LLMClient.get_client()
        assert LLMClient._client is not None
        await LLMClient.close_client()
        assert LLMClient._client is None

    @pytest.mark.asyncio
    async def test_close_client_when_none(self):
        """close_client does nothing when no client exists."""
        assert LLMClient._client is None
        await LLMClient.close_client()
        assert LLMClient._client is None

    @pytest.mark.asyncio
    async def test_startup_llm_client(self):
        """startup_llm_client initializes the client."""
        await startup_llm_client()
        assert LLMClient._client is not None
        await LLMClient._client.aclose()

    @pytest.mark.asyncio
    async def test_shutdown_llm_client(self):
        """shutdown_llm_client closes the client."""
        await startup_llm_client()
        await shutdown_llm_client()
        assert LLMClient._client is None
