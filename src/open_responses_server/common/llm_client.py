import httpx
from .config import OPENAI_BASE_URL_INTERNAL, OPENAI_API_KEY, logger

class LLMClient:
    """
    An asynchronous client for interacting with the LLM API.
    """
    _client: httpx.AsyncClient | None = None

    @classmethod
    async def get_client(cls) -> httpx.AsyncClient:
        """
        Returns the singleton instance of the httpx.AsyncClient.
        Initializes it if it doesn't exist.
        """
        if cls._client is None:
            logger.info(f"Initializing LLM client... ({OPENAI_BASE_URL_INTERNAL})")
            cls._client = httpx.AsyncClient(
                base_url=OPENAI_BASE_URL_INTERNAL,
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                timeout=httpx.Timeout(120.0)
            )
        return cls._client

    @classmethod
    async def close_client(cls):
        """
        Closes the httpx.AsyncClient session.
        """
        if cls._client:
            logger.info("Closing LLM client...")
            await cls._client.aclose()
            cls._client = None

# You can also define helper functions to use the client, as per the plan.
# For now, we'll just provide the client management.

async def startup_llm_client():
    """Function to be called on application startup."""
    await LLMClient.get_client()

async def shutdown_llm_client():
    """Function to be called on application shutdown."""
    await LLMClient.close_client() 