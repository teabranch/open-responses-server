import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- General Configuration ---
API_ADAPTER_HOST = os.environ.get("API_ADAPTER_HOST", "0.0.0.0")
API_ADAPTER_PORT = int(os.environ.get("API_ADAPTER_PORT", "8080"))

# --- OpenAI/LLM Provider Configuration ---
OPENAI_BASE_URL_INTERNAL = os.environ.get("OPENAI_BASE_URL_INTERNAL", "http://localhost:8000") # For direct calls from this server
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:8080") # Potentially for client-facing URLs or other proxy uses
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "dummy-key")

# --- MCP (Model Context Protocol) Configuration ---
MCP_TOOL_REFRESH_INTERVAL = int(os.environ.get("MCP_TOOL_REFRESH_INTERVAL", "10")) # In seconds
MCP_SERVERS_CONFIG_PATH = os.environ.get("MCP_SERVERS_CONFIG_PATH", "./servers_config.json") # Path to MCP server configurations

# --- Conversation History ---
MAX_CONVERSATION_HISTORY = int(os.environ.get("MAX_CONVERSATION_HISTORY", "100"))

# --- Logging Configuration ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FILE_PATH = os.environ.get("LOG_FILE_PATH", "./log/api_adapter.log")

# Ensure log directory exists
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)

# Get a logger instance for general use in this module if needed
logger = logging.getLogger(__name__)

logger.info("Configuration loaded.")
logger.info(f"API Host: {API_ADAPTER_HOST}, API Port: {API_ADAPTER_PORT}")
logger.info(f"OpenAI Base URL (Internal): {OPENAI_BASE_URL_INTERNAL}")
logger.info(f"MCP Tool Refresh Interval: {MCP_TOOL_REFRESH_INTERVAL}s")

# --- Potentially other shared constants ---
# Example: DEFAULT_MODEL = "gpt-3.5-turbo"
