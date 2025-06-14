import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Centralized Configuration ---

# API and Server Configuration
OPENAI_BASE_URL_INTERNAL = os.environ.get("OPENAI_BASE_URL_INTERNAL", "http://localhost:8000")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:8080")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "dummy-key")
API_ADAPTER_HOST = os.environ.get("API_ADAPTER_HOST", "0.0.0.0")
API_ADAPTER_PORT = int(os.environ.get("API_ADAPTER_PORT", "8080"))

# MCP Configuration
MCP_TOOL_REFRESH_INTERVAL = int(os.environ.get("MCP_TOOL_REFRESH_INTERVAL", "10"))
MCP_SERVERS_CONFIG_PATH = os.environ.get("MCP_SERVERS_CONFIG_PATH", "src/open_responses_server/servers_config.json")

# Conversation History Configuration
MAX_CONVERSATION_HISTORY = int(os.environ.get("MAX_CONVERSATION_HISTORY", "100"))
MAX_TOOL_CALL_ITERATIONS = int(os.environ.get("MAX_TOOL_CALL_ITERATIONS", "25"))


# --- Logging Configuration ---

def setup_logging():
    """Configures the global logger."""
    log_dir = "./log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "api_adapter.log")),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("api_adapter")
    logger.info("Logging configured.")
    return logger

# Initialize logging
logger = setup_logging()

logger.info("Configuration loaded:")
logger.info(f"  OPENAI_BASE_URL_INTERNAL: {OPENAI_BASE_URL_INTERNAL}")
logger.info(f"  OPENAI_BASE_URL: {OPENAI_BASE_URL}")
logger.info(f"  API_ADAPTER_HOST: {API_ADAPTER_HOST}")
logger.info(f"  API_ADAPTER_PORT: {API_ADAPTER_PORT}")
logger.info(f"  MCP_TOOL_REFRESH_INTERVAL: {MCP_TOOL_REFRESH_INTERVAL}")
logger.info(f"  MCP_SERVERS_CONFIG_PATH: {MCP_SERVERS_CONFIG_PATH}")
logger.info(f"  MAX_CONVERSATION_HISTORY: {MAX_CONVERSATION_HISTORY}")
logger.info(f"  MAX_TOOL_CALL_ITERATIONS: {MAX_TOOL_CALL_ITERATIONS}") 