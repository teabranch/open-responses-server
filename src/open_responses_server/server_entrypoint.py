import uvicorn
from .api_controller import app
from .common.config import API_ADAPTER_HOST, API_ADAPTER_PORT, logger

if __name__ == "__main__":
    logger.info(f"Starting Open Responses Server on {API_ADAPTER_HOST}:{API_ADAPTER_PORT}")
    uvicorn.run(app, host=API_ADAPTER_HOST, port=API_ADAPTER_PORT, reload=True) 