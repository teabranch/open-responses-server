#!/usr/bin/env python3

import argparse
import subprocess
import os
import json
import sys
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("otc_cli")

DEFAULT_HOST = os.environ.get("API_ADAPTER_HOST", "0.0.0.0")
DEFAULT_PORT = os.environ.get("API_ADAPTER_PORT", "8080")

def start_server(host=DEFAULT_HOST, port=DEFAULT_PORT):
    """Starts the FastAPI server."""
    try:
        # Ensure log directory exists
        os.makedirs("./log", exist_ok=True)
        
        # Import modules
        import uvicorn
        from .api_controller import app
        
        logger.info(f"Starting server on {host}:{port}...")
        # Run the server using uvicorn directly
        uvicorn.run("open_responses_server.server_entrypoint:app", host=host, port=int(port))
    except ImportError as e:
        logger.error(f"Error importing server module: {e}")
        logger.info("Trying to start server using subprocess...")
        try:
            subprocess.run(
                ["uvicorn", "open_responses_server.server_entrypoint:app", "--host", host, "--port", str(port)],
                check=True
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Error starting server: {e}")
        except FileNotFoundError:
            logger.error("Error: uvicorn not found. Please install it (pip install uvicorn).")
            print("\nUvicorn not found. Please install it with: pip install uvicorn")

def configure_server():
    """Allows the user to configure server settings (e.g., host, port)."""
    print("Server Configuration:")
    host = input(f"Enter host address (default: {DEFAULT_HOST}): ") or DEFAULT_HOST
    port_str = input(f"Enter port number (default: {DEFAULT_PORT}): ") or DEFAULT_PORT
    try:
        port = int(port_str)
    except ValueError:
        print("Invalid port number. Using default.")
        port = int(DEFAULT_PORT)
    
    # Configure OpenAI base URLs
    openai_internal = input(f"Enter OpenAI internal base URL (default: http://localhost:8000): ") or "http://localhost:8000"
    openai_external = input(f"Enter OpenAI external base URL (default: http://localhost:8080): ") or "http://localhost:8080"
    
    # Configure API key
    api_key = input(f"Enter OpenAI API key (default: dummy-key): ") or "dummy-key"

    # Get current directory to store the .env file
    current_dir = os.getcwd()
    env_file_path = os.path.join(current_dir, ".env")
    env_vars = {}
    
    # Read existing .env file if it exists
    if os.path.exists(env_file_path):
        with open(env_file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        key, value = line.split("=", 1)
                        env_vars[key.strip()] = value.strip()
                    except ValueError:
                        pass  # Skip lines that don't have a key=value format

    # Update environment variables
    env_vars["API_ADAPTER_HOST"] = host
    env_vars["API_ADAPTER_PORT"] = str(port)
    env_vars["OPENAI_BASE_URL_INTERNAL"] = openai_internal
    env_vars["OPENAI_BASE_URL"] = openai_external
    env_vars["OPENAI_API_KEY"] = api_key

    # Write configuration to .env file
    with open(env_file_path, "w") as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")

    print(f"Server configuration saved to {env_file_path}.")
    print("Server configured successfully. Use 'otc start' to start the server.")

def help_command():
    """Displays help information."""
    print("OpenAI Responses API Server CLI")
    print("===========================")
    print("\nUsage: otc <command> [options]")
    print("\nCommands:")
    print("  help       Show this help message.")
    print("  configure  Configure server settings (host, port, API URLs, API key).")
    print("  start      Start the FastAPI server.")
    print("\nOptions:")
    print("  --version  Show version information.")

def show_version():
    """Displays version information."""
    from open_responses_server import __version__
    print(f"OpenAI Responses API Server CLI v{__version__}")

def main():
    parser = argparse.ArgumentParser(description="CLI to manage the OpenAI Responses API Server.")
    parser.add_argument("command", nargs="?", default="help", help="Command to execute (help, configure, start)")
    parser.add_argument("--version", action="store_true", help="Show version information")

    args = parser.parse_args()

    if args.version:
        show_version()
        return

    if args.command == "start":
        start_server()
    elif args.command == "configure":
        configure_server()
    elif args.command == "help":
        help_command()
    else:
        print(f"Unknown command: {args.command}")
        help_command()

if __name__ == "__main__":
    main()