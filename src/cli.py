#!/usr/bin/env python3

import argparse
import subprocess
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DEFAULT_HOST = os.environ.get("API_ADAPTER_HOST", "0.0.0.0")
DEFAULT_PORT = os.environ.get("API_ADAPTER_PORT", "8080")

def start_server(host=DEFAULT_HOST, port=DEFAULT_PORT):
    """Starts the FastAPI server."""
    try:
        # Import path adjusted to work as a package
        from openai_to_codex_wrapper import server
        import uvicorn
        
        print(f"Starting server on {host}:{port}...")
        # Run the server using uvicorn directly from Python
        uvicorn.run("openai_to_codex_wrapper.server:app", host=host, port=int(port), reload=False)
    except ImportError as e:
        print(f"Error importing server module: {e}")
        print("Trying to start server using subprocess...")
        try:
            subprocess.run(
                ["uvicorn", "openai_to_codex_wrapper.server:app", "--host", host, "--port", str(port)],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error starting server: {e}")
        except FileNotFoundError:
            print("Error: uvicorn not found. Please install it (pip install uvicorn).")

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

    env_file_path = ".env"
    env_vars = {}
    if os.path.exists(env_file_path):
        with open(env_file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()

    env_vars["API_ADAPTER_HOST"] = host
    env_vars["API_ADAPTER_PORT"] = str(port)
    env_vars["OPENAI_BASE_URL_INTERNAL"] = openai_internal
    env_vars["OPENAI_BASE_URL"] = openai_external
    env_vars["OPENAI_API_KEY"] = api_key

    with open(env_file_path, "w") as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")

    print(f"Server configuration saved to {env_file_path}. Restart the server to apply changes.")

def help_command():
    """Displays help information."""
    print("Usage: otc <command> [options]")
    print("\nCommands:")
    print("  help       Show this help message.")
    print("  configure  Configure server settings (host, port, API URLs, API key).")
    print("  start      Start the FastAPI server.")

def main():
    parser = argparse.ArgumentParser(description="CLI to manage the OpenAI to Codex Wrapper API server.")
    parser.add_argument("command", nargs="?", default="help", help="Command to execute (help, configure, start)")

    args = parser.parse_args()

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