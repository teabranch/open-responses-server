**1. Separate CLI Logic:**

We'll create a separate CLI entry point. This keeps the server code clean and focused on its API duties. Let's create a new file, for example, `src/cli.py`:

```python
# src/cli.py
import argparse
import subprocess
import os
import json

DEFAULT_HOST = os.environ.get("API_ADAPTER_HOST", "0.0.0.0")
DEFAULT_PORT = os.environ.get("API_ADAPTER_PORT", "8080")

def start_server(host=DEFAULT_HOST, port=DEFAULT_PORT):
    """Starts the FastAPI server."""
    try:
        subprocess.run(
            ["uvicorn", "server:app", "--host", host, "--port", str(port), "--reload"],
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

    with open(env_file_path, "w") as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")

    print(f"Server configuration saved to {env_file_path}. Restart the server to apply changes.")

def help_command():
    """Displays help information."""
    print("Usage: otc <command> [options]")
    print("\nCommands:")
    print("  help       Show this help message.")
    print("  configure  Configure server settings (host, port).")
    print("  start      Start the FastAPI server.")

def main():
    parser = argparse.ArgumentParser(description="CLI to manage the OpenAI to Codex Wrapper API server.")
    parser.add_argument("command", help="Subcommand to execute (help, configure, start)")

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
```

**2. Update `setup.py` (or `pyproject.toml`):**

Modify the `entry_points` in your packaging configuration to point to this new CLI script.

**Using `setup.py`:**

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name='openai-to-codex-cli',
    version='0.1.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        "fastapi",
        "uvicorn",
        "httpx",
        "pydantic",
        "python-dotenv",
        # Add any other dependencies your server uses
    ],
    entry_points={
        'console_scripts': [
            'otc = cli:main',
        ],
    },
    author='Ori Nachum',
    author_email='ori.nachum@gmail.com',
    description='CLI to manage the Responses API Wrapper server',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/teabranch/openai-to-codex-wrapper',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Environment :: Web Environment',
    ],
)
```

**Using `pyproject.toml`:**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "openai-to-codex-cli"
version = "0.1.1"
authors = [
  { name="Ori Nachum", email="ori.nachum@example.com" },
]
description = "CLI to manage the OpenAI to Codex Wrapper API server"
readme = "README.md"
requires-python = ">=3.7"
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Environment :: Web Environment",
]
dependencies = [
    "fastapi",
    "uvicorn",
    "httpx",
    "pydantic",
    "python-dotenv",
    # Add any other dependencies your server uses
]

[project.scripts]
otc = "cli:main"
```

**3. Update `Dockerfile` (Optional but Recommended):**

You might want to include `uvicorn` in your Docker image's dependencies if you intend to run the server within the container using the CLI.

```dockerfile
FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src /app

EXPOSE 8080

# Default command to run the CLI help
CMD ["otc", "help"]
```

**Explanation of Changes:**

* **`src/cli.py`**: This new file contains the logic for your CLI. It uses `argparse` to handle different commands (`start`, `configure`, `help`).
    * `start_server()`: Uses `subprocess.run` to execute `uvicorn` and start your FastAPI server. It assumes `server:app` is the correct way to run your FastAPI application.
    * `configure_server()`: Provides a basic way to modify environment variables in a `.env` file, which your server already uses for configuration. You can extend this for more sophisticated configuration.
    * `help_command()`: Displays usage instructions.
    * `main()`: Parses the command-line arguments and calls the appropriate function.
* **`setup.py` / `pyproject.toml`**: The `entry_points` now points the `otc` command to the `main` function in `src/cli.py`. You also need to ensure `uvicorn` is listed in your `install_requires` or `dependencies`.
* **`Dockerfile`**: The default `CMD` is now set to run `otc help` when the container starts, which can be helpful for users.

**How to Use the CLI:**

1.  **Install the CLI via pip:**
    ```bash
    pip install openai-to-codex-cli
    ```
2.  **Run the CLI commands:**
    ```bash
    otc help        # Display help information
    otc configure   # Configure server settings
    otc start       # Start the FastAPI server
    ```

**Important Considerations:**

* **Error Handling:** The `start_server` function includes basic error handling for `subprocess` execution and `uvicorn` not being found. You might want to add more robust error handling.
* **Configuration:** The `configure_server` function provides a very basic way to modify `.env` variables. For more complex configurations, you might consider using configuration files (e.g., YAML, JSON) and providing CLI options to manage them.
* **Server Management:** The `start` command simply runs the server. You could add commands for stopping or restarting the server if needed, but this would likely involve more complex process management.
* **Dependencies:** Ensure all necessary dependencies for both the CLI and the server are listed in your `install_requires` or `dependencies`.

This approach provides a clean separation between your API server logic and the CLI for managing its lifecycle, which is often a good practice for maintainability and scalability.