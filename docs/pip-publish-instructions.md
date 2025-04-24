**Instructions for Converting `openai-to-codex-wrapper` to a Pip-Installable CLI ("otc")**

This document outlines the steps to transform the Python server located in `src/server.py` into a command-line interface (CLI) tool named `otc` that can be installed via pip and optionally run within a Docker container.

**1. Project Structure:**

Ensure your repository has the following structure (adjust as needed):

```
/
├── src/
│   └── server.py       # Your main server logic
│   └── __init__.py     # (Likely exists)
│   └── ...             # Other server modules
├── docs/
│   └── instructions.md # Detailed instructions for code modifications
├── tests/            # Your test suite
│   └── ...
├── README.md
├── LICENSE
├── setup.py            # For packaging and distribution
├── pyproject.toml      # Modern alternative to setup.py (recommended)
└── Dockerfile          # For containerizing the application
```

**2. CLI Entry Point (`src/server.py`):**

Refer to the detailed instructions in `docs/pip-publish-instructions.md` for how to modify `src/server.py` to include an `argparse` (or `click`, `typer`) based CLI interface. The main function that will be executed when the `otc` command is run should be named `main()`.

**Example Snippet (Conceptual - See `docs/instructions.md` for specifics):**

```python
# src/server.py
import argparse

def run_wrapper(api_key, model, prompt, ...):
    """Executes the OpenAI Codex wrapper logic."""
    print(f"Running with model: {model}, prompt: '{prompt[:50]}...'")
    # Your server/wrapper logic here
    pass

def main():
    parser = argparse.ArgumentParser(description="OpenAI Codex Wrapper CLI")
    parser.add_argument("--api-key", type=str, required=True, help="Your OpenAI API key.")
    parser.add_argument("--model", type=str, default="code-davinci-002", help="The Codex model to use.")
    parser.add_argument("prompt", type=str, help="The prompt to send to Codex.")
    # Add other relevant arguments based on your server's functionality

    args = parser.parse_args()
    run_wrapper(args.api_key, args.model, args.prompt, ...)

if __name__ == "__main__":
    main()
```

**3. Configuring `setup.py` (or `pyproject.toml`) for Pip Install:**

**Using `setup.py`:**

Create or modify the `setup.py` file in the root of your repository:

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name='openai-to-codex-cli',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # List any dependencies your server needs (e.g., openai)
    ],
    entry_points={
        'console_scripts': [
            'otc = server:main',
        ],
    },
    author='Your Name',
    author_email='your.email@example.com',
    description='A CLI wrapper for OpenAI Codex',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/teabranch/openai-to-codex-wrapper',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', # Choose your license
        'Operating System :: OS Independent',
    ],
)
```

* **`name`**: Set to `openai-to-codex-cli` (a good PyPI package name).
* **`packages=find_packages(where='src')`**: Tells setuptools to find packages within the `src` directory.
* **`package_dir={'': 'src'}`**: Maps the root package to the `src` directory.
* **`entry_points`**: Crucially, `'otc = server:main'` maps the `otc` command to the `main` function in your `src/server.py` file.

**Using `pyproject.toml` (Modern Approach):**

Create or modify the `pyproject.toml` file in the root of your repository:

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "openai-to-codex-cli"
version = "0.1.0"
authors = [
  { name="Your Name", email="your.email@example.com" },
]
description = "A CLI wrapper for OpenAI Codex"
readme = "README.md"
requires-python = ">=3.7"
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]
dependencies = [
    # List your dependencies here (e.g., "openai>=0.27")
]

[project.scripts]
otc = "server:main"
```

* The `[project.scripts]` section defines the `otc` command mapping.

**4. Building and Publishing to PyPI:**

Follow the standard PyPI publishing process:

1.  **Install build tools:**
    ```bash
    pip install --upgrade setuptools wheel twine
    ```
2.  **Navigate to the root directory.**
3.  **Build the package:**
    * Using `setup.py`: `python setup.py sdist bdist_wheel`
    * Using `pyproject.toml`: `python -m build`
4.  **Upload to PyPI:**
    ```bash
    twine upload dist/*
    ```
    You will need a PyPI account and should use an API token for secure uploading.

**5. Docker Support:**

Create a `Dockerfile` in the root of your repository:

```dockerfile
FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src /app

EXPOSE 8000 # If your server listens on a specific port

ENTRYPOINT ["otc"] # Run the CLI directly when the container starts

# You might want to add CMD with default arguments if needed
# CMD ["--help"]
```

**Explanation:**

* We copy the `src` directory containing your server code.
* `ENTRYPOINT ["otc"]` makes the `otc` command the default when the container runs. Ensure that all necessary arguments are either provided when running the Docker container or handled with default values in your CLI. You might adjust this to pass default arguments if needed.

**Building and Running the Docker Image:**

```bash
docker build -t openai-to-codex-cli .
docker run openai-to-codex-cli --api-key YOUR_API_KEY --prompt "Translate this to French: Hello world"
```

**6. GitHub Publish Pipeline (CI/CD):**

Set up a GitHub Actions workflow (e.g., `.github/workflows/publish.yml`) as described in the previous response to automate building and publishing to PyPI on tagged releases. Remember to create a `PYPI_TOKEN` secret in your repository settings.

**Key Points:**

* The core logic for the CLI will reside within the `main()` function of your `src/server.py` file (as detailed in `docs/instructions.md`).
* The `setup.py` or `pyproject.toml` file defines how the `otc` command is created and linked to your `main()` function.
* The `Dockerfile` provides a way to containerize your CLI application.

By following these updated instructions and the detailed steps in your `docs/instructions.md` file, you can successfully convert your `openai-to-codex-wrapper` into a pip-installable CLI tool named `otc`.