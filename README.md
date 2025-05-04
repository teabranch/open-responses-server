# openai-responses-server
A server the serves any AI provider with OpenAI ChatCompletions as OpenAI's Responses API and hosted tools.
I means it manages the stateful component of Responses API, and bridges Ollama, Vllm, LiteLLM and any other AI serving library.
This means you can use OpenAI's new coding assistant "Codex", that needs Responses API endpoints.

It is still missing some features, but I would appreciate your support in stars, issues, suggestions and even pull requests if you are inclined for it.

I verified it works in my main repo, in my [demo AI assistant that can hear, think and speak](https://github.com/OriNachum/autonomous-intelligence/tree/main/baby-tau) with the docker-compose-codex.yaml

Install today via pip: [openai-responses-server](https://pypi.org/project/openai-responses-server)

# Roadmap

- [x] Tool run support (Tested with llama 3.2 3b on Ollama)
- [x] Validate work from CLI
- [ ] dotenv support
- [x] Tests
  - [x] Manual
  - [x] Pipelines
- [ ] Deployments
  - [x] Pypi package
  - [ ] Docker image 
- [ ] State management (long term, not just in-memory)
- [ ] **Web search support ([crawl4ai](https://github.com/unclecode/crawl4ai))**
- [ ] File upload + search
  - [ ] **[graphiti](https://github.com/getzep/graphiti) (based on neo4j)**
- [ ] Code interpreter 
- [ ] Computer use

# OpenAI API Configuration

OPENAI_BASE_URL_INTERNAL=# Your AI Provider host api. localhost for Ollama, Groq and even OpenAI  
OPENAI_BASE_URL=http://localhost:8080 # IP and port of your openai-responses-server (ORS)
OPENAI_API_KEY=sk-mockapikey123456789abcdefghijklmnopqrstuvwxyz # For Ollama, this should be mock. The key is tunneled to the provider

# Server Configuration

API_ADAPTER_HOST=0.0.0.0
API_ADAPTER_PORT=8080

# Logging Configuration (optional)

LOG_LEVEL=INFO
LOG_FILE_PATH=./log/api_adapter.log

# Installation

## UV cli
Install uv if not installed yet.
From: https://docs.astral.sh/uv/getting-started/installation/#standalone-installer

```python
pip install uv
```

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
or 
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | more"
```

Setup environment with:
```
uv venv
``` 

Install dependecies with uv
```
uv pip install .
uv pip install -e ".[dev]"  # for development
```

Run server:
```
uv run src/openai_responses_server/cli.py start
```

# 📚 Citation

## Cited projects

UncleCode. (2024). Crawl4AI: Open-source LLM Friendly Web Crawler & Scraper [Computer software]. 
GitHub. https://github.com/unclecode/crawl4ai

## Cite this project 

If you use openai-responses-server in your research or project, please cite:  

### Code citation format
@software{openai-responses-server,
  author = {TeaBranch},
  title = {openai-responses-server: Open-source server the serves any AI provider with OpenAI ChatCompletions as OpenAI's Responses API and hosted tools.},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/teabranch/openai-responses-server}},
  commit = {Please use the commit hash you're working with}
}

### Text citation format:

TeaBranch. (2025). openai-responses-server: Open-source server the serves any AI provider with OpenAI ChatCompletions as OpenAI's Responses API and hosted tools. [Computer software]. 
GitHub. https://github.com/teabranch/openai-responses-server
