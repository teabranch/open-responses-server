# openai-responses-server
A server the serves any AI provider with OpenAI ChatCompletions as a Responses API.
I means it manages the stateful component of Responses API, and bridges Ollama, Vllm, LiteLLM and any other AI serving library.
This means you can use OpenAI's new coding assistant "Codex", that needs Responses API endpoints.

It is still missing some features, but I would appreciate your support in stars, issues, suggestions and even pull requests if you are inclined for it.

I verified it works in my main repo, in my [demo AI assistant that can hear, think and speak](https://github.com/OriNachum/autonomous-intelligence/tree/main/baby-tau) with the docker-compose-codex.yaml

Install today via pip: [openai-responses-server](https://pypi.org/project/openai-responses-server)

# Roadmap

- [x] Tool run support (Tested with llama 3.2 3b on Ollama)
- [ ] Validate work from CLI
- [ ] dotenv support
- [ ] State management (long term, not just in-memory)
- [ ] Web search support ([crawl4ai](https://github.com/unclecode/crawl4ai))
- [ ] File upload + search
  - [ ] **[graphiti](https://github.com/getzep/graphiti) (based on neo4j)**
- [ ] Code interpreter 
- [ ] Computer use

# Installation

## UV cli
Install uv if not installed yet.
From: https://docs.astral.sh/uv/getting-started/installation/#standalone-installer

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