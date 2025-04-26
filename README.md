# openai-responses-server
A server the serves any AI provider with OpenAI ChatCompletions as a Responses API.
I means it manages the stateful component of Responses API, and bridges Ollama, Vllm, LiteLLM and any other AI serving library.
This means you can use OpenAI's new coding assistant "Codex", that needs Responses API endpoints.

It is still missing some features, but I would appreciate your support in stars, issues, suggestions and even pull requests if you are inclined for it.

I verified it works in my main repo, in my [demo AI assistant that can hear, think and speak](https://github.com/OriNachum/autonomous-intelligence/tree/main/baby-tau) with the docker-compose-codex.yaml

# Roadmap

- [x] Tool run support (Tested with llama 3.2 3b on Ollama)
- [ ] Validate work from CLI
- [ ] dotenv support
- [ ] State management (long term, not just in-memory)
- [ ] Web search support
- [ ] File upload + search
- [ ] Code interpreter
- [ ] Computer use

# OpenAI API Configuration

OPENAI_BASE_URL_INTERNAL=http://localhost:8000
OPENAI_BASE_URL=http://localhost:8080
OPENAI_API_KEY=sk-mockapikey123456789abcdefghijklmnopqrstuvwxyz

# Server Configuration

API_ADAPTER_HOST=0.0.0.0
API_ADAPTER_PORT=8080

# Logging Configuration (optional)

LOG_LEVEL=INFO
LOG_FILE_PATH=./log/api_adapter.log

