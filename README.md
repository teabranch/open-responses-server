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
- [ ] Web search support
- [ ] File upload + search 
- [ ] Code interpreter 
- [ ] Computer use

