# openai-to-codex-wrapper
Wraps any OpenAI API interface so it supports Codex. Adding any missing stateful features. Ollama and Vllm compliant 

OpenAI had released their version of coding assistant as open source.

No big model library supports their Resources api yet, so they can’t work with it.

This library wraps any OpenAI compliant library and gaps resources api via chat completions endpoint.

It is still missing some features, but I would appreciate your support in stars, issues, suggestions and even pull requests if you are inclined for it.

I want to support the stateful features the other libraries don’t want to support and are needed for Codex (and more).

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

