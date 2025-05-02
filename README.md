# openai-responses-server
A server the serves any AI provider with OpenAI ChatCompletions as OpenAI's Responses API and hosted tools.
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