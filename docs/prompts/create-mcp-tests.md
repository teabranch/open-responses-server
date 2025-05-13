Read README.md, src/openai_responses_server/* and ./test_tools/*
And write a new test_tool that tests mcp_simple_prompt mcp server.
Write 2 tests for openai_responses_server:

for mcp on stdio mode
for mcp on Streamable sse mode
Run the tests and see that they work.
This means I want a respones api call that uses the mcp as stdio
and another call that uses mcp as server

I don't want any mocks - I want to base it on call_respones_proxy.sh, configure the mcp (It can be hard-coded on the servers_config.json), and see that it works.
For a working mcp, you can run test_tools/simple-prompt server, or configure it as studio as instructed in its readme.
