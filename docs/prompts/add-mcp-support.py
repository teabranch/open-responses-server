Merge mcp-chatbot-client to server.py so server.py has mcp support.

Note server.py is a middleware that wraps chat.completions endpoints as ResponsesAPI endpoints.
There are some differences, as ResponsesAPI is event based and has state