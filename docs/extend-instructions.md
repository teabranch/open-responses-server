# Extending OpenAI Responses Server with Web Search and RAG

This document provides instructions for extending the OpenAI Responses Server with web search capabilities and Retrieval-Augmented Generation (RAG) functionality. These features allow the LLM to access real-time information from the internet and retrieve relevant data from knowledge bases to enhance response quality.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Architecture](#architecture)
4. [Implementing Web Search](#implementing-web-search)
5. [Implementing RAG](#implementing-rag)
6. [Configuration](#configuration)
7. [Security Considerations](#security-considerations)
8. [Testing](#testing)
9. [File Search Engine](#file-search-engine)
10. [Web Search Engine](#web-search-engine)

## Overview

Web search and RAG capabilities extend the OpenAI Responses Server by:

- Allowing the LLM to search the internet for up-to-date information
- Retrieving content from knowledge bases to provide context-specific answers
- Enabling access to additional data not included in the model's training data

## Prerequisites

Before implementing these extensions, ensure you have:

- A working installation of the OpenAI Responses Server
- Python 3.8+ and pip
- Access to a search API (e.g., Google Search API, Bing Search API, SerpAPI)
- Vector database for RAG (e.g., Chroma, Pinecone, Weaviate, Redis)
- Optional: Document processors for data ingestion (if building your own knowledge base)

## Architecture

The extension architecture follows the tool-based approach compatible with OpenAI's function/tool calling system:

```
┌───────────────────┐    ┌─────────────────────┐    ┌───────────────────┐
│                   │    │                     │    │                   │
│  Client Request   │───►│  Responses Server   │───►│  OpenAI API       │
│                   │    │                     │    │                   │
└───────────────────┘    └─────────┬───────────┘    └────────┬──────────┘
                                   │                         │
                         ┌─────────▼───────────┐    ┌────────▼──────────┐
                         │                     │    │                   │
                         │  Tool Handlers      │◄───┤  Tool Calls       │
                         │                     │    │                   │
                         └─────────┬───────────┘    └───────────────────┘
                                   │
               ┌───────────────────┴───────────────────┐
               │                                       │
    ┌──────────▼───────────┐            ┌─────────────▼──────────┐
    │                      │            │                        │
    │  Web Search Module   │            │  RAG Module            │
    │                      │            │                        │
    └──────────┬───────────┘            └────────────┬───────────┘
               │                                     │
    ┌──────────▼───────────┐            ┌────────────▼───────────┐
    │                      │            │                        │
    │  Search API Provider │            │  Vector Database       │
    │                      │            │                        │
    └──────────────────────┘            └────────────────────────┘
```

## Implementing Web Search

### 1. Install Required Packages

```bash
pip install crawl4ai
```

### 2. Create Web Search Module

Create a new file `web_search.py` in the `src/openai_responses_server` directory:

```python
#!/usr/bin/env python3

import logging
from crawl4ai import Crawl4AI

# Configure logging
logger = logging.getLogger("web_search")

class WebSearch:
    """Web search using Crawl4AI."""

    def __init__(self):
        self.crawler = Crawl4AI()

    async def search(self, query: str, num_results: int = 5):
        """
        Perform a web search.

        Args:
            query: The search query.
            num_results: Number of results to return.

        Returns:
            List of search results with title, link, and snippet.
        """
        try:
            results = self.crawler.search(query, limit=num_results)
            return [
                {
                    "title": result.get("title", "No title"),
                    "link": result.get("url", ""),
                    "snippet": result.get("snippet", "No description"),
                }
                for result in results
            ]
        except Exception as e:
            logger.error(f"Error during web search: {str(e)}")
            return []

    async def fetch_content(self, url: str):
        """
        Fetch and extract content from a webpage.

        Args:
            url: The URL to fetch.

        Returns:
            Extracted text content from the webpage.
        """
        try:
            content = self.crawler.fetch(url)
            return content[:8000] if len(content) > 8000 else content
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {str(e)}")
            return None
```

### 3. Define Tool Schema

Create a new file `tools.py` in the `src/openai_responses_server` directory:

```python
#!/usr/bin/env python3

from typing import Dict, List, Any

# Web Search Tool Schema
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for information on a specific topic or query",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find information about"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of search results to return (default: 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
}

# Webpage Content Fetching Tool Schema
FETCH_WEBPAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_webpage_content",
        "description": "Fetch and extract the main content from a specific webpage URL",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL of the webpage to fetch content from"
                }
            },
            "required": ["url"]
        }
    }
}

# RAG Query Tool Schema (to be implemented in the RAG section)
RAG_QUERY_TOOL = {
    "type": "function",
    "function": {
        "name": "rag_query",
        "description": "Query the knowledge base for relevant information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search for in the knowledge base"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 3)",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    }
}

# Get all available tools
def get_all_tools() -> List[Dict[str, Any]]:
    """Return all available tools for the API"""
    return [
        WEB_SEARCH_TOOL,
        FETCH_WEBPAGE_TOOL,
        RAG_QUERY_TOOL
    ]
```

### 4. Update Server.py

Modify the `server.py` file to include the new tool handlers:

```python
# Add the necessary imports at the top of server.py
from .web_search import web_search, fetch_webpage_content
from .tools import get_all_tools, WEB_SEARCH_TOOL, FETCH_WEBPAGE_TOOL
# Import RAG tools when implemented
```

Then, add a new route for tool execution:

```python
@app.post("/tools/{tool_name}")
async def execute_tool(tool_name: str, request: Request):
    """
    Execute a tool based on the tool name and provided arguments.
    This handles the tool calls from the LLM.
    """
    try:
        logger = logging.getLogger("api_adapter_tools")
        logger.info(f"Executing tool: {tool_name}")
        
        request_data = await request.json()
        logger.info(f"Tool arguments: {request_data}")
        
        result = None
        
        # Web Search Tool
        if tool_name == "web_search":
            query = request_data.get("query", "")
            num_results = request_data.get("num_results", 5)
            result = await web_search(query, num_results)
        
        # Fetch Webpage Tool
        elif tool_name == "fetch_webpage_content":
            url = request_data.get("url", "")
            result = await fetch_webpage_content(url)
        
        # RAG Query Tool (to be implemented)
        elif tool_name == "rag_query":
            # RAG implementation will go here
            result = {"error": "RAG not yet implemented"}
        
        else:
            return JSONResponse(
                status_code=404,
                content={"error": f"Unknown tool: {tool_name}"}
            )
        
        logger.info(f"Tool execution result: {result}")
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error executing tool: {str(e)}"}
        )
```

Also, modify the `/responses` endpoint to include the available tools:

```python
# In create_response function, before converting the request
# Add tools if not already provided
if "tools" not in request_data or not request_data["tools"]:
    # Add all available tools
    request_data["tools"] = get_all_tools()
```

## Implementing RAG

### 1. Install Required Packages

```bash
pip install graphiti-core
```

### 2. Create RAG Module

Update the RAG module to use OpenAI's file endpoints for file management and Graphiti for indexing and retrieval. Example:

```python
import openai
from graphiti import Graphiti

class RAG:
    def __init__(self):
        self.graph = Graphiti()

    async def add_file(self, file_path: str):
        """Upload a file to OpenAI and index it with Graphiti."""
        with open(file_path, "rb") as f:
            response = openai.File.create(file=f, purpose="answers")
            file_id = response["id"]
            self.graph.index_file(file_path, file_id)

    async def query(self, query: str, num_results: int = 3):
        """Query indexed files for relevant information."""
        return self.graph.query(query, limit=num_results)
```

### 3. Update Server.py for RAG

Add the RAG functionality to the `server.py` file:

```python
# Add import at the top
from .rag import rag_query, add_to_knowledge_base
from .tools import RAG_QUERY_TOOL

# Then update the execute_tool function to include RAG
# In the execute_tool function, update the RAG section:

# RAG Query Tool
elif tool_name == "rag_query":
    query = request_data.get("query", "")
    num_results = request_data.get("num_results", 3)
    result = await rag_query(query, num_results)
```

### 4. Create Knowledge Base Management Endpoint

Add a new route to manage the knowledge base:

```python
@app.post("/knowledge_base/add")
async def add_knowledge(request: Request):
    """
    Add content to the knowledge base.
    """
    try:
        data = await request.json()
        text = data.get("text", "")
        source = data.get("source", "api_upload")
        
        if not text:
            return JSONResponse(
                status_code=400,
                content={"error": "No text provided"}
            )
        
        result = await add_to_knowledge_base(text, source)
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error adding to knowledge base: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error adding to knowledge base: {str(e)}"}
        )
```

## Configuration

### Environment Variables

Add the following environment variables to your `.env` file:

```
# Web Search API Keys
SERPAPI_KEY=your_serpapi_key_here
GOOGLE_SEARCH_KEY=your_google_api_key_here
GOOGLE_CSE_ID=your_google_cse_id_here

# RAG Configuration
VECTOR_DB_PATH=./data/vector_db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Directory Structure

Ensure your project has the following directory structure:

```
open-responses-server/
├── src/
│   └── openai_responses_server/
│       ├── __init__.py
│       ├── cli.py
│       ├── server.py
│       ├── tools.py
│       ├── web_search.py
│       ├── rag.py
│       └── version.py
├── data/
│   └── vector_db/
├── .env
├── setup.py
└── ...
```

## Security Considerations

When implementing web search and RAG capabilities, keep the following security considerations in mind:

1. **API Key Management**: Store API keys securely and never commit them to version control.

2. **Content Filtering**: Implement content filtering to prevent retrieval of harmful or inappropriate content.

3. **Rate Limiting**: Apply rate limiting to prevent abuse of search APIs and excessive costs.

4. **Data Privacy**: Be cautious about what data is stored in the vector database. Consider privacy implications.

5. **Input Sanitization**: Sanitize inputs to prevent injection attacks and other security vulnerabilities.

6. **Access Control**: Implement proper authentication and authorization for knowledge base management endpoints.

## Testing

Test your implementation with the following examples:

### 1. Test Web Search

```bash
curl -X POST http://localhost:8080/tools/web_search \
  -H "Content-Type: application/json" \
  -d '{"query": "latest advancements in AI 2025"}'
```

### 2. Test RAG Query

```bash
# First add content to knowledge base
curl -X POST http://localhost:8080/knowledge_base/add \
  -H "Content-Type: application/json" \
  -d '{"text": "OpenAI Responses Server is a powerful API adapter that translates between different APIs.", "source": "documentation"}'

# Then query the knowledge base
curl -X POST http://localhost:8080/tools/rag_query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is OpenAI Responses Server?"}'
```

### 3. Test Combined Capabilities

Make a request to the `/responses` endpoint with a query that might benefit from both web search and RAG capabilities.

## File Search Engine

For file search capabilities, consider integrating [Graphiti](https://github.com/getzep/graphiti). Graphiti is a framework for building and querying temporally-aware knowledge graphs, specifically tailored for AI agents operating in dynamic environments. It supports hybrid semantic, keyword, and graph-based search methods, making it suitable for advanced file search and retrieval tasks.

### Installation

To install Graphiti, use the following command:

```bash
pip install graphiti-core[anthropic,groq,google-genai]
```

### Key Features

- Build real-time knowledge graphs for AI agents
- Incremental data updates and efficient retrieval
- Query complex, evolving data with semantic and graph-based search

Refer to the [Graphiti documentation](https://github.com/getzep/graphiti) for detailed setup and usage instructions.

## Web Search Engine

For web search capabilities, consider integrating [Crawl4AI](https://github.com/unclecode/crawl4ai). Crawl4AI is an open-source, LLM-friendly web crawler and scraper designed for blazing-fast, AI-ready web crawling tailored for LLMs and AI agents.

### Installation

To install Crawl4AI, use the following command:

```bash
pip install crawl4ai
```

### Key Features

- Deep crawling with BFS, DFS, and BestFirst strategies
- Browser-based and lightweight HTTP-only crawlers
- AI-powered coding assistant for web data extraction
- Proxy rotation and memory-adaptive dispatching

Refer to the [Crawl4AI documentation](https://github.com/unclecode/crawl4ai) for detailed setup and usage instructions.