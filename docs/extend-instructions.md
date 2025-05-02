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
pip install serpapi requests beautifulsoup4 fastapi-utils
```

### 2. Create Web Search Module

Create a new file `web_search.py` in the `src/openai_responses_server` directory:

```python
#!/usr/bin/env python3

import os
import json
import logging
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger("web_search")

# API keys from environment
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")
GOOGLE_SEARCH_KEY = os.environ.get("GOOGLE_SEARCH_KEY")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")

class WebSearchProvider:
    """Base class for web search providers"""
    
    async def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search the web using the provider.
        
        Args:
            query: The search query
            num_results: Number of results to return
            
        Returns:
            List of search results with title, link, and snippet
        """
        raise NotImplementedError("Subclasses must implement search()")
    
    async def fetch_content(self, url: str) -> Optional[str]:
        """
        Fetch and extract content from a webpage.
        
        Args:
            url: The URL to fetch
            
        Returns:
            Extracted text content from the webpage
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "header", "footer", "nav"]):
                script.extract()
                
            # Extract text
            text = soup.get_text(separator="\n", strip=True)
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Truncate if too long
            if len(text) > 8000:
                text = text[:8000] + "... [content truncated]"
                
            return text
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {str(e)}")
            return None

class SerpApiProvider(WebSearchProvider):
    """SerpAPI search provider"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://serpapi.com/search"
        
    async def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        try:
            params = {
                "q": query,
                "api_key": self.api_key,
                "engine": "google",
                "num": num_results
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = []
            if "organic_results" in data:
                for item in data["organic_results"][:num_results]:
                    results.append({
                        "title": item.get("title", "No title"),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", "No description")
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error searching with SerpAPI: {str(e)}")
            return []

class GoogleCustomSearchProvider(WebSearchProvider):
    """Google Custom Search Engine provider"""
    
    def __init__(self, api_key: str, cse_id: str):
        self.api_key = api_key
        self.cse_id = cse_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        
    async def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        try:
            params = {
                "key": self.api_key,
                "cx": self.cse_id,
                "q": query,
                "num": min(num_results, 10)  # API limit is 10
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = []
            if "items" in data:
                for item in data["items"][:num_results]:
                    results.append({
                        "title": item.get("title", "No title"),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", "No description")
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error searching with Google CSE: {str(e)}")
            return []

# Factory function to get the appropriate search provider
def get_search_provider() -> WebSearchProvider:
    """Get the configured search provider based on environment variables"""
    if SERPAPI_KEY:
        return SerpApiProvider(SERPAPI_KEY)
    elif GOOGLE_SEARCH_KEY and GOOGLE_CSE_ID:
        return GoogleCustomSearchProvider(GOOGLE_SEARCH_KEY, GOOGLE_CSE_ID)
    else:
        raise ValueError("No search provider configured. Set SERPAPI_KEY or GOOGLE_SEARCH_KEY and GOOGLE_CSE_ID environment variables.")

# Main search function
async def web_search(query: str, num_results: int = 5) -> Dict[str, Any]:
    """
    Search the web using the configured provider.
    
    Args:
        query: The search query
        num_results: Number of results to return
        
    Returns:
        Dictionary with search results and metadata
    """
    provider = get_search_provider()
    results = await provider.search(query, num_results)
    
    return {
        "query": query,
        "results": results,
        "result_count": len(results)
    }

async def fetch_webpage_content(url: str) -> Dict[str, Any]:
    """
    Fetch and extract content from a webpage.
    
    Args:
        url: The URL to fetch
        
    Returns:
        Dictionary with URL and extracted content
    """
    provider = get_search_provider()
    content = await provider.fetch_content(url)
    
    return {
        "url": url,
        "content": content if content else "Failed to retrieve content."
    }
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
pip install langchain chromadb sentence-transformers tiktoken
```

### 2. Create RAG Module

Create a new file `rag.py` in the `src/openai_responses_server` directory:

```python
#!/usr/bin/env python3

import os
import logging
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
import tempfile

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.schema import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger("rag_module")

# Configuration from environment variables
VECTOR_DB_PATH = os.environ.get("VECTOR_DB_PATH", "./data/vector_db")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

class RAGSystem:
    """Retrieval-Augmented Generation system using vector database"""
    
    def __init__(self, persist_directory: str = VECTOR_DB_PATH, embedding_model: str = EMBEDDING_MODEL):
        """
        Initialize the RAG system.
        
        Args:
            persist_directory: Directory to store the vector database
            embedding_model: HuggingFace model to use for embeddings
        """
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        
        # Create directory if it doesn't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Initialize vector store
        try:
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
            logger.info(f"Loaded vector store from {persist_directory}")
        except Exception as e:
            logger.warning(f"Could not load existing vector store: {str(e)}. Creating new one.")
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
    
    async def query(self, query_text: str, num_results: int = 3) -> List[Dict[str, Any]]:
        """
        Query the RAG system for relevant documents.
        
        Args:
            query_text: The query text
            num_results: Maximum number of results to return
            
        Returns:
            List of relevant documents with content and metadata
        """
        try:
            docs = self.vector_store.similarity_search(query_text, k=num_results)
            
            results = []
            for doc in docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            return results
        except Exception as e:
            logger.error(f"Error querying vector store: {str(e)}")
            return []
    
    async def add_text(self, text: str, metadata: Dict[str, Any]) -> bool:
        """
        Add text to the RAG system.
        
        Args:
            text: The text to add
            metadata: Metadata associated with the text
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            chunks = text_splitter.split_text(text)
            
            # Convert to documents
            documents = [Document(page_content=chunk, metadata=metadata) for chunk in chunks]
            
            # Add to vector store
            self.vector_store.add_documents(documents)
            self.vector_store.persist()
            
            return True
        except Exception as e:
            logger.error(f"Error adding text to vector store: {str(e)}")
            return False
    
    async def add_document(self, file_path: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Add a document file to the RAG system.
        
        Args:
            file_path: Path to the document file
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if metadata is None:
                metadata = {}
            
            # Add filename to metadata
            file_name = Path(file_path).name
            metadata["source"] = file_name
            
            # Load document based on file extension
            extension = file_path.lower().split(".")[-1]
            
            if extension == "txt":
                loader = TextLoader(file_path)
            elif extension == "pdf":
                loader = PyPDFLoader(file_path)
            else:
                logger.error(f"Unsupported file format: {extension}")
                return False
            
            documents = loader.load()
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            split_documents = text_splitter.split_documents(documents)
            
            # Add documents to vector store
            self.vector_store.add_documents(split_documents)
            self.vector_store.persist()
            
            return True
        except Exception as e:
            logger.error(f"Error adding document to vector store: {str(e)}")
            return False


# Initialize RAG system
rag_system = RAGSystem()

async def rag_query(query: str, num_results: int = 3) -> Dict[str, Any]:
    """
    Query the RAG system for relevant information.
    
    Args:
        query: The query text
        num_results: Maximum number of results to return
        
    Returns:
        Dictionary with query results
    """
    results = await rag_system.query(query, num_results)
    
    return {
        "query": query,
        "results": results,
        "result_count": len(results)
    }

async def add_to_knowledge_base(text: str, source: str) -> Dict[str, Any]:
    """
    Add text to the knowledge base.
    
    Args:
        text: The text to add
        source: Source identifier for the text
        
    Returns:
        Status of the operation
    """
    success = await rag_system.add_text(text, {"source": source})
    
    return {
        "success": success,
        "source": source
    }
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
openai-responses-server/
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