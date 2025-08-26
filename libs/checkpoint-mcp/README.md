# checkpoint-mcp

This package provides an async checkpoint store for LangGraph using the MCP (Model Context Protocol) library.

## Summary
This package offers MCP as store thereby providing the ability to integrate with any MCP server that implements a specific set of tools (see [MCP Server Tools](#mcp-server-tools) section below). Any compliant MCP server can be used as a backend (e.g. PostgreSQL MCP server), enabling flexible and interchangeable checkpoint storage and agent memory.

## Features
- Implements async batch operations for storing and retrieving values using MCP:
    - `Put`: Store values with namespace and key.
    - `Get`: Retrieve values by namespace and key.
    - `Search`: Search for values by namespace prefix and optional query, returning each value with a relevance score.
    - `ListNamespaces`: List all available namespaces.
- Supports efficient batch processing for multiple operations in a single call.
- Can be used as a backend for memory in LangGraph using LangMem.

## MCP Server Tools

The MCP server must implement these tools to support the store:
- `store_put` - Store an item
- `store_get` - Retrieve an item  
- `store_search` - Search for items
- `store_list_namespaces` - List namespaces

## Notes

Current implementation/initial iteration only supports:
- Async store operations
- Streamable HTTP MCP servers  
- Connection without authentication to MCP server


## Usage Example

```python
import asyncio
from langgraph.store.mcp import AsyncMCPStore

async def main():
    # Use as an async context manager for proper lifecycle management
    async with AsyncMCPStore.from_mcp_config(host="your-mcp-server", port=8000) as store:
        # Store a value using high-level API
        await store.aput(namespace=("namespace",), key='key', value={'data': 'value'})
        
        # Search for values using high-level API
        results = await store.asearch(namespace_prefix=("namespace",), query='value', limit=10)
        
        # Get a specific value
        item = await store.aget(namespace=("namespace",), key='key')
        
        # List available namespaces
        namespaces = await store.alist_namespaces()

# Run the async function
asyncio.run(main())
```

## Usage with Embeddings

The AsyncMCPStore supports embedding-based indexing for semantic search:

```python
import asyncio
from langgraph.store.mcp import AsyncMCPStore

async def embedding_example():
    # Configure embeddings (replace with your embedding function)
    def mock_embeddings(texts):
        # Your embedding logic here
        return [[0.1, 0.2, 0.3] for _ in texts]
    
    # Create store with embedding configuration
    index_config = {
        "dims": 3,
        "field": ["content", "title"], 
        "embed": mock_embeddings
    }
    
    async with AsyncMCPStore.from_mcp_config(
        host="your-mcp-server", 
        port=8000, 
        index_config=index_config
    ) as store:
        # Store documents with automatic embedding
        await store.aput(
            namespace=("docs",),
            key="doc1",
            value={
                "title": "Python Guide", 
                "content": "Learn Python programming"
            },
            index=["title", "content"]  # Fields to embed
        )
        
        # Search with semantic similarity
        results = await store.asearch(
            namespace_prefix=("docs",),
            query="programming tutorial",
            limit=5
        )

# Run the embedding example
asyncio.run(embedding_example())
```

## Integration with LangMem

You can use this async checkpoint with LangMem for long term memory:

```python
import asyncio
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.store.mcp import AsyncMCPStore

async def setup_langmem():
    # Create the async store using from_mcp_config for easier configuration
    async with AsyncMCPStore.from_mcp_config(
        host="your-mcp-server",
        port=8000,
        username="your-username",
        password="your-password"
    ) as store:
        manage_tool = create_manage_memory_tool(store=store)
        search_tool = create_search_memory_tool(store=store)
        
        # Your application logic here
        return manage_tool, search_tool

# Run the async setup
asyncio.run(setup_langmem())
```

## Running the Example MCP Server

A simple MCP server is provided for testing purposes. To run:

```bash
python mcp_server.py
```

## Tests

You can run the provided tests to validate the AsyncMCPStore implementation:

```bash
python -m pytest tests/ -v
```