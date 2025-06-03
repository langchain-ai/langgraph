---
search:
  boost: 2
tags:
  - agent
hide:
  - tags
---

# MCP Integration

[Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction) is an open protocol that standardizes how applications provide tools and context to language models. LangGraph agents can use tools defined on MCP servers through the `langchain-mcp-adapters` library.

![MCP](./assets/mcp.png)

Install the `langchain-mcp-adapters` library to use MCP tools in LangGraph:

```bash
pip install langchain-mcp-adapters
```

## Use MCP tools

The `langchain-mcp-adapters` package enables agents to use tools defined across one or more MCP servers.

```python title="Agent using tools defined on MCP servers"
# highlight-next-line
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

# highlight-next-line
client = MultiServerMCPClient(
    {
        "math": {
            "command": "python",
            # Replace with absolute path to your math_server.py file
            "args": ["/path/to/math_server.py"],
            "transport": "stdio",
        },
        "weather": {
            # Ensure you start your weather server on port 8000
            "url": "http://localhost:8000/mcp",
            "transport": "streamable_http",
        }
    }
)
# highlight-next-line
tools = await client.get_tools()
agent = create_react_agent(
    "anthropic:claude-3-7-sonnet-latest",
    # highlight-next-line
    tools
)
math_response = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
)
weather_response = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "what is the weather in nyc?"}]}
)
```

## Custom MCP servers

To create your own MCP servers, you can use the `mcp` library. This library provides a simple way to define tools and run them as servers.

Install the MCP library:

```bash
pip install mcp
```
Use the following reference implementations to test your agent with MCP tool servers.

```python title="Example Math Server (stdio transport)"
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

```python title="Example Weather Server (Streamable HTTP transport)"
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location."""
    return "It's always sunny in New York"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

## Additional resources

- [MCP documentation](https://modelcontextprotocol.io/introduction)
- [MCP Transport documentation](https://modelcontextprotocol.io/docs/concepts/transports)

## Integrating MCP Tools into LangGraph Graph Definitions

When building LangGraph applications that will be deployed to LangGraph Server, you may want to pre-configure your graph with MCP tools rather than fetching them at runtime. Here's how to do this effectively:

### Using Async Bootstrap for Graph Setup

You can use an async bootstrap pattern to load MCP tools during graph initialization. Here's an example:

```python title="Graph with pre-configured MCP tools"
from typing import Annotated, Sequence, TypedDict
from langgraph.graph import Graph, StateGraph
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool

# Define your graph state
class GraphState(TypedDict):
    messages: Annotated[Sequence[dict], "The messages in the conversation"]
    next: Annotated[str, "The next node to call"]

# Async bootstrap function to load tools
async def load_mcp_tools() -> list[BaseTool]:
    client = MultiServerMCPClient({
        "math": {
            "command": "python",
            "args": ["/path/to/math_server.py"],
            "transport": "stdio",
        }
    })
    return await client.get_tools()

# Create graph with pre-loaded tools
async def create_graph() -> Graph:
    # Load tools during graph creation
    tools = await load_mcp_tools()
    
    # Define your nodes using the pre-loaded tools
    def tools_node(state: GraphState) -> GraphState:
        # Your tools node implementation using the pre-loaded tools
        return state
    
    # Build the graph
    workflow = StateGraph(GraphState)
    workflow.add_node("tools", tools_node)
    # Add other nodes and edges...
    
    return workflow.compile()

# Usage in your application
async def main():
    graph = await create_graph()
    # Use the graph...
```

### Alternative: Using Tool Factories

Another pattern is to use a tool factory that can be called during graph initialization:

```python title="Using a tool factory pattern"
from functools import partial
from typing import Callable

class MCPToolFactory:
    def __init__(self, client_config: dict):
        self.client_config = client_config
        self._tools = None
    
    async def get_tools(self) -> list[BaseTool]:
        if self._tools is None:
            client = MultiServerMCPClient(self.client_config)
            self._tools = await client.get_tools()
        return self._tools

# Create a factory instance
tool_factory = MCPToolFactory({
    "math": {
        "command": "python",
        "args": ["/path/to/math_server.py"],
        "transport": "stdio",
    }
})

# Use in graph definition
async def create_graph() -> Graph:
    tools = await tool_factory.get_tools()
    # Build your graph with the tools...
```

### Best Practices

1. **Error Handling**: Always include proper error handling when loading MCP tools:
   ```python
   async def load_mcp_tools() -> list[BaseTool]:
       try:
           client = MultiServerMCPClient(config)
           return await client.get_tools()
       except Exception as e:
           # Handle connection or tool loading errors
           logger.error(f"Failed to load MCP tools: {e}")
           return []
   ```

2. **Caching**: Consider caching the tools after first load to avoid repeated async calls:
   ```python
   class CachedMCPTools:
       def __init__(self):
           self._tools = None
       
       async def get_tools(self) -> list[BaseTool]:
           if self._tools is None:
               self._tools = await load_mcp_tools()
           return self._tools
   ```

3. **Configuration Management**: Store MCP server configurations in environment variables or configuration files:
   ```python
   import os
   
   MCP_CONFIG = {
       "math": {
           "command": os.getenv("MCP_MATH_COMMAND", "python"),
           "args": [os.getenv("MCP_MATH_SERVER_PATH", "/path/to/math_server.py")],
           "transport": "stdio",
       }
   }
   ```

These patterns allow you to create self-contained LangGraph applications that have their MCP tools pre-configured, making them easier to deploy and maintain.