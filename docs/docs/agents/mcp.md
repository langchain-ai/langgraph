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
async with MultiServerMCPClient(
    {
        "math": {
            "command": "python",
            # Replace with absolute path to your math_server.py file
            "args": ["/path/to/math_server.py"],
            "transport": "stdio",
        },
        "weather": {
            # Ensure your start your weather server on port 8000
            "url": "http://localhost:8000/sse",
            "transport": "sse",
        }
    }
) as client:
    agent = create_react_agent(
        "anthropic:claude-3-7-sonnet-latest",
        # highlight-next-line
        client.get_tools()
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

```python title="Example Weather Server (SSE transport)"
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location."""
    return "It's always sunny in New York"

if __name__ == "__main__":
    mcp.run(transport="sse")
```

## Additional resources

- [MCP documentation](https://modelcontextprotocol.io/introduction)
- [MCP Transport documentation](https://modelcontextprotocol.io/docs/concepts/transports)