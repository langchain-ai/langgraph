# MCP Integration

[Model Context Protocol](https://modelcontextprotocol.io/introduction) (MCP) is an open protocol that standardizes how applications provide tools and context to LLMs. LangGraph's `create_react_agent` supports tools defined in MCP servers via using `langchain-mcp-adapters` library.

![MCP](./assets/mcp.png)

```bash
pip install langchain-mcp-adapters
```

## Use MCP tools

`langchain-mcp-adapters` allows you to connect your agent to tools defined across multiple MCP servers:

```python
# highlight-next-line
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

# highlight-next-line
async with MultiServerMCPClient(
    {
        "math": {
            "command": "python",
            # Make sure to update to the full absolute path to your math_server.py file
            "args": ["/path/to/math_server.py"],
            "transport": "stdio",
        },
        "weather": {
            # make sure you start your weather server on port 8000
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
    math_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
    weather_response = await agent.ainvoke({"messages": "what is the weather in nyc?"})
```

## Create MCP tool servers

```bash
pip install mcp
```

Here are reference servers you can use to run the above example:

* a math server that the client communicates with via stdio

    ```python
    # math_server.py
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

* a weather server that the client communicates with via HTTP + SSE

    ```python
    # weather_server.py
    from typing import List
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("Weather")

    @mcp.tool()
    async def get_weather(location: str) -> str:
        """Get weather for location."""
        return "It's always sunny in New York"

    if __name__ == "__main__":
        mcp.run(transport="sse")
    ```