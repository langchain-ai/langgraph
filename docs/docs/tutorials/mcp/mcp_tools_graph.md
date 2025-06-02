# Injecting MCP-Sourced Tools at Graph Definition Time
*(Fixes [Issue #4856](https://github.com/langchain-ai/langgraph/issues/4856))*

This short guide shows how to **fetch tools from an MCP server once—while
building the graph—and pass them to a `ToolNode` statically**.  
Callers of the compiled graph do not need to supply the tools at runtime.

---

## 1  Fetch the tools asynchronously

```python
import asyncio
from modelcontext.client import MultiServerMCPClient  # pip install modelcontext

async def get_tools() -> list:
    """Retrieve tool specifications from one or more MCP servers."""
    client = MultiServerMCPClient(
        servers=[
            {
                "url": "https://api.my-mcp.com",
                "api_key": "sk-XXXXXXXXXXXXXXXXXXXXXXXX",
            }
        ]
    )
    return await client.get_tools()

# Run the coroutine once at import time
MCP_TOOLS = asyncio.run(get_tools())

