
"""
Minimal example: inject MCP-sourced tools at graph build time.

Run with:
    python examples/mcp_tools_graph.py

Make sure to insert a valid MCP API key before running.
"""
import asyncio
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langchain_mcp_adapters.client import MultiServerMCPClient


async def get_tools():
    """Fetch tool specifications from the MCP server and return them as a list
    of LangChain Tool objects."""
    client = MultiServerMCPClient(
        servers=[
            {
                "url": "https://api.my-mcp.com",
                "api_key": "sk-XXXXXXXXXXXXXXXXXXXXXXXX",
            }
        ]
    )
    return await client.get_tools()


# Fetch tools once, at import time
MCP_TOOLS = asyncio.run(get_tools())


def build_graph():
    """Create and compile a graph that already contains the MCP tools."""
    graph = StateGraph()
    graph.add_node("tools", ToolNode(tools=MCP_TOOLS))
    # Additional nodes / edges can be added here
    return graph.compile()


if __name__ == "__main__":
    agent = build_graph()
    # Simple test query
    output = agent.invoke({"input": "What is the latest pull-request number?"})
    print("\n=== Agent Output ===")
    print(output)
