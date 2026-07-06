"""LangGraph agent with remote BGPT MCP tools for scientific evidence retrieval.

Wires BGPT's hosted streamable-HTTP MCP server into a LangGraph ReAct agent via
langchain-mcp-adapters. BGPT returns structured study evidence (methods,
limitations, conflicts of interest, falsifiability) rather than abstracts alone.

Endpoints:
- MCP: https://bgpt.pro/mcp/stream
- REST: POST https://bgpt.pro/api/mcp-search

Free tier works without an API key.

Usage:
    pip install langgraph langchain-mcp-adapters langchain[openai] mcp

    # List BGPT tools wired into the graph
    python bgpt_research_agent.py --list-tools

    # Run a research query (requires OPENAI_API_KEY)
    export OPENAI_API_KEY=...
    python bgpt_research_agent.py --query "CAR-T response rates in lymphoma"
"""

from __future__ import annotations

import argparse
import asyncio
import os

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

BGPT_MCP_URL = "https://bgpt.pro/mcp/stream"


async def build_bgpt_agent():
    """Create a LangGraph ReAct agent with BGPT MCP tools."""
    client = MultiServerMCPClient(
        {
            "bgpt": {
                "transport": "streamable_http",
                "url": BGPT_MCP_URL,
            }
        }
    )
    tools = await client.get_tools()
    agent = create_react_agent("openai:gpt-4.1-mini", tools)
    return agent, tools


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="LangGraph agent with remote BGPT MCP tools"
    )
    parser.add_argument("--list-tools", action="store_true", help="List BGPT tools")
    parser.add_argument("--query", help="Research question for the agent")
    args = parser.parse_args()

    agent, tools = await build_bgpt_agent()

    if args.list_tools or not args.query:
        print(f"BGPT MCP: {BGPT_MCP_URL}")
        print(f"Tools ({len(tools)}):")
        for tool in tools:
            print(f"  - {tool.name}")
        if not args.query:
            return

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY to run an agent query.")

    result = await agent.ainvoke(
        {
            "messages": [
                (
                    "user",
                    f"Use BGPT to find scientific evidence for: {args.query}. "
                    "Report methods, sample sizes, limitations, and falsifiability.",
                )
            ]
        }
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
