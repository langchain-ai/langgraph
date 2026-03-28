from langchain_core.messages import AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode
import asyncio
from types import SimpleNamespace


async def main() -> None:
    mcp_server_config = {
        "slow_server": {
            "url": "http://127.0.0.1:8099/mcp",
            "transport": "streamable_http",
            "timeout": 30,
            "sse_read_timeout": 5,  # keep small to force timeout
        }
    }

    mcp_client = MultiServerMCPClient(mcp_server_config)
    mcp_tools = await mcp_client.get_tools()

    tool_node = ToolNode(mcp_tools)

    # ToolNode expects a messages state containing tool_calls
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "slow_tool",
                        "args": {"q": "hello"},
                    }
                ],
            )
        ]
    }

    print("Calling ToolNode.ainvoke...")
    # ToolNode expects a pregel runtime when called directly (outside a graph).
    config = {
        "configurable": {
            "__pregel_runtime": SimpleNamespace(
                store=None,
                context=None,
                stream_writer=lambda *args, **kwargs: None,
            )
        }
    }
    try:
        result = await asyncio.wait_for(
            tool_node.ainvoke(state, config=config),
            timeout=20,
        )
        print("Result:", result)
    except TimeoutError:
        print(
            "Reproduced: ToolNode.ainvoke did not complete within 20s "
            "(likely stuck waiting after MCP timeout)."
        )


if __name__ == "__main__":
    asyncio.run(main())