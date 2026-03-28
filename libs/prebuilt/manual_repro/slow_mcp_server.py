import anyio
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("slow-server", host="127.0.0.1", port=8099)


@mcp.tool()
async def slow_tool(q: str) -> str:
	# Intentionally exceed client sse_read_timeout.
	await anyio.sleep(60)
	return f"done: {q}"


if __name__ == "__main__":
	# Streamable HTTP MCP endpoint.
	mcp.run(transport="streamable-http")

