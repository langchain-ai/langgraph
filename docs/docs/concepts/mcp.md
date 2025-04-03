# Model Context Protocol (MCP)

MCP is an open protocol that standardizes how applications provide context to LLMs.

[LangGraph Server](./langgraph_server.md) implements the MCP specification using the 
[Streamable HTTP](https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/transports/#streamable-http) transport.

This allows you to expose your LangGraph **agents** as **MCP tools** and use them in any MCP-compliant client that
supports the Streamable HTTP transport.

The MCP endpoint is available at `/mcp` on the LangGraph server.

## Requirements

To use MCP, you need to install the following dependencies:

* langgraph-api>=0.0.45
* langgraph-sdk>=0.1.61 (python).

!!! warning "JavaScript/TypeScript support"

    The MCP protocol is currently only supported in Python.

## Authentication

The MCP endpoint is protected by the same authentication as the rest of the LangGraph API.

Please read the [authentication documentation](./auth.md) for more information on how to set up authentication for your LangGraph server.



