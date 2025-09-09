# MCP

[Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction) is an open protocol that standardizes how applications provide tools and context to language models. LangGraph agents can use tools defined on MCP servers through the `langchain-mcp-adapters` library.

![MCP](../agents/assets/mcp.png)

Install the `langchain-mcp-adapters` library to use MCP tools in LangGraph:

:::python
```bash
pip install langchain-mcp-adapters
```
:::

:::js
```bash
npm install @langchain/mcp-adapters
```
:::