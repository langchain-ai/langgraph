# MCP

[Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction) is an open protocol that standardizes how applications provide tools and context to language models. LangGraph agents can use tools defined on MCP servers through the `langchain-mcp-adapters` library.

![MCP](../agents/assets/mcp.png)

Install the `langchain-mcp-adapters` library to use MCP tools in LangGraph:

```bash
pip install langchain-mcp-adapters
```

## Authenticate to an MCP server

You can set up [custom authentication middleware](../how-tos/auth/custom_auth.md) to authenticate a user with an MCP server to get access to user-scoped tools within your LangGraph Platform deployment. 

!!! note
    Custom authentication is a LangGraph Platform feature.

An example architecture for this flow:

```mermaid
sequenceDiagram
  %% Actors
  participant ClientApp as Client
  participant AuthProv  as Auth Provider
  participant LangGraph as LangGraph Backend
  participant SecretStore as Secret Store
  participant MCPServer as MCP Server

  %% Platform login / AuthN
  ClientApp  ->> AuthProv: 1. Login (username / password)
  AuthProv   -->> ClientApp: 2. Return token
  ClientApp  ->> LangGraph: 3. Request with token

  Note over LangGraph: 4. Validate token (@auth.authenticate)
  LangGraph  -->> AuthProv: 5. Fetch user info
  AuthProv   -->> LangGraph: 6. Confirm validity

  %% Fetch user tokens from secret store
  LangGraph  ->> SecretStore: 6a. Fetch user tokens
  SecretStore -->> LangGraph: 6b. Return tokens

  Note over LangGraph: 7. Apply access control (@auth.on.*)

  %% MCP round-trip
  Note over LangGraph: 8. Build MCP client with user token
  LangGraph  ->> MCPServer: 9. Call MCP tool (with header)
  Note over MCPServer: 10. MCP validates header and runs tool
  MCPServer  -->> LangGraph: 11. Tool response

  %% Return to caller
  LangGraph  -->> ClientApp: 12. Return resources / tool output
```

For more information, see [MCP endpoint in LangGraph Server](../concepts/server-mcp.md#use-user-scoped-mcp-tools-in-your-deployment).

