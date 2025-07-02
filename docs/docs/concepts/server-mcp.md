---
tags:
  - mcp
  - platform
hide:
  - tags
---

# MCP endpoint in LangGraph Server

The [Model Context Protocol (MCP)](./mcp.md) is an open protocol for describing tools and data sources in a model-agnostic format, enabling LLMs to discover and use them via a structured API. 

[LangGraph Server](./langgraph_server.md) implements MCP using the [Streamable HTTP transport](https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/transports/#streamable-http). This allows LangGraph **agents** to be exposed as **MCP tools**, making them usable with any MCP-compliant client supporting Streamable HTTP.

The MCP endpoint is available at `/mcp` on [LangGraph Server](./langgraph_server.md).

## Requirements

To use MCP, ensure you have the following dependencies installed:

- `langgraph-api >= 0.2.3`
- `langgraph-sdk >= 0.1.61`

Install them with:

```bash
pip install "langgraph-api>=0.2.3" "langgraph-sdk>=0.1.61"
```

## Usage overview

To enable MCP:

- Upgrade to use langgraph-api>=0.2.3. If you are deploying LangGraph Platform, this will be done for you automatically if you create a new revision.
- MCP tools (agents) will be automatically exposed.
- Connect with any MCP-compliant client that supports Streamable HTTP.


### Client

Use an MCP-compliant client to connect to the LangGraph server. The following examples show how to connect using different programming languages.

=== "JavaScript/TypeScript"

    ```bash
    npm install @modelcontextprotocol/sdk
    ```

    > **Note**
    > Replace `serverUrl` with your LangGraph server URL and configure authentication headers as needed.

    ```js
    import { Client } from "@modelcontextprotocol/sdk/client/index.js";
    import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";

    // Connects to the LangGraph MCP endpoint
    async function connectClient(url) {
        const baseUrl = new URL(url);
        const client = new Client({
            name: 'streamable-http-client',
            version: '1.0.0'
        });

        const transport = new StreamableHTTPClientTransport(baseUrl);
        await client.connect(transport);

        console.log("Connected using Streamable HTTP transport");
        console.log(JSON.stringify(await client.listTools(), null, 2));
        return client;
    }

    const serverUrl = "http://localhost:2024/mcp";

    connectClient(serverUrl)
        .then(() => {
            console.log("Client connected successfully");
        })
        .catch(error => {
            console.error("Failed to connect client:", error);
        });
    ```

=== "Python"


    Install the adapter with:

    ```bash
    pip install langchain-mcp-adapters
    ```

    Here is an example of how to connect to a remote MCP endpoint and use an agent as a tool:

    ```python
    # Create server parameters for stdio connection
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    import asyncio

    from langchain_mcp_adapters.tools import load_mcp_tools
    from langgraph.prebuilt import create_react_agent

    server_params = {
        "url": "https://mcp-finance-agent.xxx.us.langgraph.app/mcp",
        "headers": {
            "X-Api-Key":"lsv2_pt_your_api_key"
        }
    }

    async def main():
        async with streamablehttp_client(**server_params) as (read, write, _):
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()

                # Load the remote graph as if it was a tool
                tools = await load_mcp_tools(session)

                # Create and run a react agent with the tools
                agent = create_react_agent("openai:gpt-4.1", tools)

                # Invoke the agent with a message
                agent_response = await agent.ainvoke({"messages": "What can the finance agent do for me?"})
                print(agent_response)

    if __name__ == "__main__":
        asyncio.run(main())
    ```

## Expose an agent as MCP tool

When deployed, your agent will appear as a tool in the MCP endpoint
with this configuration:

- **Tool name**: The agent's name.
- **Tool description**: The agent's description.
- **Tool input schema**: The agent's input schema.

### Setting name and description 

You can set the name and description of your agent in `langgraph.json`:

```json
{
    "graphs": {
        "my_agent": {
            "path": "./my_agent/agent.py:graph",
            "description": "A description of what the agent does"
        }
    },
    "env": ".env"
}
```

After deployment, you can update the name and description using the LangGraph SDK.

### Schema

Define clear, minimal input and output schemas to avoid exposing unnecessary internal complexity to the LLM.

The default [MessagesState](./low_level.md#messagesstate) uses `AnyMessage`, which supports many message types but is too general for direct LLM exposure.

Instead, define **custom agents or workflows** that use explicitly typed input and output structures.

For example, a workflow answering documentation questions might look like this:

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

# Define input schema
class InputState(TypedDict):
    question: str

# Define output schema
class OutputState(TypedDict):
    answer: str

# Combine input and output
class OverallState(InputState, OutputState):
    pass

# Define the processing node
def answer_node(state: InputState):
    # Replace with actual logic and do something useful
    return {"answer": "bye", "question": state["question"]}

# Build the graph with explicit schemas
builder = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
builder.add_node(answer_node)
builder.add_edge(START, "answer_node")
builder.add_edge("answer_node", END)
graph = builder.compile()

# Run the graph
print(graph.invoke({"question": "hi"}))
```

For more details, see the [low-level concepts guide](https://langchain-ai.github.io/langgraph/concepts/low_level/#state).

## Use user-scoped MCP tools in your deployment

!!! tip "Prerequisites"

    You have added your own [custom auth middleware](https://langchain-ai.github.io/langgraph/how-tos/auth/custom_auth/) that populates the `langgraph_auth_user` object, making it accessible through configurable context for every node in your graph. 

To make user-scoped tools available to your LangGraph Platform deployment, start with implementing a snippet like the following: 

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

def get_mcp_tools_node(state, config):
    user = config["configurable"].get("langgraph_auth_user")
		 # e.g., user["github_token"], user["email"], etc.
		
    client = MultiServerMCPClient({
        "github": {
            "transport": "streamable_http", # (1)
            "url": "https://my-github-mcp-server/mcp", # (2)
            "headers": {
                "Authorization": f"Bearer {user['github_token']}" 
            }
        }
    })
    tools = await client.get_tools() # (3)
    return {"tools": tools}
	
```

1. MCP only supports adding headers to requests made to `streamable_http` and `sse` `transport` servers.
2. Your MCP server URL.
3. Get available tools from your MCP server.

## Session behavior  

The current LangGraph MCP implementation does not support sessions. Each `/mcp` request is stateless and independent.

## Authentication

The `/mcp` endpoint uses the same authentication as the rest of the LangGraph API. Refer to the [authentication guide](./auth.md) for setup details.

## Disable MCP

To disable the MCP endpoint, set `disable_mcp` to `true` in your `langgraph.json` configuration file:

```json
{
  "http": {
    "disable_mcp": true
  }
}
```

This will prevent the server from exposing the `/mcp` endpoint.
