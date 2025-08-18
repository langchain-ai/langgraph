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

:::python
To use MCP, ensure you have the following dependencies installed:

- `langgraph-api >= 0.2.3`
- `langgraph-sdk >= 0.1.61`

Install them with:

```bash
pip install "langgraph-api>=0.2.3" "langgraph-sdk>=0.1.61"
```

:::

:::js
To use MCP, ensure you have both the api and sdk packages installed.

```bash
npm install @langchain/langgraph-api @langchain/langgraph-sdk
```

:::

## Exposing an agent as MCP tool

When deployed, your agent will appear as a tool in the MCP endpoint
with this configuration:

- **Tool name**: The agent's name.
- **Tool description**: The agent's description.
- **Tool input schema**: The agent's input schema.

### Setting name and description

You can set the name and description of your agent in `langgraph.json`:

:::python

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

:::
:::js

```json
{
  "graphs": {
    "my_agent": {
      "path": "./my_agent/agent.ts:graph",
      "description": "A description of what the agent does"
    }
  },
  "env": ".env"
}
```

:::

After deployment, you can update the name and description using the LangGraph SDK.

### Schema

Define clear, minimal input and output schemas to avoid exposing unnecessary internal complexity to the LLM.

:::python
The default [MessagesState](./low_level.md#messagesstate) uses `AnyMessage`, which supports many message types but is too general for direct LLM exposure.
:::

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

## Usage overview

To enable MCP:

- Upgrade to use langgraph-api>=0.2.3. If you are deploying LangGraph Platform, this will be done for you automatically if you create a new revision.
- MCP tools (agents) will be automatically exposed.
- Connect with any MCP-compliant client that supports Streamable HTTP.

### Client

:::python
Use an MCP-compliant client to connect to the LangGraph server. The following example shows how to connect using [langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters).

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

:::

:::js
Use an MCP-compliant client to connect to the LangGraph server. The following example shows how to connect using [`@langchain/mcp-adapters`](https://npmjs.com/package/@langchain/mcp-adapters).

```bash
npm install @langchain/mcp-adapters
```

Here is an example of how to connect to a remote MCP endpoint and use an agent as a tool:

```typescript
import { MultiServerMCPClient } from "@langchain/mcp-adapters";
import { createReactAgent } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";

async function main() {
  const client = new MultiServerMCPClient({
    mcpServers: {
      "finance-agent": {
        url: "https://mcp-finance-agent.xxx.us.langgraph.app/mcp",
        headers: {
          "X-Api-Key": "lsv2_pt_your_api_key",
        },
      },
    },
  });

  const tools = await client.getTools();

  const model = new ChatOpenAI({
    model: "gpt-4o-mini",
    temperature: 0,
  });

  const agent = createReactAgent({
    model,
    tools,
  });

  const response = await agent.invoke({
    input: "What can the finance agent do for me?",
  });

  console.log(response);
}

main();
```

:::

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
