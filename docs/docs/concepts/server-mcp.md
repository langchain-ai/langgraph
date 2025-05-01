---
tags:
    - mcp
    - platform
hide:
    - toc
---

# Model context protocol (MCP)

The **Model Context Protocol (MCP)** is an open standard that defines how applications supply context to large language models (LLMs).

The [LangGraph server](./langgraph_server.md) implements MCP using the [Streamable HTTP transport](https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/transports/#streamable-http). This allows LangGraph **agents** to be exposed as **MCP tools**, making them usable with any MCP-compliant client supporting Streamable HTTP.

The MCP endpoint is available at:

```
/mcp
```

on the LangGraph server.

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

- Upgrade to a recent LangGraph server version.
- MCP tools (agents) will be automatically exposed.
- Connect with any MCP-compliant client that supports Streamable HTTP.

### JavaScript client example

You can use the official JavaScript client:

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

## Design your agent

To expose your agent effectively as an MCP tool, design its input and output schemas to be **clear and simple** for LLMs to process.

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
builder = StateGraph(OverallState, input=InputState, output=OutputState)
builder.add_node(answer_node)
builder.add_edge(START, "answer_node")
builder.add_edge("answer_node", END)
graph = builder.compile()

# Run the graph
print(graph.invoke({"question": "hi"}))
```

For more details, see the [low-level concepts guide](https://langchain-ai.github.io/langgraph/concepts/low_level/#state).

!!! important "Keep schemas clean"

    Define clear, minimal input and output schemas to avoid exposing unnecessary internal complexity to the LLM.

## MCP Sessions 

The current implementation of MCP in LangGraph does not support sessions. This means that each request to the `/mcp` endpoint is stateless and does not maintain any session information between requests.

## Authentication

The `/mcp` endpoint uses the same authentication as the rest of the LangGraph API.

Refer to the [authentication guide](./auth.md) for setup details.

## Disabling MCP

To disable the MCP endpoint, set `disable_mcp` to `true` in your `langgraph.json` configuration file:

```json
{
  "http": {
    "disable_mcp": true
  }
}
```

This will prevent the server from exposing the `/mcp` endpoint.