# Overview

LangGraph is a low-level orchestration framework for building controllable agents. It enables agent orchestration — offering customizable architectures, long-term memory, and human-in-the-loop to reliably handle complex tasks.

There is an ecosystem of prebuilt packages for building **agents** built on top of LangGraph. We typically see users start with the prebuilt implementations, and go low level to customize their agents as needed.

## Package ecosystem

| Package | Purpose | Installation |
| ------- | ------- | ------------ |
| `langgraph-prebuilt` (part of `langgraph`) | [Create agents](./agents.md) | `pip install -U langgraph langchain` |
| `langgraph-supervisor` | Build [supervisor](./multi-agent.md#supervisor) multi-agent systems. | `pip install -U langgraph-supervisor` |
| `langgraph-swarm` | Build [swarm](./multi-agent.md#swarm) multi-agent systems. | `pip install -U langgraph-swarm` |
| `langchain-mcp-adapters` | Use tools and other resources [from MCP servers](./mcp.md) in your agent | `pip install -U langchain-mcp-adapters` |
| `langmem` | Manage short-term and long-term [memory](./memory.md) in your agent. | `pip install -U langmem` |
| `agentevals` | [Evaluate agents](./evals.md). | `pip install -U agentevals` |

## Why LangGraph?

Building agents with LangGraph has these central benefits:

- [**Memory**](./memory.md): Easy to add short-term memory (remember a single conversation) and long-term memory (remember information across conversations) to your agents.

- [**Human-in-the-loop**](./human-in-the-loop.md): Agent execution can be interrupted and resumed, allowing for tool call review, approval and editing via human input.

- [**Streaming**](./streaming.md): Support for streaming agent state to the user (or developer) over the course of agent's execution. You can stream agent progress, LLM tokens, tool updates, or all of those combined.

- [**Deployment**](./deployment.md): Easy onramp for testing, debugging, and deploying applications via [LangGraph Platform](https://langchain-ai.github.io/langgraph/concepts/langgraph_platform/). This includes [Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/), an IDE that enables visualization, interaction, and debugging of workflows or agents. This also includes numerous [options](https://langchain-ai.github.io/langgraph/tutorials/deployment/) for deployment.