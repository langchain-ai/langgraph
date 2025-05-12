---
search:
  boost: 2
---

# LangGraph Studio

!!! info "Prerequisites"

    - [LangGraph Platform](./langgraph_platform.md)
    - [LangGraph Server](./langgraph_server.md)
    - [LangGraph CLI](./langgraph_cli.md)

LangGraph Studio is a specialized agent IDE that enables visualization, interaction, and debugging of agentic systems that implement the LangGraph Server API protocol. Studio also integrates with LangSmith to enable tracing, evaluation, and prompt engineering.

![](img/lg_studio.png)

## Features

The key features of LangGraph Studio are:

- Visualize your graph architecture
- Run and interact with your agent in a GUI
- Create and manage [assistants](assistants.md)
- View and manage [threads](../cloud/concepts/threads.md)
- View and manage [long term memory](memory.md)
- Debug agent state via [time travel](time-travel.md)


LangGraph Studio works for graphs that are deployed on [LangGraph Platform](../cloud/quick_start.md) or for graphs that are running locally via the [LangGraph Server](../tutorials/langgraph-platform/local-server.md).

LangGraph Studio supports two modes:

1. Graph
2. Chat

Graph mode exposes the full feature-set of Studio and is useful when you would like as many details about the execution of your agent, including the nodes traversed, intermediate states, and LangSmith integrations (such as adding to datasets an playground).

Chat mode is a simpler UI for iterating on and testing chat-specific agents. It is useful for business users and those who want to test overall agent behavior.