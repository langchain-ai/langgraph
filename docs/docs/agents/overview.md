---
title: Overview
search:
  boost: 2
tags:
  - agent
hide:
  - tags
---

# Agent development with LangGraph

**LangGraph** provides both low-level primitives and high-level prebuilt components for building agent-based applications. This section focuses on the **prebuilt**, **reusable** components designed to help you construct agentic systems quickly and reliably—without the need to implement orchestration, memory, or human feedback handling from scratch.

## Key features

LangGraph includes several capabilities essential for building robust, production-ready agentic systems:

- [**Memory integration**](./memory.md): Native support for *short-term* (session-based) and *long-term* (persistent across sessions) memory, enabling stateful behaviors in chatbots and assistants.
- [**Human-in-the-loop control**](./human-in-the-loop.md): Execution can pause *indefinitely* to await human feedback—unlike websocket-based solutions limited to real-time interaction. This enables asynchronous approval, correction, or intervention at any point in the workflow.
- [**Streaming support**](./streaming.md): Real-time streaming of agent state, model tokens, tool outputs, or combined streams.
- [**Deployment tooling**](./deployment.md): Includes infrastructure-free deployment tools. [**LangGraph Platform**](https://langchain-ai.github.io/langgraph/concepts/langgraph_platform/) supports testing, debugging, and deployment.
    - **[Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/)**: A visual IDE for inspecting and debugging workflows.
    - Supports multiple [**deployment options**](https://langchain-ai.github.io/langgraph/tutorials/deployment/) for production.

## High-level building blocks

LangGraph comes with a set of prebuilt components that implement common agent behaviors and workflows. These abstractions are built on top of the LangGraph framework, offering a faster path to production while remaining flexible for advanced customization.

Using LangGraph for agent development allows you to focus on your application's logic and behavior, instead of building and maintaining the supporting infrastructure for state, memory, and human feedback.

## Package ecosystem

The high-level components are organized into several packages, each with a specific focus.

| Package                                    | Description                                                                 | Installation                            |
|--------------------------------------------|-----------------------------------------------------------------------------|-----------------------------------------|
| `langgraph-prebuilt` (part of `langgraph`) | Prebuilt components to [**create agents**](./agents.md)                     | `pip install -U langgraph langchain`    |
| `langgraph-supervisor`                     | Tools for building [**supervisor**](./multi-agent.md#supervisor) agents     | `pip install -U langgraph-supervisor`   |
| `langgraph-swarm`                          | Tools for building a [**swarm**](./multi-agent.md#swarm) multi-agent system | `pip install -U langgraph-swarm`        |
| `langchain-mcp-adapters`                   | Interfaces to [**MCP servers**](./mcp.md) for tool and resource integration | `pip install -U langchain-mcp-adapters` |
| `langmem`                                  | Agent memory management: [**short-term and long-term**](./memory.md)        | `pip install -U langmem`                |
| `agentevals`                               | Utilities to [**evaluate agent performance**](./evals.md)                   | `pip install -U agentevals`             |

