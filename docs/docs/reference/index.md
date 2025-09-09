---
title: Reference
description: API reference for LangGraph
search:
  boost: 0.5
---

<style>
.md-sidebar {
  display: block !important;
}
</style>

# Reference

Welcome to the LangGraph reference docs! These pages detail the core interfaces you will use when building with LangGraph. Each section covers a different part of the ecosystem.

!!! tip

    If you are just getting started, see [LangGraph basics](../concepts/why-langgraph.md) for an introduction to the main concepts and usage patterns.


## LangGraph

The core APIs for the LangGraph open source library.

- [Graphs](graphs.md): Main graph abstraction and usage.
- [Functional API](func.md): Functional programming interface for graphs.
- [Pregel](pregel.md): Pregel-inspired computation model.
- [Checkpointing](checkpoints.md): Saving and restoring graph state.
- [Storage](store.md): Storage backends and options.
- [Caching](cache.md): Caching mechanisms for performance.
- [Types](types.md): Type definitions for graph components.
- [Config](config.md): Configuration options.
- [Errors](errors.md): Error types and handling.
- [Constants](constants.md): Global constants.
- [Channels](channels.md): Message passing and channels.

## Prebuilt components

Higher-level abstractions for common workflows, agents, and other patterns.

- [Agents](agents.md): Built-in agent patterns.
- [Supervisor](supervisor.md): Orchestration and delegation.
- [Swarm](swarm.md): Multi-agent collaboration.
- [MCP Adapters](mcp.md): Integrations with external systems.

## LangGraph Platform

Tools for deploying and connecting to the LangGraph Platform.

- [SDK (Python)](../cloud/reference/sdk/python_sdk_ref.md): Python SDK for interacting with instances of the LangGraph Server.
- [SDK (JS/TS)](../cloud/reference/sdk/js_ts_sdk_ref.md): JavaScript/TypeScript SDK for interacting with instances of the LangGraph Server.
- [RemoteGraph](remote_graph.md): `Pregel` abstraction for connecting to LangGraph Server instances.

See the [LangGraph Platform reference](https://docs.langchain.com/langgraph-platform/reference-overview) for more reference documentation.