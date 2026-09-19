---
type: Getting Started
title: Quick Start
description: Navigate the LangGraph repository and wiki by understanding the monorepo structure, core concepts, and use-case-based pathways through the documentation.
tags: [monorepo, repository-structure, navigation, learning-path, architecture, deployment]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-03T15:19:56.419Z
sources:
  - id: openwiki-source-8037e2358a2c4f9b2c722a11
    resource: repo://AGENTS.md
  - id: openwiki-source-d938b485005732564a37e4a1
    resource: repo://libs/checkpoint/README.md
  - id: openwiki-source-75f1e442bb6ed78ddb1df39d
    resource: repo://libs/cli/README.md
  - id: openwiki-source-36058e684ab11833a28af83b
    resource: repo://libs/langgraph/langgraph/graph/state.py
  - id: openwiki-source-947a0c3d1c7e56087c7c9d5e
    resource: repo://libs/prebuilt/README.md
  - id: openwiki-source-583d6fb6d1a776a95837b56a
    resource: repo://libs/sdk-py/README.md
  - id: openwiki-source-23775c3de52f3ab95a13cb8b
    resource: repo://README.md
generated: { by: "openwiki/0.5.0", at: "2026-09-03T15:19:56.419Z" }
---

## Welcome to LangGraph

LangGraph is a low-level orchestration framework for building, managing, and deploying long-running, stateful agents. This Quick Start guide helps you navigate the monorepo structure and the wiki documentation to find exactly what you need.

---

## What is LangGraph?

LangGraph provides production-ready infrastructure for any long-running, stateful workflow or agent:

- **Durable execution** — Build agents that persist through failures and resume from exactly where they left off
- **Human-in-the-loop** — Seamlessly incorporate human oversight by inspecting and modifying agent state at any point
- **Comprehensive memory** — Create truly stateful agents with both short-term working memory and long-term persistent memory across sessions
- **Debugging and observability** — Gain deep visibility into complex agent behavior with visualization and tracing
- **Production-ready deployment** — Deploy sophisticated agent systems confidently with scalable infrastructure

---

## Monorepo Structure

The LangGraph repository is a **monorepo** with multiple interdependent libraries under `/libs/`:

```
LangGraph Monorepo
├── checkpoint (base interfaces)
├── checkpoint-postgres (Postgres backend)
├── checkpoint-sqlite (SQLite backend)
├── langgraph (core orchestration framework)
├── prebuilt (high-level agent APIs)
├── cli (command-line tooling)
├── sdk-py (Python API client)
└── sdk-js (JavaScript/TypeScript API client)
```

### Dependency Flow

Understanding dependencies helps predict what changes affect what:

```
checkpoint (base, no deps)
├── checkpoint-postgres
├── checkpoint-sqlite
├── langgraph (core runtime)
└── prebuilt (high-level agents)

sdk-py depends on langgraph + cli
sdk-js (standalone)
```

**Key Principle:** The base `checkpoint` library defines the interface all checkpoint implementations follow. The `langgraph` core depends on checkpoints and implements the graph execution engine. `prebuilt` provides high-level agent patterns built on top of `langgraph`. The CLI (`cli`) and SDKs (`sdk-py`, `sdk-js`) let you run and interact with graphs in production.

---

## Quick Install

Install the main LangGraph library:

```bash
pip install -U langgraph
```

Or use your package manager:

```bash
uv add langgraph
```

For development mode with hot-reload:

```bash
uv add "langgraph-cli[inmem]"
```

For specific checkpoint backends:

```bash
uv add langgraph-checkpoint-postgres  # Postgres
uv add langgraph-checkpoint-sqlite    # SQLite
```

---

## Wiki Organization

This wiki is organized by **system ownership and workflows**, not file structure. Use the routing guide below to find what you need.

### 📐 Architecture & Design

Start here to understand **how LangGraph works** at a fundamental level:

| Page | Purpose |
|------|---------|
| **[Core Concepts](/openwiki/architecture/core-concepts.md)** | Foundational concepts: Pregel algorithm, nodes, edges, channels, state, task execution. Start here if you're new to LangGraph. |
| **[Graph Execution Model](/openwiki/architecture/graph-execution-model.md)** | Deep dive into the superstep algorithm, task scheduling, state advancement, and runtime flow. |
| **[State and Channels](/openwiki/architecture/state-and-channels.md)** | How state is defined, typed, and updated; channel semantics; reducers. |
| **[Checkpointing and Memory](/openwiki/architecture/checkpointing-and-memory.md)** | The checkpoint abstraction, how persistence enables durable execution and state resumption. |

**When to read:** You're learning the architecture, designing a custom agent system, or troubleshooting execution behavior.

---

### 🔧 Core Concepts & Advanced Topics

Specialized concepts that enable powerful patterns:

| Page | Purpose |
|------|---------|
| **[Command and Send](/openwiki/concepts/command-and-send.md)** | Advanced control flow: `Command` for error/interrupt handling, `Send` for dynamic branching and fan-out. |
| **[Functional API](/openwiki/concepts/functional-api.md)** | The `@entrypoint` and `@task` decorators for function-based graph composition (alternative to `StateGraph`). |
| **[Managed Values](/openwiki/concepts/managed-values.md)** | Special state fields auto-populated by runtime (e.g., "is this the last step?"). |
| **[Subgraphs and Nesting](/openwiki/concepts/subgraphs-and-nesting.md)** | Composing graphs by nesting them as nodes, schema adaptation, and isolation. |

**When to read:** You need to handle complex control flow, nest graphs, or use advanced patterns.

---

### 🛠️ Workflows & Patterns

Practical guides for common building tasks:

| Page | Purpose |
|------|---------|
| **[Graph Building Patterns](/openwiki/workflows/graph-building.md)** | Constructing `StateGraph` instances with nodes, edges, branching, error handling, and subgraphs. Start here for practical graph construction. |
| **[Execution and Streaming](/openwiki/workflows/execution-and-streaming.md)** | How to invoke graphs, stream results, handle interrupts, and consume output in different modes. |
| **[Error Handling and Recovery](/openwiki/workflows/error-handling-and-recovery.md)** | Error handling mechanisms, retry policies, timeouts, and node-level error handlers. |

**When to read:** You're building a graph or implementing a specific workflow pattern.

---

### 🧰 Integrations

Using LangGraph with external systems and APIs:

| Page | Purpose |
|------|---------|
| **[Prebuilt Agents and Components](/openwiki/integrations/prebuilt-agents.md)** | Overview of `langgraph.prebuilt`: `create_react_agent`, `ToolNode`, and ready-made agent components. Use when you want to quickly build tool-calling agents. |
| **[SDK and API Integration](/openwiki/integrations/sdk-and-api.md)** | Interaction with LangGraph via Python/JavaScript SDKs and the REST API. |

**When to read:** You're integrating LangGraph with external systems or using prebuilt components.

---

### ⚙️ Operations

Developing, deploying, and managing LangGraph applications:

| Page | Purpose |
|------|---------|
| **[Checkpoint Persistence](/openwiki/operations/checkpoint-persistence.md)** | Setup and operation of checkpoint savers for durable execution and state management. |
| **[CLI and Deployment](/openwiki/operations/cli-and-deployment.md)** | `langgraph` CLI commands for development (`langgraph dev`), building (`langgraph build`), and production deployment. Start here for local development or deploying to production. |

**When to read:** You're setting up local development, configuring persistence, or deploying to production.

---

### 🧪 Testing

Writing effective tests for LangGraph applications:

| Page | Purpose |
|------|---------|
| **[Test Patterns](/openwiki/testing/test-patterns.md)** | How to write tests for graphs, channels, nodes, and integration workflows. |

**When to read:** You're building a test suite for your graph.

---

## Common Use Cases & Learning Paths

### "I'm new to LangGraph. Where do I start?"

1. **Understand the basics:** Read [Core Concepts](/openwiki/architecture/core-concepts.md) to learn about Pregel, nodes, edges, channels, and state.
2. **Build your first graph:** Follow [Graph Building Patterns](/openwiki/workflows/graph-building.md) to construct a simple StateGraph.
3. **Run and debug:** Use [Execution and Streaming](/openwiki/workflows/execution-and-streaming.md) to invoke your graph and observe output.

**Recommended time:** 30-45 minutes.

---

### "I want to build a tool-calling agent."

1. **Use prebuilt components:** Start with [Prebuilt Agents and Components](/openwiki/integrations/prebuilt-agents.md). The `create_react_agent` factory handles most common cases.
2. **Customize with custom tools:** Implement your tools and pass them to the factory.
3. **Add persistence (optional):** See [Checkpoint Persistence](/openwiki/operations/checkpoint-persistence.md) to save agent state between runs.

**Recommended time:** 20-30 minutes for basic agent; add 15-20 minutes for persistence.

---

### "I need to handle errors, retries, and timeouts."

1. **Review error handling mechanics:** Read [Error Handling and Recovery](/openwiki/workflows/error-handling-and-recovery.md) to understand retry policies, error handlers, and timeout behavior.
2. **Implement in your graph:** Apply the patterns to your nodes using `set_node_defaults()` and per-node configuration.

**Recommended time:** 15-25 minutes.

---

### "I want to add human-in-the-loop interrupts to my graph."

1. **Understand state checkpoints:** Read [Checkpointing and Memory](/openwiki/architecture/checkpointing-and-memory.md) to understand how state is persisted.
2. **Configure persistence:** Follow [Checkpoint Persistence](/openwiki/operations/checkpoint-persistence.md) to set up a checkpoint saver.
3. **Implement interrupts:** Use patterns from [Execution and Streaming](/openwiki/workflows/execution-and-streaming.md) to inspect state mid-execution and resume with modifications.

**Recommended time:** 30-40 minutes.

---

### "I'm deploying to production. What do I need?"

1. **Set up persistence:** Configure a checkpoint saver (Postgres, SQLite, or custom). See [Checkpoint Persistence](/openwiki/operations/checkpoint-persistence.md).
2. **Build and containerize:** Use the [CLI and Deployment](/openwiki/operations/cli-and-deployment.md) guide to build Docker images and deploy.
3. **Integrate with production systems:** Use [SDK and API Integration](/openwiki/integrations/sdk-and-api.md) to interact with your graph from application code.

**Recommended time:** 45-60 minutes (including testing).

---

### "I need to compose multiple graphs together."

1. **Understand subgraphs:** Read [Subgraphs and Nesting](/openwiki/concepts/subgraphs-and-nesting.md) to learn how to nest graphs as nodes.
2. **Use dynamic routing:** Learn [Command and Send](/openwiki/concepts/command-and-send.md) for complex control flow and dynamic fan-out.
3. **Build and test:** Apply patterns from [Graph Building Patterns](/openwiki/workflows/graph-building.md) and [Test Patterns](/openwiki/testing/test-patterns.md).

**Recommended time:** 40-60 minutes.

---

### "I want to understand the advanced execution model."

1. **Start with core concepts:** Refresh yourself on [Core Concepts](/openwiki/architecture/core-concepts.md).
2. **Deep dive into execution:** Read [Graph Execution Model](/openwiki/architecture/graph-execution-model.md) for supersteps, task scheduling, and state advancement.
3. **Understand state mechanics:** Review [State and Channels](/openwiki/architecture/state-and-channels.md) for channel semantics and concurrency.
4. **Optional: Explore source code:** Read the Pregel implementation in `repo://libs/langgraph/langgraph/pregel/` for the runtime details.

**Recommended time:** 60-90 minutes.

---

## Key Concepts at a Glance

### StateGraph

The primary API for building LangGraph applications. A builder that accepts a typed state schema and lets you register nodes, edges, and policies before compiling into an executable graph.

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    messages: list

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_edge(START, "agent")
graph.add_edge("agent", END)

compiled = graph.compile()  # Must compile to run
result = compiled.invoke({"messages": []})
```

### Nodes

Functions or runnables that read and write state. Executed in parallel when possible.

### Edges

Control flow paths connecting nodes. Can be deterministic or conditional (routing based on state inspection).

### State & Channels

Shared, typed storage for state across nodes. Each channel has:
- **Type:** The Python type (e.g., `list`, `dict`, custom class)
- **Reducer:** Optional function to merge concurrent updates (e.g., `operator.add` for lists)

### Supersteps

Atomic execution units. All nodes that have input available execute in parallel within a superstep. The runtime waits for all superstep tasks to complete, applies writes, then moves to the next superstep.

### Checkpoints

Snapshots of graph state at a given superstep, stored via a checkpointer. Enable durable execution, resumption, interrupts, and time-travel debugging.

### Command

Return value from nodes to signal special control flow: retry the node, interrupt for human input, or branch to specific nodes without conditional routing.

### Send

Dynamically branch to multiple nodes in parallel from a single node (fan-out pattern).

---

## What's in the Monorepo?

### `/libs/checkpoint`

Base interfaces (`BaseCheckpointSaver`, `CheckpointTuple`) that all checkpoint implementations follow. This is the foundation for durable execution.

**Use when:** Implementing a custom checkpoint backend or understanding the checkpoint protocol.

### `/libs/checkpoint-postgres`, `/libs/checkpoint-sqlite`

Production-ready implementations of checkpointers using Postgres and SQLite.

**Use when:** Deploying to production and need to persist state reliably.

### `/libs/langgraph`

Core orchestration framework:
- `StateGraph` — Graph builder
- `Pregel` — Graph execution engine
- Channels, state management, task scheduling
- Streaming and interrupt handling

**Use when:** Building any graph-based agent or workflow.

### `/libs/prebuilt`

High-level APIs for common patterns:
- `create_react_agent` — ReAct-style tool-calling agent factory
- `ToolNode` — Executor for tool calls
- Agent inbox schemas

**Use when:** Building tool-calling agents quickly without low-level customization.

### `/libs/cli`

Command-line interface:
- `langgraph new` — Project scaffolding
- `langgraph dev` — Local development with hot-reload
- `langgraph build` — Docker image creation
- `langgraph up`, `langgraph deploy` — Deployment commands

**Use when:** Setting up local development or deploying to production.

### `/libs/sdk-py`

Python SDK for the LangGraph API. Lets you interact with a running LangGraph server from Python code.

**Use when:** Building a client application that invokes graphs on a remote server.

### `/libs/sdk-js`

JavaScript/TypeScript SDK for the LangGraph API.

**Use when:** Building a Node.js or browser client application that invokes graphs on a remote server.

---

## Quick Reference: Which Library?

| Task | Library | Notes |
|------|---------|-------|
| Build a graph | `langgraph` (StateGraph) | Core API |
| Use prebuilt patterns | `langgraph.prebuilt` | High-level agents |
| Persist state | `checkpoint`, `checkpoint-postgres`, `checkpoint-sqlite` | Durable execution |
| Command-line tools | `langgraph-cli` | Development & deployment |
| Connect from Python | `langgraph-sdk` | API client |
| Connect from Node.js/browser | `langgraph-sdk-js` | API client |

---

## Next Steps

1. **Pick a learning path** from the "Common Use Cases" section above.
2. **Visit the relevant wiki pages** to dive deeper into the concepts you need.
3. **Explore examples** in `/examples/` (note: many are archived; prefer docs.langchain.com for up-to-date examples).
4. **Read source code** in `/libs/langgraph/langgraph/` once you're comfortable with concepts.
5. **Check API reference** at [reference.langchain.com/python/langgraph](https://reference.langchain.com/python/langgraph/) for detailed function signatures.

---

## Key Insights

- **LangGraph is a framework, not an agent library.** Use it when you need fine control over execution, persistence, and state management. For quick agent building, start with prebuilt patterns or LangChain agents.
- **Compilation is mandatory.** `StateGraph` is a builder; you must call `.compile()` to get an executable graph.
- **State is explicit.** Unlike function calls, all communication between nodes flows through the state schema. This makes graphs testable, debuggable, and introspectable.
- **Checkpoints are optional but powerful.** Without a checkpointer, graphs run in-memory and resume from the beginning on failure. With a checkpointer, they resume from the last checkpoint.
- **The monorepo structure reflects dependencies.** Changes to `checkpoint` affect everything downstream. Changes to `langgraph` affect `prebuilt` and `cli`. Plan testing accordingly.

---

## Additional Resources

- **[LangGraph Documentation](https://docs.langchain.com/oss/python/langgraph/overview)** — Comprehensive guides and tutorials
- **[API Reference](https://reference.langchain.com/python/langgraph/)** — Full function signatures and type definitions
- **[LangSmith](https://www.langchain.com/langsmith)** — Observability and debugging platform for LangGraph
- **[LangGraph Academy](https://academy.langchain.com/courses/intro-to-langgraph)** — Free structured course
- **[LangChain Forum](https://forum.langchain.com)** — Community support and discussions
