# LangGraph

> LangGraph is a library for building stateful, multi-actor applications with LLMs, used to create agent and multi-agent workflows. Check out an introductory tutorial here.
> 
> LangGraph is inspired by Pregel and Apache Beam. The public interface draws inspiration from NetworkX. LangGraph is built by LangChain Inc, the creators of LangChain, but can be used without LangChain. '

## Tutorials

- [LangGraph Quickstart](./tutorials/introduction.ipynb): Build a chatbot that can use tools and keep track of conversation history. Add human-in-the-loop capabilities and explore how time-travel works.
- [Common Workflows](./tutorials/workflows/index.md): Overview of the most common workflows using LLMs implemented with LangGraph.
- [LangGraph Server Quickstart](./tutorials/langgraph-platform/local-server.md): Launch a LangGraph server locally and interact with it using REST API and LangGraph Studio Web UI.

## How-tos

### Graph API Basics

- [How to update graph state from nodes](./how-tos/state-reducers.ipynb)
- [How to create a sequence of steps](./how-tos/sequence.ipynb)
- [How to create branches for parallel execution](./how-tos/branching.ipynb)
- [How to create and control loops with recursion limits](./how-tos/recursion-limit.ipynb)
- [How to visualize your graph](./how-tos/visualization.ipynb)

### Fine-grained Control

These guides demonstrate LangGraph features that grant fine-grained control over the
execution of your graph.

- [How to create map-reduce branches for parallel execution](./how-tos/map-reduce.ipynb)
- [How to update state and jump to nodes in graphs and subgraphs](./how-tos/command.ipynb)
- [How to add runtime configuration to your graph](./how-tos/configuration.ipynb)
- [How to add node retries](./how-tos/node-retries.ipynb)
- [How to return state before hitting recursion limit](./how-tos/return-when-recursion-limit-hits.ipynb)

### Persistence

[LangGraph Persistence](./concepts/persistence.md) makes it easy to persist state across graph runs (per-thread persistence) and across threads (cross-thread persistence). These how-to guides show how to add persistence to your graph.

- [How to add thread-level persistence to your graph](./how-tos/persistence.ipynb)
- [How to add thread-level persistence to a subgraph](./how-tos/subgraph-persistence.ipynb)
- [How to add cross-thread persistence to your graph](./how-tos/cross-thread-persistence.ipynb)
- [How to use Postgres checkpointer for persistence](./how-tos/persistence_postgres.ipynb)
- [How to use MongoDB checkpointer for persistence](./how-tos/persistence_mongodb.ipynb)
- [How to create a custom checkpointer using Redis](./how-tos/persistence_redis.ipynb)

See the below guides for how-to add persistence to your workflow using the [Functional API](./concepts/functional_api.md):

- [How to add thread-level persistence (functional API)](./how-tos/persistence-functional.ipynb)
- [How to add cross-thread persistence (functional API)](./how-tos/cross-thread-persistence-functional.ipynb)
