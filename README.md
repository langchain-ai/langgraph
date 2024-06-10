# 🦜🕸️LangGraph

![Version](https://img.shields.io/pypi/v/langgraph)
[![Downloads](https://static.pepy.tech/badge/langgraph/month)](https://pepy.tech/project/langgraph)
[![Open Issues](https://img.shields.io/github/issues-raw/langchain-ai/langgraph)](https://github.com/langchain-ai/langgraph/issues)
[![](https://dcbadge.vercel.app/api/server/6adMQxSpJS?compact=true&style=flat)](https://discord.com/channels/1038097195422978059/1170024642245832774)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://langchain-ai.github.io/langgraph/)

⚡ Building language agents as graphs ⚡

## Overview

[LangGraph](https://langchain-ai.github.io/langgraph/) is a library for building stateful, multi-actor applications with LLMs.
Inspired by [Pregel](https://research.google/pubs/pub37252/) and [Apache Beam](https://beam.apache.org/), LangGraph lets you coordinate and checkpoint multiple chains (or actors) across cyclic computational steps using regular python functions (or [JS](https://github.com/langchain-ai/langgraphjs)). The public interface draws inspiration from [NetworkX](https://networkx.org/documentation/latest/).

The main use is for adding **cycles** and **persistence** to your LLM application. If you only need quick Directed Acyclic Graphs (DAGs), you can already accomplish this using [LangChain Expression Language](https://python.langchain.com/docs/expression_language/).

Cycles are important for agentic behaviors, where you call an LLM in a loop, asking it what action to take next.

### Key Features

- **Stateful Graphs**: Maintain state across nodes for complex, multi-step computations.
- **Cycles and Branching**: Implement loops and conditionals in your apps.
- **Persistence**: Automatically save state after each step in the graph. Pause and resume the graph execution at any point to support error recovery, human-in-the-loop workflows, time travel and more.
- **Human-in-the-loop**: Interrupt graph execution to approve or edit next action planned by the agent.
- **Streaming support**: Stream outputs as they are produced by each node (including token streaming).
- **Integration with LangChain**: LangGraph integrates seamlessly with [LangChain](https://github.com/langchain-ai/langchain/) and [LangSmith](https://docs.smith.langchain.com/).


## Installation

```shell
pip install -U langgraph
```

## Example

One of the central concepts of LangGraph is state. Each graph execution creates a state that is passed between nodes in the graph as they execute, and each node updates this internal state with its return value after it executes. The way that the graph updates its internal state is defined by either the type of graph chosen or a custom function.

Let's take a look at a simple example. The graph below contains a single node called `"oracle"` that executes a chat model, then returns the result:

```shell
pip install langchain_openai
```

```shell
export OPENAI_API_KEY=sk-...
```

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import END, MessageGraph

model = ChatOpenAI(temperature=0)

graph = MessageGraph()

graph.add_node("oracle", model)
graph.add_edge("oracle", END)

graph.set_entry_point("oracle")

runnable = graph.compile()
runnable.invoke(HumanMessage("What is 1 + 1?"))
```

```
[HumanMessage(content='What is 1 + 1?'), AIMessage(content='1 + 1 equals 2.')]
```

### Step-by-step Breakdown:

1.	Initialize the model and graph (`MessageGraph`).
2. <details>
    <summary>Add nodes and edges to define the graph structure.</summary>

      - we add a single node to the graph, called `"oracle"`, which simply calls the model with the given input.
      - we add an edge from this `"oracle"` node to the special string `END` (`"__end__"`). This means that execution will end after the current node.
   </details>
3.	Set the entry point for graph execution (`oracle`).
4.	<details>
    <summary>Compile the graph.</summary>
    When we compile the graph, we are translating it to low-level [pregel operations](https://research.google/pubs/pregel-a-system-for-large-scale-graph-processing/) ensuring that it can be run.
    </details>
5. <details>
   <summary>Execute the graph</summary>

      1. LangGraph adds the input message to the internal state, then passes the state to the entrypoint node, `"oracle"`.
      2. The `"oracle"` node executes, invoking the chat model.
      3. The chat model returns an `AIMessage`. LangGraph adds this to the state.
      4. Execution progresses to the special `END` value and outputs the final state.
    And as a result, we get a list of two chat messages as output.
   </details>

## Advanced usage

For more advanced examples of LangGraph agents with with tool calling, conditional edges and cycles see [Quick Start](https://langchain-ai.github.io/langgraph/how-tos/docs/quickstart/)


## Documentation

We hope this gave you a taste of what you can build! Check out the rest of the docs to learn more.

* [Tutorials](https://langchain-ai.github.io/langgraph/tutorials/): Learn to build with LangGraph through guided examples.
* [How-to Guides](https://langchain-ai.github.io/langgraph/how-tos/): Accomplish specific things within LangGraph, from streaming, to adding memory & persistence, to common design patterns (branching, subgraphs, etc.), these are the place to go if you want to copy and run a specific code snippet.
* [Conceptual Guides](https://langchain-ai.github.io/langgraph/concepts/): In-depth explanations of the key concepts and principles behind LangGraph, such as nodes, edges, state and more.
* [API Reference](https://langchain-ai.github.io/langgraph/reference/graphs/): Review important classes and methods, simple examples of how to use the graph and checkpointing APIs, higher-level prebuilt components and more.

## Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see [here](https://python.langchain.com/v0.2/docs/contributing/).