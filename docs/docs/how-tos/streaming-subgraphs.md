# How to stream from subgraphs

!!! info "Prerequisites"

    This guide assumes familiarity with the following:
    
    - [Subgraphs](../../concepts/low_level/#subgraphs.md)
    - [Chat Models](https://python.langchain.com/docs/concepts/chat_models/)

If you have created a graph with [subgraphs](../subgraph.md), you may wish to stream outputs from those subgraphs. To do so, you can specify `subgraphs=True` in parent graph's `.stream()` method:


```python
for chunk in parent_graph.stream(
    {"foo": "foo"},
    # highlight-next-line
    subgraphs=True
):
    print(chunk)
```

## Setup

First let's install the required packages


```
%%capture --no-stderr
%pip install -U langgraph
```

<div class="admonition tip">
    <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
    <p style="padding-top: 5px;">
        Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started <a href="https://docs.smith.langchain.com">here</a>. 
    </p>
</div>

## Example

Let's define a simple example:


```python
from langgraph.graph import START, StateGraph
from typing import TypedDict


# Define subgraph
class SubgraphState(TypedDict):
    foo: str  # note that this key is shared with the parent graph state
    bar: str


def subgraph_node_1(state: SubgraphState):
    return {"bar": "bar"}


def subgraph_node_2(state: SubgraphState):
    return {"foo": state["foo"] + state["bar"]}


subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_node(subgraph_node_2)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
subgraph = subgraph_builder.compile()


# Define parent graph
class ParentState(TypedDict):
    foo: str


def node_1(state: ParentState):
    return {"foo": "hi! " + state["foo"]}


builder = StateGraph(ParentState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", subgraph)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
graph = builder.compile()
```

Let's now stream the outputs from the graph:


```python
for chunk in graph.stream({"foo": "foo"}, stream_mode="updates"):
    print(chunk)
```

You can see that we're only emitting the updates from the parent graph nodes (`node_1` and `node_2`). To emit the updates from the _subgraph_ nodes you can specify `subgraphs=True`:


```python
for chunk in graph.stream(
    {"foo": "foo"},
    stream_mode="updates",
    # highlight-next-line
    subgraphs=True,
):
    print(chunk)
```

Voila! The streamed outputs now contain updates from both the parent graph and the subgraph. **Note** that we are receiving not just the node updates, but we also the namespaces which tell us what graph (or subgraph) we are streaming from.
