# How to add thread-level persistence to a subgraph

<div class="admonition tip">
    <p class="admonition-title">Prerequisites</p>
    <p>
        This guide assumes familiarity with the following:
        <ul>
            <li>            
                <a href="https://langchain-ai.github.io/langgraph/concepts/low_level/#subgraphs">
                    Subgraphs
                </a>
            </li>
            <li>
                <a href="https://langchain-ai.github.io/langgraph/concepts/persistence/">
                    Persistence
                </a>
            </li>
        </ul>
    </p>
</div>

This guide shows how you can add [thread-level](https://langchain-ai.github.io/langgraph/how-tos/persistence/) persistence to graphs that use [subgraphs](https://langchain-ai.github.io/langgraph/how-tos/subgraph/).

## Setup

First, let's install the required packages


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

## Define the graph with persistence

To add persistence to a graph with subgraphs, all you need to do is pass a [checkpointer](https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.base.BaseCheckpointSaver) when **compiling the parent graph**. LangGraph will automatically propagate the checkpointer to the child subgraphs.

!!! note
    You **shouldn't provide** a checkpointer when compiling a subgraph. Instead, you must define a **single** checkpointer that you pass to `parent_graph.compile()`, and LangGraph will automatically propagate the checkpointer to the child subgraphs. If you pass the checkpointer to the `subgraph.compile()`, it will simply be ignored. This also applies when you [add a node function that invokes the subgraph](../subgraph#add-a-node-function-that-invokes-the-subgraph.md).

Let's define a simple graph with a single subgraph node to show how to do this.


```python
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict


# subgraph


class SubgraphState(TypedDict):
    foo: str  # note that this key is shared with the parent graph state
    bar: str


def subgraph_node_1(state: SubgraphState):
    return {"bar": "bar"}


def subgraph_node_2(state: SubgraphState):
    # note that this node is using a state key ('bar') that is only available in the subgraph
    # and is sending update on the shared state key ('foo')
    return {"foo": state["foo"] + state["bar"]}


subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_node(subgraph_node_2)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
subgraph = subgraph_builder.compile()


# parent graph


class State(TypedDict):
    foo: str


def node_1(state: State):
    return {"foo": "hi! " + state["foo"]}


builder = StateGraph(State)
builder.add_node("node_1", node_1)
# note that we're adding the compiled subgraph as a node to the parent graph
builder.add_node("node_2", subgraph)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
```






We can now compile the graph with an in-memory checkpointer (`MemorySaver`).


```python
checkpointer = MemorySaver()
# You must only pass checkpointer when compiling the parent graph.
# LangGraph will automatically propagate the checkpointer to the child subgraphs.
graph = builder.compile(checkpointer=checkpointer)
```

## Verify persistence works

Let's now run the graph and inspect the persisted state for both the parent graph and the subgraph to verify that persistence works. We should expect to see the final execution results for both the parent and subgraph in `state.values`.


```python
config = {"configurable": {"thread_id": "1"}}
```


```python
for _, chunk in graph.stream({"foo": "foo"}, config, subgraphs=True):
    print(chunk)
```

We can now view the parent graph state by calling `graph.get_state()` with the same config that we used to invoke the graph.


```python
graph.get_state(config).values
```






To view the subgraph state, we need to do two things:

1. Find the most recent config value for the subgraph
2. Use `graph.get_state()` to retrieve that value for the most recent subgraph config.

To find the correct config, we can examine the state history from the parent graph and find the state snapshot before we return results from `node_2` (the node with subgraph):


```python
state_with_subgraph = [
    s for s in graph.get_state_history(config) if s.next == ("node_2",)
][0]
```

The state snapshot will include the list of `tasks` to be executed next. When using subgraphs, the `tasks` will contain the config that we can use to retrieve the subgraph state:


```python
subgraph_config = state_with_subgraph.tasks[0].state
subgraph_config
```







```python
graph.get_state(subgraph_config).values
```






If you want to learn more about how to modify the subgraph state for human-in-the-loop workflows, check out this [how-to guide](https://langchain-ai.github.io/langgraph/how-tos/subgraphs-manage-state/).
