# How to transform inputs and outputs of a subgraph

It's possible that your subgraph state is completely independent from the parent graph state, i.e. there are no overlapping channels (keys) between the two. For example, you might have a supervisor agent that needs to produce a report with a help of multiple ReAct agents. ReAct agent subgraphs might keep track of a list of messages whereas the supervisor only needs user input and final report in its state, and doesn't need to keep track of messages.

In such cases you need to transform the inputs to the subgraph before calling it and then transform its outputs before returning. This guide shows how to do that.

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

## Define graph and subgraphs

Let's define 3 graphs:
- a parent graph
- a child subgraph that will be called by the parent graph
- a grandchild subgraph that will be called by the child graph

### Define grandchild


```python
from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START, END


class GrandChildState(TypedDict):
    my_grandchild_key: str


def grandchild_1(state: GrandChildState) -> GrandChildState:
    # NOTE: child or parent keys will not be accessible here
    return {"my_grandchild_key": state["my_grandchild_key"] + ", how are you"}


grandchild = StateGraph(GrandChildState)
grandchild.add_node("grandchild_1", grandchild_1)

grandchild.add_edge(START, "grandchild_1")
grandchild.add_edge("grandchild_1", END)

grandchild_graph = grandchild.compile()
```


```python
grandchild_graph.invoke({"my_grandchild_key": "hi Bob"})
```






### Define child


```python
class ChildState(TypedDict):
    my_child_key: str


def call_grandchild_graph(state: ChildState) -> ChildState:
    # NOTE: parent or grandchild keys won't be accessible here
    # we're transforming the state from the child state channels (`my_child_key`)
    # to the child state channels (`my_grandchild_key`)
    grandchild_graph_input = {"my_grandchild_key": state["my_child_key"]}
    # we're transforming the state from the grandchild state channels (`my_grandchild_key`)
    # back to the child state channels (`my_child_key`)
    grandchild_graph_output = grandchild_graph.invoke(grandchild_graph_input)
    return {"my_child_key": grandchild_graph_output["my_grandchild_key"] + " today?"}


child = StateGraph(ChildState)
# NOTE: we're passing a function here instead of just compiled graph (`child_graph`)
child.add_node("child_1", call_grandchild_graph)
child.add_edge(START, "child_1")
child.add_edge("child_1", END)
child_graph = child.compile()
```


```python
child_graph.invoke({"my_child_key": "hi Bob"})
```






<div class="admonition info">
    <p class="admonition-title">Note</p>
    <p>
    We're wrapping the <code>grandchild_graph</code> invocation in a separate function (<code>call_grandchild_graph</code>) that transforms the input state before calling the grandchild graph and then transforms the output of grandchild graph back to child graph state. If you just pass <code>grandchild_graph</code> directly to <code>.add_node</code> without the transformations, LangGraph will raise an error as there are no shared state channels (keys) between child and grandchild states.
    </p>
</div>

Note that child and grandchild subgraphs have their own, **independent** state that is not shared with the parent graph.

### Define parent


```python
class ParentState(TypedDict):
    my_key: str


def parent_1(state: ParentState) -> ParentState:
    # NOTE: child or grandchild keys won't be accessible here
    return {"my_key": "hi " + state["my_key"]}


def parent_2(state: ParentState) -> ParentState:
    return {"my_key": state["my_key"] + " bye!"}


def call_child_graph(state: ParentState) -> ParentState:
    # we're transforming the state from the parent state channels (`my_key`)
    # to the child state channels (`my_child_key`)
    child_graph_input = {"my_child_key": state["my_key"]}
    # we're transforming the state from the child state channels (`my_child_key`)
    # back to the parent state channels (`my_key`)
    child_graph_output = child_graph.invoke(child_graph_input)
    return {"my_key": child_graph_output["my_child_key"]}


parent = StateGraph(ParentState)
parent.add_node("parent_1", parent_1)
# NOTE: we're passing a function here instead of just a compiled graph (`<code>child_graph</code>`)
parent.add_node("child", call_child_graph)
parent.add_node("parent_2", parent_2)

parent.add_edge(START, "parent_1")
parent.add_edge("parent_1", "child")
parent.add_edge("child", "parent_2")
parent.add_edge("parent_2", END)

parent_graph = parent.compile()
```

<div class="admonition info">
    <p class="admonition-title">Note</p>
    <p>
    We're wrapping the <code>child_graph</code> invocation in a separate function (<code>call_child_graph</code>) that transforms the input state before calling the child graph and then transforms the output of the child graph back to parent graph state. If you just pass <code>child_graph</code> directly to <code>.add_node</code> without the transformations, LangGraph will raise an error as there are no shared state channels (keys) between parent and child states.
    </p>
</div>

Let's run the parent graph and make sure it correctly calls both the child and grandchild subgraphs:


```python
parent_graph.invoke({"my_key": "Bob"})
```






Perfect! The parent graph correctly calls both the child and grandchild subgraphs (which we know since the ", how are you" and "today?" are added to our original "my_key" state value).
