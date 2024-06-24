# Graph Definitions

Graphs are the core abstraction of LangGraph. Each [StateGraph](#stategraph) implementation is used to create graph workflows. Once compiled, you can run the [CompiledGraph](#compiledgraph) to run the application.

## StateGraph

```python
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
class MyState(TypedDict)
    ...
graph = StateGraph(MyState)
```

::: langgraph.graph.StateGraph
handler: python

## MessageGraph

::: langgraph.graph.message.MessageGraph

## CompiledGraph

::: langgraph.graph.graph.CompiledGraph
handler: python

## StreamMode

::: langgraph.pregel.StreamMode
handler: python

## Constants

The following constants and classes are used to help control graph execution.

## START

START is a string constant (`"__start__"`) that serves as a "virtual" node in the graph.
Adding an edge (or conditional edges) from `START` to node one or more nodes in your graph
will direct the graph to begin execution there.

```python
from langgraph.graph import START
...
builder.add_edge(START, "my_node")
# Or to add a conditional starting point
builder.add_conditional_edges(START, my_condition)
```

## END

END is a string constant (`"__end__"`) that serves as a "virtual" node in the graph. Adding
an edge (or conditional edges) from one or more nodes in your graph to the `END` "node" will
direct the graph to cease execution as soon as it reaches this point.

```python
from langgraph.graph import END
...
builder.add_edge("my_node", END) # Stop any time my_node completes
# Or to conditionally terminate
def my_condition(state):
    if state["should_stop"]:
        return END
    return "my_node"
builder.add_conditional_edges("my_node", my_condition)
```

## Send

::: langgraph.constants.Send
handler: python
