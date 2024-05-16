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

