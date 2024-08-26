# Rebuild Graph at Runtime

You might need to rebuild your graph with a different configuration for a new run. For example, you might need to use a different graph state or graph structure depending on the config. This guide shows how you can do this.

!!! note "Note"
    In most cases, customizing behavior based on the config should be handled by a single graph where each node can read a config and change its behavior based on it

## Prerequisites

Make sure to check out [this how-to guide](./setup.md) on setting up your app for deployment first.

## Define graphs

Let's say you have an app with a simple graph that calls an LLM and returns the response to the user. The app file directory looks like the following:

```
my-app/
|-- requirements.txt
|-- .env
|-- openai_agent.py     # code for your graph
```

where the graph is defined in `openai_agent.py`. 

### No rebuild

In the standard LangGraph API configuration, the server uses the compiled graph instance that's defined at the top level of `openai_agent.py`, which looks like the following:

```python
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessageGraph

model = ChatOpenAI(temperature=0)

graph_workflow = MessageGraph()

graph_workflow.add_node("agent", model)
graph_workflow.add_edge("agent", END)
graph_workflow.add_edge(START, "agent")

agent = graph_workflow.compile()
```

To make the server aware of your graph, you need to specify a path to the variable that contains the `CompiledStateGraph` instance in your LangGraph API configuration (`langgraph.json`), e.g.:

```
{
    "dependencies": ["."],
    "graphs": {
        "openai_agent": "./openai_agent.py:agent",
    },
    "env": "./.env"
}
```

### Rebuild

To make your graph rebuild on each new run with custom configuration, you need to rewrite `openai_agent.py` to instead provide a _function_ that takes a config and returns a graph (or compiled graph) instance. Let's say we want to return our existing graph for user ID '1', and a tool-calling agent for other users. We can modify `openai_agent.py` as follows:

```python
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessageGraph
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


model = ChatOpenAI(temperature=0)

def make_default_graph():
    """Make a simple LLM agent"""
    graph_workflow = StateGraph(State)
    def call_model(state):
        return {"messages": [model.invoke(state["messages"])]}

    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_edge("agent", END)
    graph_workflow.add_edge(START, "agent")

    agent = graph_workflow.compile()
    return agent


def make_alternative_graph():
    """Make a tool-calling agent"""

    @tool
    def add(a: float, b: float):
        """Adds two numbers."""
        return a + b

    tool_node = ToolNode([add])
    model_with_tools = model.bind_tools([add])
    def call_model(state):
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    def should_continue(state: State):
        if state["messages"][-1].tool_calls:
            return "tools"
        else:
            return END

    graph_workflow = StateGraph(State)

    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_node("tools", tool_node)
    graph_workflow.add_edge("tools", "agent")
    graph_workflow.add_edge(START, "agent")
    graph_workflow.add_conditional_edges("agent", should_continue)

    agent = graph_workflow.compile()
    return agent


# this is the graph making function that will decide which graph to
# build based on the provided config
def make_graph(config: RunnableConfig):
    user_id = config.get("configurable", {}).get("user_id")
    # route to different graph state / structure based on the user ID
    if user_id == "1":
        return make_default_graph()
    else:
        return make_alternative_graph()
```

Finally, you need to specify the path to your graph-making function (`make_graph`) in `langgraph.json`:

```
{
    "dependencies": ["."],
    "graphs": {
        "openai_agent": "./openai_agent.py:make_graph",
    },
    "env": "./.env"
}
```

See more info on LangGraph API configuration file [here](../reference/cli.md#configuration-file)