# Rebuild Graph at Runtime

You might need to rebuild your graph with a different configuration for a new run. This guide shows how you can do this.

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
from langgraph.graph import END, MessageGraph

model = ChatOpenAI(temperature=0)

graph_workflow = MessageGraph()

graph_workflow.add_node("agent", model)
graph_workflow.add_edge("agent", END)
graph_workflow.set_entry_point("agent")

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

To make your graph rebuild on each new run with custom configuration, you need to rewrite `openai_agent.py` to instead provide a _function_ that takes a config and returns a graph (or compiled graph) instance as follows:

```python
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessageGraph
from langchain_core.runnables import RunnableConfig

model = ChatOpenAI(temperature=0)

def make_graph(config: RunnableConfig)
    graph_workflow = MessageGraph()

    graph_workflow.add_node("agent", model)
    graph_workflow.add_edge("agent", END)
    graph_workflow.set_entry_point("agent")

    agent = graph_workflow.compile()
    return agent
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