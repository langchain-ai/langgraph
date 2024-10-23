# LangGraph Deploy

With LangGraph Deploy, you define a simple configuration file and it generates a Docker image with:

🎁 Automatic HTTP prediction server: Your agent definition is used to dynamically generate a RESTful HTTP API.

🥞 Automatic queue worker. Long-running agents or batch processing is best architected with a queue. LangGraph Deploy manages this out of the box.

☁️ Persistence storage. Optimized Postgres Checkpointers and Stores are spun up and attached to your agent, giving it persistence. This enables memory and human-in-the-loop interation patterns.

🚀 Ready for production. Deploy your model anywhere that Docker images run. Your own infrastructure, or LangGraph Cloud.

## How it works

### Define the Docker environment your agent runs in with `langgraph.json`:

```
{
  "dependencies": ["./my_agent"],
  "graphs": {
    "agent": "./my_agent/agent.py:graph"
  },
  "env": ".env"
}
```
### Define your graph in `agent.py`:

# TODO: this should be more self contained
```
# my_agent/agent.py
from typing import Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END, START
from my_agent.utils.nodes import call_model, should_continue, tool_node # import nodes
from my_agent.utils.state import AgentState # import state

# Define the config
class GraphConfig(TypedDict):
    model_name: Literal["anthropic", "openai"]

workflow = StateGraph(AgentState, config_schema=GraphConfig)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)
workflow.add_edge("action", "agent")

graph = workflow.compile()
```

To build the docker image, you first need to install the Cli:

```shell
pip install langgraph-cli
```

You can then use:

# TODO: why is `-t` needed?

```
langgraph build -t my-image
```

This will build a docker image with the LangGraph Deploy server.