# How to Set Up a LangGraph Application for Deployment

A LangGraph application must be configured with a [LangGraph API configuration file](../reference/cli.md#configuration-file) in order to be deployed to LangGraph Cloud (or to be self-hosted). This how-to guide discusses the basic steps to setup a LangGraph application for deployment using `pyproject.toml` to define your package's dependencies.

This walkthrough is based on [this repository](https://github.com/langchain-ai/langgraph-example-pyproject), which you can play around with to learn more about how to setup your LangGraph application for deployment.

!!! tip "Setup with requirements.txt"
    If you prefer using `requirements.txt` for dependency management, check out [this how-to guide](./setup.md).

!!! tip "Setup with a Monorepo"
    If you are interested in deploying a graph located inside a monorepo, take a look at [this](https://github.com/langchain-ai/langgraph-example-monorepo) repository for an example of how to do so.

The final repo structure will look something like this:

```bash
my-app/
├── my_agent # all project code lies within here
│   ├── utils # utilities for your graph
│   │   ├── __init__.py
│   │   ├── tools.py # tools for your graph
│   │   ├── nodes.py # node functions for you graph
│   │   └── state.py # state definition of your graph
│   ├── __init__.py
│   └── agent.py # code for constructing your graph
├── .env # environment variables
├── langgraph.json  # configuration file for LangGraph
└── pyproject.toml # dependencies for your project
```

After each step, an example file directory is provided to demonstrate how code can be organized.

## Specify Dependencies

Dependencies can optionally be specified in one of the following files: `pyproject.toml`, `setup.py`, or `requirements.txt`. If none of these files is created, then dependencies can be specified later in the [LangGraph API configuration file](#create-langgraph-api-config).

The dependencies below will be included in the image, you can also use them in your code, as long as with a compatible version range:

```
langgraph>=0.2.56,<0.4.0
langgraph-sdk>=0.1.53
langgraph-checkpoint>=2.0.15,<3.0
langchain-core>=0.2.38,<0.4.0
langsmith>=0.1.63
orjson>=3.9.7
httpx>=0.25.0
tenacity>=8.0.0
uvicorn>=0.26.0
sse-starlette>=2.1.0,<2.2.0
uvloop>=0.18.0
httptools>=0.5.0
jsonschema-rs>=0.20.0
structlog>=23.1.0
```

Example `pyproject.toml` file:

```toml
[tool.poetry]
name = "my-agent"
version = "0.0.1"
description = "An excellent agent build for LangGraph cloud."
authors = ["Polly the parrot <1223+polly@users.noreply.github.com>"]
license = "MIT"
readme = "README.md"

[tool.poetry.dependencies]
python = ">=3.9"
langgraph = "^0.2.0"
langchain-fireworks = "^0.1.3"


[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

Example file directory:

```bash
my-app/
└── pyproject.toml   # Python packages required for your graph
```

## Specify Environment Variables

Environment variables can optionally be specified in a file (e.g. `.env`). See the [Environment Variables reference](../reference/env_var.md) to configure additional variables for a deployment.

Example `.env` file:

```
MY_ENV_VAR_1=foo
MY_ENV_VAR_2=bar
FIREWORKS_API_KEY=key
```

Example file directory:

```bash
my-app/
├── .env             # file with environment variables
└── pyproject.toml
```

## Define Graphs

Implement your graphs! Graphs can be defined in a single file or multiple files. Make note of the variable names of each [CompiledGraph][langgraph.graph.graph.CompiledGraph] to be included in the LangGraph application. The variable names will be used later when creating the [LangGraph API configuration file](../reference/cli.md#configuration-file).

Example `agent.py` file, which shows how to import from other modules you define (code for the modules is not shown here, please see [this repo](https://github.com/langchain-ai/langgraph-example-pyproject) to see their implementation):

```python
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

!!! warning "Assign `CompiledGraph` to Variable"
    The build process for LangGraph Cloud requires that the `CompiledGraph` object be assigned to a variable at the top-level of a Python module.

Example file directory:

```bash
my-app/
├── my_agent # all project code lies within here
│   ├── utils # utilities for your graph
│   │   ├── __init__.py
│   │   ├── tools.py # tools for your graph
│   │   ├── nodes.py # node functions for you graph
│   │   └── state.py # state definition of your graph
│   ├── __init__.py
│   └── agent.py # code for constructing your graph
├── .env
└── pyproject.toml
```

## Create LangGraph API Config

Create a [LangGraph API configuration file](../reference/cli.md#configuration-file) called `langgraph.json`. See the [LangGraph CLI reference](../reference/cli.md#configuration-file) for detailed explanations of each key in the JSON object of the configuration file.

Example `langgraph.json` file:

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./my_agent/agent.py:graph"
  },
  "env": ".env"
}
```

Note that the variable name of the `CompiledGraph` appears at the end of the value of each subkey in the top-level `graphs` key (i.e. `:<variable_name>`).

!!! warning "Configuration Location"
    The LangGraph API configuration file must be placed in a directory that is at the same level or higher than the Python files that contain compiled graphs and associated dependencies.

Example file directory:

```bash
my-app/
├── my_agent # all project code lies within here
│   ├── utils # utilities for your graph
│   │   ├── __init__.py
│   │   ├── tools.py # tools for your graph
│   │   ├── nodes.py # node functions for you graph
│   │   └── state.py # state definition of your graph
│   ├── __init__.py
│   └── agent.py # code for constructing your graph
├── .env # environment variables
├── langgraph.json  # configuration file for LangGraph
└── pyproject.toml # dependencies for your project
```

## Next

After you setup your project and place it in a github repo, it's time to [deploy your app](./cloud.md).
