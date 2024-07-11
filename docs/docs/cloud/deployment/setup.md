# How to Set Up a LangGraph Application for Deployment

A LangGraph application must be configured with a [LangGraph API configuration file](../reference/cli.md#configuration-file) in order to be deployed to LangGraph Cloud (or to be self-hosted). This how-to guide discusses the basic steps to setup a LangGraph application for deployment using `requirements.txt` to specify project dependencies. If you prefer using poetry for dependency management, check out [this how-to guide](./setup_pyproject.md) on using `pyproject.toml` for LangGraph Cloud.

The final repo structure will look something like this:

```bash
my-app/
|-- requirements.txt    # package dependencies
|-- .env                # environment variables
|-- openai_agent.py     # code for an agent
|-- anthropic_agent.py  # code for another agent
|-- langgraph.json      # configuration file for LangGraph
```

After each step, an example file directory is provided to demonstrate how code can be organized.

## Specify Dependencies

Dependencies can optionally be specified in one of the following files: `pyproject.toml`, `setup.py`, or `requirements.txt`. If none of these files is created, then dependencies can be specified later in the [LangGraph API configuration file](#create-langgraph-api-config).

Example `requirements.txt` file:
```
langgraph
langchain_openai
```

Example file directory:
```
my-app/
|-- requirements.txt    # Python packages required for your graph
```

## Specify Environment Variables

Environment variables can optionally be specified in a file (e.g. `.env`). See the [Environment Variables reference](../reference/env_var.md) to configure additional variables for a deployment.

Example `.env` file:
```
MY_ENV_VAR_1=foo
MY_ENV_VAR_2=bar
OPENAI_API_KEY=key
```

Example file directory:
```
my-app/
|-- requirements.txt
|-- .env                # file with environment variables
```

## Define Graphs

Implement your graphs! Graphs can be defined in a single file or multiple files. Make note of the variable names of each [CompiledGraph][compiledgraph] to be included in the LangGraph application. The variable names will be used later when creating the [LangGraph API configuration file](../reference/cli.md#configuration-file).

Example `openai_agent.py` file:
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

!!! warning "Assign `CompiledGraph` to Variable"
    The build process for LangGraph Cloud requires that the `CompiledGraph` object be assigned to a variable at the top-level of a Python module.

Example file directory:
```
my-app/
|-- requirements.txt
|-- .env
|-- openai_agent.py     # code for your graph
|-- anthropic_agent.py  # code for your graph
```

## Create LangGraph API Config

Create a [LangGraph API configuration file](../reference/cli.md#configuration-file) called `langgraph.json`. See the [LangGraph CLI reference](../reference/cli.md#configuration-file) for detailed explanations of each key in the JSON object of the configuration file.

Example `langgraph.json` file:
```json
{
    "dependencies": [
        "."
    ],
    "graphs": {
        "openai_agent": "./openai_agent.py:agent",
        "anthropic_agent": "./anthropic_agent.py:agent"
    },
    "env": "./.env"
}
```

Note that the variable name of the `CompiledGraph` appears at the end of the value of each subkey in the top-level `graphs` key (i.e. `:<variable_name>`).

!!! warning "Configuration Location"
    The LangGraph API configuration file must be located in a directory equal to or higher than all other Python files in the repository.

Example file directory:

```bash
my-app/
|-- requirements.txt
|-- .env
|-- openai_agent.py
|-- anthropic_agent.py
|-- langgraph.json      # configuration file for LangGraph
```

## Upload to GitHub

To deploy the LangGraph application to LangGraph Cloud, the code must be uploaded to a GitHub repository.

## Next

After you setup your repo, it's time to [deploy your app](./cloud.md).