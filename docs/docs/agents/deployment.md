# Deployment

To deploy your agent, you will need to first create a new LangGraph app.

## Create a LangGraph app

```bash
pip install -U "langgraph-cli[inmem]"
langgraph new path/to/your/app --template new-langgraph-project-python
```

This will create an empty LangGraph project. You can modify it by replacing the code in `src/agent/graph.py` with your agent code. For example:

```python
from langgraph.prebuilt import create_react_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

graph = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    prompt="You are a helpful assistant"
)
```

### Install dependencies

In the root of your new LangGraph app, install the dependencies in `edit` mode so your local changes are used by the server:

```shell
pip install -e .
```

### Create an `.env` file

You will find a `.env.example` in the root of your new LangGraph app. Create
a `.env` file in the root of your new LangGraph app and copy the contents of the `.env.example` file into it, filling in the necessary API keys:

```bash
LANGSMITH_API_KEY=lsv2...
ANTHROPIC_API_KEY=sk-
```

## Launch LangGraph server locally

```shell
langgraph dev
```

This will start up the LangGraph API server locally. If this runs successfully, you should see something like:

>    Ready!
> 
>    - API: [http://localhost:2024](http://localhost:2024/)
>     
>    - Docs: http://localhost:2024/docs
>     
>    - LangGraph Studio Web UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

See this [tutorial](https://langchain-ai.github.io/langgraph/tutorials/langgraph-platform/local-server/) to learn more about running LangGraph app locally.

## LangGraph Studio Web UI

LangGraph Studio Web is a specialized UI that you can connect to LangGraph API server to enable visualization, interaction, and debugging of your application locally. Test your graph in the LangGraph Studio Web UI by visiting the URL provided in the output of the `langgraph dev` command.

>    - LangGraph Studio Web UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

## Deployment

Now that you have a LangGraph app running locally, learn how you can [deploy it using LangGraph Cloud](https://langchain-ai.github.io/langgraph/cloud/quick_start/).