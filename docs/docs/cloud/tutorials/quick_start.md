# Quick Start
This quick start guide will cover how to develop an application for LangGraph Cloud, run it locally in Docker, and call the APIs to invoke a graph.

Alternatively, clone or fork the [`langgraph/example`](https://github.com/langchain-ai/langgraph-example) GitHub repository and follow the instructions in the `README`.

## Develop
1. Create a new application with the following directory and files:

        <my-app>/
        |-- agent.py            # code for your LangGraph agent
        |-- requirements.txt    # Python packages required for your graph
        |-- langgraph.json      # configuration file for LangGraph
        |-- .env                # environment files with API keys

2. The `agent.py` file should contain the following Python code for defining a simple graph: 

    ```python
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, MessageGraph

    model = ChatOpenAI(temperature=0)

    graph_workflow = MessageGraph()

    graph_workflow.add_node("agent", model)
    graph_workflow.add_edge("agent", END)
    graph_workflow.set_entry_point("agent")

    graph = graph_workflow.compile()
    ```

3. The `requirements.txt` file should contain the following dependencies:

        langgraph
        langchain_openai

4. The `langgraph.json` file should contain the following JSON object:

    ```json
    {
        "dependencies": ["."],
        "graphs": {
            "agent": "./agent.py:graph"
        },
        "env": ".env"
    }
    ```

    Learn more about the LangGraph CLI configuration file [here](../reference/cli.md#configuration-file).

5. The `.env` file should contain the environment variables:

        OPENAI_API_KEY=<add your key here>
        LANGGRAPH_AUTH_TYPE=noop

    !!! warning "Disable Authentication"
        When testing locally, set `LANGGRAPH_AUTH_TYPE` to `noop` to disable authentication.

## Run Locally
1. Install the [LangGraph CLI](../reference/cli.md#installation).

2. Run the following command to start the API server in Docker:

        langgraph up -c langgraph.json

3. The API server is now running at `http://localhost:8123`. Navigate to [`http://localhost:8123/docs`](http://localhost:8123/docs) to view the API docs.

## Deploy to Cloud

Follow [these instructions](./deployment/managed.md) to deploy to LangGraph Cloud.
