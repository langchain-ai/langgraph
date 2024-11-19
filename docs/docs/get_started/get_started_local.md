# Get started with LangGraph Platform

!!! note "Requirements"

    - [Docker](https://docs.docker.com/get-docker/)
    - [LangGraph CLI](https://langchain-ai.github.io/langgraph/cloud/reference/cli/) >= 0.1.54

This is a quick start guide to help you get a LangGraph app up and running locally.

## Install the LangGraph CLI

=== "pip" 
    ```bash
    pip install -U langgraph-cli
    ```
=== "Homebrew (macOS only)"
    ```bash
    brew install langgraph-cli
    ```

## Create a LangGraph App

Create a new app from the `react-agent` template. This template is a simple agent that can be flexibly extended to many tools.

=== "Python Server"

    ```shell
    langgraph new path/to/your/app --template react-agent-python 
    ```

=== "Node Server"

    ```shell
    langgraph new path/to/your/app --template react-agent-js
    ```

!!! tip "Templates"

    Use `langgraph new` to get an interactive menu of available templates.

## Create a `.env` file

You will find a `.env.example` in the root of your new LangGraph app. Create
a `.env` file in the root of your new LangGraph app and copy the contents of the `.env.example` file into it, filling in the necessary API keys:

```bash
LANGSMITH_API_KEY=lsv2...
ANTHROPIC_API_KEY=sk-
TAVILY_API_KEY=tvly-...
OPENAI_API_KEY=sk-...
```

<details><summary>Get API Keys</summary>
    <ul>
        <li> <b>LANGSMITH_API_KEY</b>: Go to the <a href="https://smith.langchain.com/settings">LangSmith Settings page</a>. Then clck <b>Create API Key</b>.
        </li>
        <li>
            <b>ANTHROPIC_API_KEY</b>: Get an API key from <a href="https://www.anthropic.com/api">Anthropic</a>.
        </li>
        <li>
            <b>OPENAI_API_KEY</b>: Get an API key from <a href="https://openai.com/">OpenAI</a>.
        </li>
        <li>
            <b>TAVILY_API_KEY</b>: Get an API key on the <a href="https://tavily.com/">Tavily website</a>.
        </li>
    </ul>
</details>

## Launch LangGraph Server

```shell
langgraph up
```

This will start up the LangGraph API server locally. If this runs successfully, you should see something like:

>    Ready!
> 
>    - API: [http://localhost:8123](http://localhost:8123/)
>     
>    - Docs: http://localhost:8123/docs
>     
>    - LangGraph Studio Web UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:8123

## LangGraph Studio Web UI

Test your graph in the LangGraph Studio Web UI by visiting the URL provided in the output of the `langgraph up` command.

>    - LangGraph Studio Web UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:8123

!!! warning "Safari Compatibility"
    
    Currently, LangGraph Studio Web does not support Safari when running a server locally.


## Use the API

=== "Python SDK (Async)"

    **Install the LangGraph Python SDK**

    ```shell
    pip install langgraph-sdk
    ```

    **Send a message to the assistant (threadless run)**

    ```python
    from langgraph_sdk import get_client

    client = get_client(url="http://localhost:8123")

    async for chunk in client.runs.stream(
        None,  # Threadless run
        "agent", # Name of assistant. Defined in langgraph.json.
        input={
            "messages": [{
                "role": "human",
                "content": "What is LangGraph?",
            }],
        },
        stream_mode="updates",
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")
    ```

=== "Python SDK (Sync)"

    **Install the LangGraph Python SDK**

    ```shell
    pip install langgraph-sdk
    ```

    **Send a message to the assistant (threadless run)**

    ```python
    from langgraph_sdk import get_sync_client

    client = get_sync_client(url="http://localhost:8123")

    for chunk in client.runs.stream(
        None,  # Threadless run
        "agent", # Name of assistant. Defined in langgraph.json.
        input={
            "messages": [{
                "role": "human",
                "content": "What is LangGraph?",
            }],
        },
        stream_mode="updates",
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")
    ```

=== "Javascript SDK"

    **Install the LangGraph JS SDK**

    ```shell
    npm install @langchain/langgraph-sdk
    ```

    **Send a message to the assistant (threadless run)**

    ```js
    const { Client } = await import("@langchain/langgraph-sdk");

    // only set the apiUrl if you changed the default port when calling langgraph up
    const client = new Client({ apiUrl: "http://localhost:8123"});

    const streamResponse = client.runs.stream(
        null, // Threadless run
        "agent", // Assistant ID
        {
            input: {
                "messages": [
                    { "role": "user", "content": "What is LangGraph?"}
                ]
            },
            streamMode: "messages",
        }
    );

    for await (const chunk of streamResponse) {
        console.log(`Receiving new event of type: ${chunk.event}...`);
        console.log(JSON.stringify(chunk.data));
        console.log("\n\n");
    }
    ```

=== "Rest API"

    ```bash
    curl -s --request POST \
        --url "http://localhost:8123/runs/stream" \
        --header 'Content-Type: application/json' \
        --data "{
            \"assistant_id\": \"agent\",
            \"input\": {
                \"messages\": [
                    {
                        \"role\": \"human\",
                        \"content\": \"What is LangGraph?\"
                    }
                ]
            },
            \"stream_mode\": \"updates\"
        }" 
    ```

!!! tip "Auth"

    If you're connecting to a remote server, you may need to provide a LangSmith
    API key in the `Authorization` header. Please see the API Reference for the clients
    for more information.

## Related

There are many more things you can do with LangGraph Platform. For more information, see:

* [LangGraph Server API](../../cloud/reference/api/api_ref.html)
* [Python SDK Reference](../../cloud/reference/sdk/python_sdk_ref/)
* [JS/TS SDK Reference](../../cloud/reference/sdk/js_ts_sdk_ref/)