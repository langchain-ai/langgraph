# Run a local server

This guide shows you how to run a LangGraph application locally.

## Prerequisites

Before you begin, ensure you have the following:

- An API key for [LangSmith](https://smith.langchain.com/settings) - free to sign up

## 1. Install the LangGraph CLI

:::python

```shell
# Python >= 3.11 is required.

pip install --upgrade "langgraph-cli[inmem]"
```

:::

:::js

```shell
npx @langchain/langgraph-cli
```

:::

## 2. Create a LangGraph app 🌱

:::python
Create a new app from the [`new-langgraph-project-python` template](https://github.com/langchain-ai/new-langgraph-project). This template demonstrates a single-node application you can extend with your own logic.

```shell
langgraph new path/to/your/app --template new-langgraph-project-python
```

!!! tip "Additional templates"

    If you use `langgraph new` without specifying a template, you will be presented with an interactive menu that will allow you to choose from a list of available templates.

:::

:::js
Create a new app from the [`new-langgraph-project-js` template](https://github.com/langchain-ai/new-langgraphjs-project). This template demonstrates a single-node application you can extend with your own logic.

```shell
npm create langgraph
```

:::

## 3. Install dependencies

In the root of your new LangGraph app, install the dependencies in `edit` mode so your local changes are used by the server:

:::python

```shell
cd path/to/your/app
pip install -e .
```

:::

:::js

```shell
cd path/to/your/app
npm install
```

:::

## 4. Create a `.env` file

You will find a `.env.example` in the root of your new LangGraph app. Create a `.env` file in the root of your new LangGraph app and copy the contents of the `.env.example` file into it, filling in the necessary API keys:

```bash
LANGSMITH_API_KEY=lsv2...
```

## 5. Launch LangGraph Server 🚀

Start the LangGraph API server locally:

:::python

```shell
langgraph dev
```

:::

:::js

```shell
npx @langchain/langgraph-cli dev
```

:::

Sample output:

```
>    Ready!
>
>    - API: [http://localhost:2024](http://localhost:2024/)
>
>    - Docs: http://localhost:2024/docs
>
>    - LangGraph Studio Web UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

The `langgraph dev` command starts LangGraph Server in an in-memory mode. This mode is suitable for development and testing purposes. For production use, deploy LangGraph Server with access to a persistent storage backend. For more information, see [Deployment options](../../concepts/deployment_options.md).

## 6. Test your application in LangGraph Studio

[LangGraph Studio](../../concepts/langgraph_studio.md) is a specialized UI that you can connect to LangGraph API server to visualize, interact with, and debug your application locally. Test your graph in LangGraph Studio by visiting the URL provided in the output of the `langgraph dev` command:

```
>    - LangGraph Studio Web UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

For a LangGraph Server running on a custom host/port, update the baseURL parameter.

??? info "Safari compatibility"

    Use the `--tunnel` flag with your command to create a secure tunnel, as Safari has limitations when connecting to localhost servers:

    ```shell
    langgraph dev --tunnel
    ```

## 7. Test the API

:::python
=== "Python SDK (async)"

    1. Install the LangGraph Python SDK:

        ```shell
        pip install langgraph-sdk
        ```

    1. Send a message to the assistant (threadless run):

        ```python
        from langgraph_sdk import get_client
        import asyncio

        client = get_client(url="http://localhost:2024")

        async def main():
            async for chunk in client.runs.stream(
                None,  # Threadless run
                "agent", # Name of assistant. Defined in langgraph.json.
                input={
                "messages": [{
                    "role": "human",
                    "content": "What is LangGraph?",
                    }],
                },
            ):
                print(f"Receiving new event of type: {chunk.event}...")
                print(chunk.data)
                print("\n\n")

        asyncio.run(main())
        ```

=== "Python SDK (sync)"

    1. Install the LangGraph Python SDK:

        ```shell
        pip install langgraph-sdk
        ```

    1. Send a message to the assistant (threadless run):

        ```python
        from langgraph_sdk import get_sync_client

        client = get_sync_client(url="http://localhost:2024")

        for chunk in client.runs.stream(
            None,  # Threadless run
            "agent", # Name of assistant. Defined in langgraph.json.
            input={
                "messages": [{
                    "role": "human",
                    "content": "What is LangGraph?",
                }],
            },
            stream_mode="messages-tuple",
        ):
            print(f"Receiving new event of type: {chunk.event}...")
            print(chunk.data)
            print("\n\n")
        ```

=== "Rest API"

    ```bash
    curl -s --request POST \
        --url "http://localhost:2024/runs/stream" \
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
            \"stream_mode\": \"messages-tuple\"
        }"
    ```

:::

:::js
=== "Javascript SDK"

    1. Install the LangGraph JS SDK:

        ```shell
        npm install @langchain/langgraph-sdk
        ```

    1. Send a message to the assistant (threadless run):

        ```js
        const { Client } = await import("@langchain/langgraph-sdk");

        // only set the apiUrl if you changed the default port when calling langgraph dev
        const client = new Client({ apiUrl: "http://localhost:2024"});

        const streamResponse = client.runs.stream(
            null, // Threadless run
            "agent", // Assistant ID
            {
                input: {
                    "messages": [
                        { "role": "user", "content": "What is LangGraph?"}
                    ]
                },
                streamMode: "messages-tuple",
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
        --url "http://localhost:2024/runs/stream" \
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
            \"stream_mode\": \"messages-tuple\"
        }"
    ```

:::

## Next steps

Now that you have a LangGraph app running locally, take your journey further by exploring deployment and advanced features:

- [Deployment quickstart](../../cloud/quick_start.md): Deploy your LangGraph app using LangGraph Platform.
- [LangGraph Platform overview](../../concepts/langgraph_platform.md): Learn about foundational LangGraph Platform concepts.
- [LangGraph Server API Reference](../../cloud/reference/api/api_ref.html): Explore the LangGraph Server API documentation.

:::python

- [Python SDK Reference](../../cloud/reference/sdk/python_sdk_ref.md): Explore the Python SDK API Reference.
:::

:::js

- [JS/TS SDK Reference](../../cloud/reference/sdk/js_ts_sdk_ref.md): Explore the JS/TS SDK API Reference.
:::
