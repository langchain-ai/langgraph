# How to test a LangGraph app locally

This guide assumes you have a LangGraph app correctly set up with a proper configuration file and a corresponding compiled graph, and that you have a proper LangChain API key.

## Setup

Install the proper packages:

```shell
pip install langgraph-cli
```

## Start the API server

Once you have downloaded the CLI, you can run the following command to start the API server for local testing:

```shell
langgraph up
```

This will start up the LangGraph API server locally. If this runs successfully, you should see something like:

```shell
Ready!
- API: http://localhost:8123
2024-06-26 19:20:41,056:INFO:uvicorn.access 127.0.0.1:44138 - "GET /ok HTTP/1.1" 200
```

### Interact with the server

We can now interact with the API server using the LangGraph SDK. First, we need to start our client, select our assistant (in this case a graph we called "agent", make sure to select the proper assistant you wish to test).

=== "Python"

    ```python
    from langgraph_sdk import get_client

    # only pass the url argument to get_client() if you changed the default port when calling langgraph up
    client = get_client()
    assistant_id = "agent"
    thread = await client.threads.create()
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    // only set the apiUrl if you changed the default port when calling langgraph up
    const client = new Client();
    const assistantId = "agent"
    const thread = await client.threads.create();
    ```

Now we can invoke our graph to ensure it is working. Make sure to change the input to match the proper schema for your graph. 

=== "Python"

    ```python
    input = {"messages": [{"role": "human", "content": "what's the weather in sf"}]}
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=input,
        stream_mode="updates",
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")
    ```
=== "Javascript"

    ```js
    const input = { "messages": [{ "role": "human", "content": "what's the weather in sf"}] }

    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantId,
      {
        input: input,
        streamMode: "updates",
      }
    );
    for await (const chunk of streamResponse) {
      console.log(`Receiving new event of type: ${chunk.event}...`);
      console.log(chunk.data);
      console.log("\n\n");
    }
    ```

If your graph works correctly, you should see your graph output displayed in the console. Of course, there are many more ways you might need to test your graph, for a full list of commands you can send with the SDK, see the [Python](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/) and [JS/TS](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/js_ts_sdk_ref/) references.