# Streaming

Streaming is critical for making LLM applications feel responsive to end users.  
When creating a streaming run, the **streaming mode** determines what kinds of data are streamed back to the API client.

## Supported streaming modes

LangGraph Platform supports the following streaming modes:

| Mode                 | Description                                                                                                                                                    | LangGraph Library Method                                                |
|----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| **`values`**         | Stream the full graph state after each [super-step](https://langchain-ai.github.io/langgraph/concepts/low_level/#graphs). [Guide](../how-tos/streaming.md#stream-graph-state) | `.stream()` / `.astream()` with `stream_mode="values"`                  |
| **`updates`**        | Stream only the updates to the graph state after each node. [Guide](../how-tos/streaming.md#stream-graph-state)                                                              | `.stream()` / `.astream()` with `stream_mode="updates"`                 |
| **`messages-tuple`** | Stream LLM tokens for any messages generated inside the graph (useful for chat apps). [Guide](../how-tos/streaming.md#messages)                                   | `.stream()` / `.astream()` with `stream_mode="messages"`                |
| **`debug`**          | Stream debug information throughout graph execution. [Guide](../how-tos/streaming.md#debug)                                                                       | `.stream()` / `.astream()` with `stream_mode="debug"`                   |
| **`custom`**         | Stream custom data. [Guide](../../how-tos/streaming.md#stream-custom-data)                                                                                                                                             | `.stream()` / `.astream()` with `stream_mode="custom"`                  |
| **`events`**         | Stream all events (including the state of the graph); mainly useful when migrating large LCEL apps. [Guide](../how-tos/streaming.md#stream-events)                       | `.astream_events()`                                                     |

✅ You can also **combine multiple modes** at the same time.  See the [how-to guide](../how-tos/streaming.md#stream-multiple-modes) for configuration details.

## Stateless runs

If you don't want to **persist the outputs** of a streaming run in the [checkpointer](../../concepts/persistence.md) DB, you can create a stateless run without creating a thread:

=== "Python"

    ```python
    from langgraph_sdk import get_client
    client = get_client(url=<DEPLOYMENT_URL>, api_key=<API_KEY>)

    async for chunk in client.runs.stream(
        # highlight-next-line
        None,  # (1)!
        assistant_id,
        input=inputs,
        stream_mode="updates"
    ):
        print(chunk.data)
    ```

    1. We are passing `None` instead of a `thread_id` UUID.

=== "JavaScript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";
    const client = new Client({ apiUrl: <DEPLOYMENT_URL>, apiKey: <API_KEY> });

    // create a streaming run
    // highlight-next-line
    const streamResponse = client.runs.stream(
      // highlight-next-line
      null,  // (1)!
      assistantID,
      {
        input,
        streamMode: "updates"
      }
    );
    for await (const chunk of streamResponse) {
      console.log(chunk.data);
    }
    ```

    1. We are passing `None` instead of a `thread_id` UUID.

=== "cURL"

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/runs/stream \
    --header 'Content-Type: application/json' \
    --header 'x-api-key: <API_KEY>'
    --data "{
      \"assistant_id\": \"agent\",
      \"input\": <inputs>,
      \"stream_mode\": \"updates\"
    }"
    ```

## Join and stream

LangGraph Platform allows you to join an active [background run](../how-tos/background_run.md) and stream outputs from it. To do so, you can use [LangGraph SDK's](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/) `client.runs.join_stream` method:

=== "Python"

    ```python
    from langgraph_sdk import get_client
    client = get_client(url=<DEPLOYMENT_URL>, api_key=<API_KEY>)

    # highlight-next-line
    async for chunk in client.runs.join_stream(
        thread_id,
        # highlight-next-line
        run_id,  # (1)!
    ):
        print(chunk)
    ```

    1. This is the `run_id` of an existing run you want to join.


=== "JavaScript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";
    const client = new Client({ apiUrl: <DEPLOYMENT_URL>, apiKey: <API_KEY> });

    // highlight-next-line
    const streamResponse = client.runs.joinStream(
      threadID,
      // highlight-next-line
      runId  // (1)!
    );
    for await (const chunk of streamResponse) {
      console.log(chunk);
    }
    ```

    1. This is the `run_id` of an existing run you want to join.

=== "cURL"

    ```bash
    curl --request GET \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/<RUN_ID>/stream \
    --header 'Content-Type: application/json' \
    --header 'x-api-key: <API_KEY>'
    ```

!!! warning "Outputs not buffered"

    When you use `.join_stream`, output is not buffered, so any output produced before joining will not be received.

## API Reference

For API usage and implementation, refer to the [API reference](../reference/api/api_ref.html#tag/thread-runs/POST/threads/{thread_id}/runs/stream). 

