# Stream outputs

## Streaming API

[LangGraph SDK](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/) allows you to stream outputs from the LangGraph API server.

Basic usage example:

=== "Python"

    ```python
    from langgraph_sdk import get_client
    client = get_client(url=<DEPLOYMENT_URL>)

    # Using the graph deployed with the name "agent"
    assistant_id = "agent"

    # create a thread
    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    # create a streaming run
    # highlight-next-line
    async for chunk in client.runs.stream(
        thread_id,
        assistant_id,
        input=inputs,
        stream_mode="updates"
    ):
        print(chunk.data)
    ```

=== "JavaScript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";
    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });

    // Using the graph deployed with the name "agent"
    const assistantID = "agent";

    // create a thread
    const thread = await client.threads.create();
    const threadID = thread["thread_id"];

    // create a streaming run
    // highlight-next-line
    const streamResponse = client.runs.stream(
      threadID,
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

=== "cURL"

    Create a thread:

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads \
    --header 'Content-Type: application/json' \
    --data '{}'
    ```

    Create a streaming run:

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
    --header 'Content-Type: application/json' \
    --data "{
      \"assistant_id\": \"agent\",
      \"input\": <inputs>,
      \"stream_mode\": [
        \"updates\"
      ]
    }"
    ```

??? example "Extended example: streaming updates"

    This is an example graph you can run in the LangGraph API server.
    See [LangGraph Platform quickstart](../quick_start.md) for more details.

    ```python
    # graph.py
    from typing import TypedDict
    from langgraph.graph import StateGraph, START, END

    class State(TypedDict):
        topic: str
        joke: str

    def refine_topic(state: State):
        return {"topic": state["topic"] + " and cats"}

    def generate_joke(state: State):
        return {"joke": f"This is a joke about {state['topic']}"}

    graph = (
        StateGraph(State)
        .add_node(refine_topic)
        .add_node(generate_joke)
        .add_edge(START, "refine_topic")
        .add_edge("refine_topic", "generate_joke")
        .add_edge("generate_joke", END)
        .compile()
    )
    ```

    Once you have a running LangGraph API server, you can interact with it using
    [LangGraph SDK](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/)

    === "Python"

        ```python
        from langgraph_sdk import get_client
        client = get_client(url=<DEPLOYMENT_URL>)

        # Using the graph deployed with the name "agent"
        assistant_id = "agent"

        # create a thread
        thread = await client.threads.create()
        thread_id = thread["thread_id"]

        # create a streaming run
        # highlight-next-line
        async for chunk in client.runs.stream(  # (1)!
            thread_id,
            assistant_id,
            input={"topic": "ice cream"},
            # highlight-next-line
            stream_mode="updates"  # (2)!
        ):
            print(chunk.data)
        ```

        1. The `client.runs.stream()` method returns an iterator that yields streamed outputs.
        2. Set `stream_mode="updates"` to stream only the updates to the graph state after each node. Other stream modes are also available. See [supported stream modes](#supported-stream-modes) for details.

    === "JavaScript"

        ```js
        import { Client } from "@langchain/langgraph-sdk";
        const client = new Client({ apiUrl: <DEPLOYMENT_URL> });

        // Using the graph deployed with the name "agent"
        const assistantID = "agent";

        // create a thread
        const thread = await client.threads.create();
        const threadID = thread["thread_id"];

        // create a streaming run
        // highlight-next-line
        const streamResponse = client.runs.stream(  // (1)!
          threadID,
          assistantID,
          {
            input: { topic: "ice cream" },
            // highlight-next-line
            streamMode: "updates"  // (2)!
          }
        );
        for await (const chunk of streamResponse) {
          console.log(chunk.data);
        }
        ```

        1. The `client.runs.stream()` method returns an iterator that yields streamed outputs.
        2. Set `streamMode: "updates"` to stream only the updates to the graph state after each node. Other stream modes are also available. See [supported stream modes](#supported-stream-modes) for details.

    === "cURL"

        Create a thread:

        ```bash
        curl --request POST \
        --url <DEPLOYMENT_URL>/threads \
        --header 'Content-Type: application/json' \
        --data '{}'
        ```

        Create a streaming run:

        ```bash
        curl --request POST \
        --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
        --header 'Content-Type: application/json' \
        --data "{
          \"assistant_id\": \"agent\",
          \"input\": {\"topic\": \"ice cream\"},
          \"stream_mode\": [
            \"updates\"
          ]
        }"
        ```

    ```output
    {'run_id': '1f02c2b3-3cef-68de-b720-eec2a4a8e920', 'attempt': 1}
    {'refine_topic': {'topic': 'ice cream and cats'}}
    {'generate_joke': {'joke': 'This is a joke about ice cream and cats'}}
    ```


### Supported stream modes

| Mode                 | Description                                                                                                                                                    | LangGraph Library Method                                                |
|----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| [`values`](#stream-graph-state)         | Stream the full graph state after each [super-step](../../concepts/low_level.md#graphs). | `.stream()` / `.astream()` with [`stream_mode="values"`](../../how-tos/streaming.md#stream-graph-state)                  |
| [`updates`](#stream-graph-state)        | Streams the updates to the state after each step of the graph. If multiple updates are made in the same step (e.g., multiple nodes are run), those updates are streamed separately.                                                              | `.stream()` / `.astream()` with [`stream_mode="updates"`](../../how-tos/streaming.md#stream-graph-state)                 |
| [`messages-tuple`](#messages) | Streams LLM tokens and metadata for the graph node where the LLM is invoked (useful for chat apps).                                   | `.stream()` / `.astream()` with [`stream_mode="messages"`](../../how-tos/streaming.md#messages)                |
| [`debug`](#debug)          | Streams as much information as possible throughout the execution of the graph.                                                                       | `.stream()` / `.astream()` with [`stream_mode="debug"`](../../how-tos/streaming.md#stream-graph-state)                   |
| [`custom`](#stream-custom-data)         | Streams custom data from inside your graph                                                                                                                                             | `.stream()` / `.astream()` with [`stream_mode="custom"`](../../how-tos/streaming.md#stream-custom-data)                  |
| [`events`](#stream-events)         | Stream all events (including the state of the graph); mainly useful when migrating large LCEL apps.                       | `.astream_events()`

### Stream multiple modes

You can pass a list as the `stream_mode` parameter to stream multiple modes at once.

The streamed outputs will be tuples of `(mode, chunk)` where `mode` is the name of the stream mode and `chunk` is the data streamed by that mode.

=== "Python"

    ```python
    async for chunk in client.runs.stream(
        thread_id,
        assistant_id,
        input=inputs,
        stream_mode=["updates", "custom"]
    ):
        print(chunk)
    ```

=== "JavaScript"

    ```js
    const streamResponse = client.runs.stream(
      threadID,
      assistantID,
      {
        input,
        streamMode: ["updates", "custom"]
      }
    );
    for await (const chunk of streamResponse) {
      console.log(chunk);
    }
    ```

=== "cURL"

    ```bash
    curl --request POST \
     --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
     --header 'Content-Type: application/json' \
     --data "{
       \"assistant_id\": \"agent\",
       \"input\": <inputs>,
       \"stream_mode\": [
         \"updates\"
         \"custom\"
       ]
     }"
    ```

## Stream graph state

Use the stream modes `updates` and `values` to stream the state of the graph as it executes.

* `updates` streams the **updates** to the state after each step of the graph.
* `values` streams the **full value** of the state after each step of the graph.

??? example "Example graph"

    ```python
    from typing import TypedDict
    from langgraph.graph import StateGraph, START, END

    class State(TypedDict):
      topic: str
      joke: str

    def refine_topic(state: State):
        return {"topic": state["topic"] + " and cats"}

    def generate_joke(state: State):
        return {"joke": f"This is a joke about {state['topic']}"}

    graph = (
      StateGraph(State)
      .add_node(refine_topic)
      .add_node(generate_joke)
      .add_edge(START, "refine_topic")
      .add_edge("refine_topic", "generate_joke")
      .add_edge("generate_joke", END)
      .compile()
    )
    ```

!!! note "Stateful runs"

    Examples below assume that you want to **persist the outputs** of a streaming run in the [checkpointer](../../concepts/persistence.md) DB and have created a thread. To create a thread:

    === "Python"

        ```python
        from langgraph_sdk import get_client
        client = get_client(url=<DEPLOYMENT_URL>)

        # Using the graph deployed with the name "agent"
        assistant_id = "agent"
        # create a thread
        thread = await client.threads.create()
        thread_id = thread["thread_id"]
        ```

    === "JavaScript"

        ```js
        import { Client } from "@langchain/langgraph-sdk";
        const client = new Client({ apiUrl: <DEPLOYMENT_URL> });

        // Using the graph deployed with the name "agent"
        const assistantID = "agent";
        // create a thread
        const thread = await client.threads.create();
        const threadID = thread["thread_id"]
        ```

    === "cURL"

        ```bash
        curl --request POST \
        --url <DEPLOYMENT_URL>/threads \
        --header 'Content-Type: application/json' \
        --data '{}'
        ```

    If you don't need to persist the outputs of a run, you can pass `None` instead of `thread_id` when streaming.

=== "updates"

    Use this to stream only the **state updates** returned by the nodes after each step. The streamed outputs include the name of the node as well as the update.

    === "Python"

        ```python
        async for chunk in client.runs.stream(
            thread_id,
            assistant_id,
            input={"topic": "ice cream"},
            # highlight-next-line
            stream_mode="updates"
        ):
            print(chunk.data)
        ```

    === "JavaScript"

        ```js
        const streamResponse = client.runs.stream(
          threadID,
          assistantID,
          {
            input: { topic: "ice cream" },
            // highlight-next-line
            streamMode: "updates"
          }
        );
        for await (const chunk of streamResponse) {
          console.log(chunk.data);
        }
        ```

    === "cURL"

        ```bash
        curl --request POST \
        --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
        --header 'Content-Type: application/json' \
        --data "{
          \"assistant_id\": \"agent\",
          \"input\": {\"topic\": \"ice cream\"},
          \"stream_mode\": \"updates\"
        }"
        ```

===  "values"

    Use this to stream the **full state** of the graph after each step.

    === "Python"

        ```python
        async for chunk in client.runs.stream(
            thread_id,
            assistant_id,
            input={"topic": "ice cream"},
            # highlight-next-line
            stream_mode="values"
        ):
            print(chunk.data)
        ```

    === "JavaScript"

        ```js
        const streamResponse = client.runs.stream(
          threadID,
          assistantID,
          {
            input: { topic: "ice cream" },
            // highlight-next-line
            streamMode: "values"
          }
        );
        for await (const chunk of streamResponse) {
          console.log(chunk.data);
        }
        ```

    === "cURL"

        ```bash
        curl --request POST \
        --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
        --header 'Content-Type: application/json' \
        --data "{
          \"assistant_id\": \"agent\",
          \"input\": {\"topic\": \"ice cream\"},
          \"stream_mode\": \"values\"
        }"
        ```


## Subgraphs

To include outputs from [subgraphs](../../concepts/subgraphs.md) in the streamed outputs, you can set `subgraphs=True` in the `.stream()` method of the parent graph. This will stream outputs from both the parent graph and any subgraphs.

```python
for chunk in client.runs.stream(
    thread_id,
    assistant_id,
    input={"foo": "foo"},
    # highlight-next-line
    stream_subgraphs=True, # (1)!
    stream_mode="updates",
):
    print(chunk)
```

1. Set `stream_subgraphs=True` to stream outputs from subgraphs.

??? example "Extended example: streaming from subgraphs"

    This is an example graph you can run in the LangGraph API server.
    See [LangGraph Platform quickstart](../quick_start.md) for more details.

    ```python
    # graph.py
    from langgraph.graph import START, StateGraph
    from typing import TypedDict

    # Define subgraph
    class SubgraphState(TypedDict):
        foo: str  # note that this key is shared with the parent graph state
        bar: str

    def subgraph_node_1(state: SubgraphState):
        return {"bar": "bar"}

    def subgraph_node_2(state: SubgraphState):
        return {"foo": state["foo"] + state["bar"]}

    subgraph_builder = StateGraph(SubgraphState)
    subgraph_builder.add_node(subgraph_node_1)
    subgraph_builder.add_node(subgraph_node_2)
    subgraph_builder.add_edge(START, "subgraph_node_1")
    subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
    subgraph = subgraph_builder.compile()

    # Define parent graph
    class ParentState(TypedDict):
        foo: str

    def node_1(state: ParentState):
        return {"foo": "hi! " + state["foo"]}

    builder = StateGraph(ParentState)
    builder.add_node("node_1", node_1)
    builder.add_node("node_2", subgraph)
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", "node_2")
    graph = builder.compile()
    ```

    Once you have a running LangGraph API server, you can interact with it using
    [LangGraph SDK](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/)

    === "Python"

        ```python
        from langgraph_sdk import get_client
        client = get_client(url=<DEPLOYMENT_URL>)

        # Using the graph deployed with the name "agent"
        assistant_id = "agent"

        # create a thread
        thread = await client.threads.create()
        thread_id = thread["thread_id"]
    
        async for chunk in client.runs.stream(
            thread_id,
            assistant_id,
            input={"foo": "foo"},
            # highlight-next-line
            stream_subgraphs=True, # (1)!
            stream_mode="updates",
        ):
            print(chunk)
        ```
        
        1. Set `stream_subgraphs=True` to stream outputs from subgraphs.

    === "JavaScript"

        ```js
        import { Client } from "@langchain/langgraph-sdk";
        const client = new Client({ apiUrl: <DEPLOYMENT_URL> });

        // Using the graph deployed with the name "agent"
        const assistantID = "agent";

        // create a thread
        const thread = await client.threads.create();
        const threadID = thread["thread_id"];

        // create a streaming run
        const streamResponse = client.runs.stream(
          threadID,
          assistantID,
          {
            input: { foo: "foo" },
            // highlight-next-line
            streamSubgraphs: true,  // (1)!
            streamMode: "updates"
          }
        );
        for await (const chunk of streamResponse) {
          console.log(chunk);
        }
        ```

        1. Set `streamSubgraphs: true` to stream outputs from subgraphs.

    === "cURL"

        Create a thread:

        ```bash
        curl --request POST \
        --url <DEPLOYMENT_URL>/threads \
        --header 'Content-Type: application/json' \
        --data '{}'
        ```

        Create a streaming run:

        ```bash
        curl --request POST \
        --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
        --header 'Content-Type: application/json' \
        --data "{
          \"assistant_id\": \"agent\",
          \"input\": {\"foo\": \"foo\"},
          \"stream_subgraphs\": true,
          \"stream_mode\": [
            \"updates\"
          ]
        }"
        ```

    **Note** that we are receiving not just the node updates, but we also the namespaces which tell us what graph (or subgraph) we are streaming from.

## Debugging {#debug}

Use the `debug` streaming mode to stream as much information as possible throughout the execution of the graph. The streamed outputs include the name of the node as well as the full state.

=== "Python"

    ```python
    async for chunk in client.runs.stream(
        thread_id,
        assistant_id,
        input={"topic": "ice cream"},
        # highlight-next-line
        stream_mode="debug"
    ):
        print(chunk.data)
    ```

=== "JavaScript"

    ```js
    const streamResponse = client.runs.stream(
      threadID,
      assistantID,
      {
        input: { topic: "ice cream" },
        // highlight-next-line
        streamMode: "debug"
      }
    );
    for await (const chunk of streamResponse) {
      console.log(chunk.data);
    }
    ```

=== "cURL"

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
    --header 'Content-Type: application/json' \
    --data "{
      \"assistant_id\": \"agent\",
      \"input\": {\"topic\": \"ice cream\"},
      \"stream_mode\": \"debug\"
    }"
    ```

## LLM tokens {#messages}

Use the `messages-tuple` streaming mode to stream Large Language Model (LLM) outputs **token by token** from any part of your graph, including nodes, tools, subgraphs, or tasks.

The streamed output from [`messages-tuple` mode](#supported-stream-modes) is a tuple `(message_chunk, metadata)` where:

- `message_chunk`: the token or message segment from the LLM.
- `metadata`: a dictionary containing details about the graph node and LLM invocation.

> If your LLM is not available as a LangChain integration, you can stream its outputs using `custom` mode instead. See [use with any LLM](#use-with-any-llm) for details.
 
!!! warning "Manual config required for async in Python < 3.11"

    When using Python < 3.11 with async code in your graph, you must explicitly pass `RunnableConfig` to `ainvoke()` to enable proper streaming. See [Async with Python < 3.11](#async) for details or upgrade to Python 3.11+.

??? example "Example graph"

    ```python
    from dataclasses import dataclass

    from langchain.chat_models import init_chat_model
    from langgraph.graph import StateGraph, START

    @dataclass
    class MyState:
        topic: str
        joke: str = ""

    llm = init_chat_model(model="openai:gpt-4o-mini")

    def call_model(state: MyState):
        """Call the LLM to generate a joke about a topic"""
        # highlight-next-line
        llm_response = llm.invoke( # (1)!
            [
                {"role": "user", "content": f"Generate a joke about {state.topic}"}
            ]
        )
        return {"joke": llm_response.content}

    graph = (
        StateGraph(MyState)
        .add_node(call_model)
        .add_edge(START, "call_model")
        .compile()
    )
    ```

    1. Note that the message events are emitted even when the LLM is run using `.invoke` rather than `.stream`.

=== "Python"

    ```python
    async for chunk in client.runs.stream(
        thread_id,
        assistant_id,
        input={"topic": "ice cream"},
        # highlight-next-line
        stream_mode="messages-tuple",
    ):
        if chunk.event != "messages":
            continue

        message_chunk, metadata = chunk.data  # (1)!
        if message_chunk["content"]:
            print(message_chunk["content"], end="|", flush=True)
    ```

    1. The "messages-tuple" stream mode returns an iterator of tuples `(message_chunk, metadata)` where `message_chunk` is the token streamed by the LLM and `metadata` is a dictionary with information about the graph node where the LLM was called and other information.

=== "JavaScript"

    ```js
    const streamResponse = client.runs.stream(
      threadID,
      assistantID,
      {
        input: { topic: "ice cream" },
        // highlight-next-line
        streamMode: "messages-tuple"
      }
    );
    for await (const chunk of streamResponse) {
      if (chunk.event !== "messages") {
        continue;
      }
      console.log(chunk.data[0]["content"]);  // (1)!
    }
    ```

    1. The "messages-tuple" stream mode returns an iterator of tuples `(message_chunk, metadata)` where `message_chunk` is the token streamed by the LLM and `metadata` is a dictionary with information about the graph node where the LLM was called and other information.

=== "cURL"

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
    --header 'Content-Type: application/json' \
    --data "{
      \"assistant_id\": \"agent\",
      \"input\": {\"topic\": \"ice cream\"},
      \"stream_mode\": \"messages-tuple\"
    }"
    ```

### Filter LLM tokens

* To filter the streamed tokens by LLM invocation, you can [associate `tags` with LLM invocations](../../how-tos/streaming.md#filter-by-llm-invocation).
* To stream tokens only from specific nodes, use `stream_mode="messages"` and [filter the outputs by the `langgraph_node` field](../../how-tos/streaming.md#filter-by-node) in the streamed metadata.

## Stream custom data

To send **custom user-defined data** from inside a LangGraph node or tool, follow these steps:

1. Use `get_stream_writer()` to access the stream writer and emit custom data.
2. Set `stream_mode="custom"` when calling `.stream()` or `.astream()` to get the custom data in the stream. You can combine multiple modes (e.g., `["updates", "custom"]`), but at least one must be `"custom"`.

!!! warning "No `get_stream_writer()` in async for Python < 3.11"

    In async code running on Python < 3.11, `get_stream_writer()` will not work.  
    Instead, add a `writer` parameter to your node or tool and pass it manually.  
    See [Async with Python < 3.11](#async) for usage examples.

??? example "Example graph"

      ```python
      from typing import TypedDict
      from langgraph.config import get_stream_writer
      from langgraph.graph import StateGraph, START

      class State(TypedDict):
          query: str
          answer: str

      def node(state: State):
          writer = get_stream_writer()  # (1)!
          writer({"custom_key": "Generating custom data inside node"}) # (2)!
          return {"answer": "some data"}

      graph = (
          StateGraph(State)
          .add_node(node)
          .add_edge(START, "node")
          .compile()
      )
      ```

      1. Get the stream writer to send custom data.
      2. Emit a custom key-value pair (e.g., progress update).

=== "Python"

    ```python
    async for chunk in client.runs.stream(
        thread_id,
        assistant_id,
        input={"query": "example"},
        # highlight-next-line
        stream_mode="custom"
    ):
        print(chunk.data)
    ```

=== "JavaScript"

    ```js
    const streamResponse = client.runs.stream(
      threadID,
      assistantID,
      {
        input: { query: "example" },
        // highlight-next-line
        streamMode: "custom"
      }
    );
    for await (const chunk of streamResponse) {
      console.log(chunk.data);
    }
    ```

=== "cURL"

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
    --header 'Content-Type: application/json' \
    --data "{
      \"assistant_id\": \"agent\",
      \"input\": {\"query\": \"example\"},
      \"stream_mode\": \"custom\"
    }"
    ```

See [this guide](../../how-tos/streaming.md#stream-custom-data) for more examples.

## Stream events

To stream all events, including the state of the graph:

=== "Python"

    ```python
    async for chunk in client.runs.stream(
        thread_id,
        assistant_id,
        input={"topic": "ice cream"},
        # highlight-next-line
        stream_mode="events"
    ):
        print(chunk.data)
    ```

=== "JavaScript"

    ```js
    const streamResponse = client.runs.stream(
      threadID,
      assistantID,
      {
        input: { topic: "ice cream" },
        // highlight-next-line
        streamMode: "events"
      }
    );
    for await (const chunk of streamResponse) {
      console.log(chunk.data);
    }
    ```

=== "cURL"

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
    --header 'Content-Type: application/json' \
    --data "{
      \"assistant_id\": \"agent\",
      \"input\": {\"topic\": \"ice cream\"},
      \"stream_mode\": \"events\"
    }"
    ```

## Use with any LLM

You can use `stream_mode="custom"` to stream data from **any LLM API** — even if that API does **not** implement the LangChain chat model interface. See [this guide](../../how-tos/streaming.md#use-with-any-llm) for more information.

## Disable streaming for specific chat models

If your application mixes models that support streaming with those that do not, you may need to explicitly disable streaming for 
models that do not support it. See [this guide](../../how-tos/streaming.md#disable-streaming-for-specific-chat-models) on how to disable streaming.

## Async with Python < 3.11 { #async }

In Python versions < 3.11, [asyncio tasks](https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task) do not support the `context` parameter.  
This limits LangGraph ability to automatically propagate context, and affects LangGraph’s streaming mechanisms in two key ways:

1. You **must** explicitly pass [`RunnableConfig`](https://python.langchain.com/docs/concepts/runnables/#runnableconfig) into async LLM calls (e.g., `ainvoke()`), as callbacks are not automatically propagated.
2. You **cannot** use `get_stream_writer()` in async nodes or tools — you must pass a `writer` argument directly.

See [this guide](../../how-tos/streaming.md#async) for more information and examples.