# Use threads

In this guide, we will show how to create, view, and inspect [threads](../../concepts/persistence.md#threads).

## Create a thread

To run your graph and the state persisted, you must first create a thread.

### Empty thread

To create a new thread, use the [LangGraph SDK](../../concepts/sdk.md) `create` method. See the [Python](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/#langgraph_sdk.client.ThreadsClient.create) and [JS](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/js_ts_sdk_ref/#create_3) SDK reference docs for more information.

=== "Python"

    ```python
    from langgraph_sdk import get_client

    client = get_client(url=<DEPLOYMENT_URL>)
    thread = await client.threads.create()

    print(thread)
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    const thread = await client.threads.create();

    console.log(thread);
    ```

=== "CURL"

    ```bash
    curl --request POST \
        --url <DEPLOYMENT_URL>/threads \
        --header 'Content-Type: application/json' \
        --data '{}'
    ```

Output:

    {
      "thread_id": "123e4567-e89b-12d3-a456-426614174000",
      "created_at": "2025-05-12T14:04:08.268Z",
      "updated_at": "2025-05-12T14:04:08.268Z",
      "metadata": {},
      "status": "idle",
      "values": {}
    }

### Copy thread

Alternatively, if you already have a thread in your application whose state you wish to copy, you can use the `copy` method. This will create an independent thread whose history is identical to the original thread at the time of the operation. See the [Python](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/#langgraph_sdk.client.ThreadsClient.copy) and [JS](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/js_ts_sdk_ref/#copy) SDK reference docs for more information.

=== "Python"

    ```python
    copied_thread = await client.threads.copy(<THREAD_ID>)
    ```

=== "Javascript"

    ```js
    const copiedThread = await client.threads.copy(<THREAD_ID>);
    ```

=== "CURL"

    ```bash
    curl --request POST --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/copy \
    --header 'Content-Type: application/json'
    ```

### Prepopulated State

Finally, you can create a thread with an arbitrary pre-defined state by providing a list of `supersteps` into the `create` method. The `supersteps` describe a list of a sequence of state updates. For example:

=== "Python"

    ```python
    from langgraph_sdk import get_client

    client = get_client(url=<DEPLOYMENT_URL>)
    thread = await client.threads.create(
      graph_id="agent",
      supersteps=[
        {
          updates: [
            {
              values: {},
              as_node: '__input__',
            },
          ],
        },
        {
          updates: [
            {
              values: {
                messages: [
                  {
                    type: 'human',
                    content: 'hello',
                  },
                ],
              },
              as_node: '__start__',
            },
          ],
        },
        {
          updates: [
            {
              values: {
                messages: [
                  {
                    content: 'Hello! How can I assist you today?',
                    type: 'ai',
                  },
                ],
              },
              as_node: 'call_model',
            },
          ],
        },
      ])

    print(thread)
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    const thread = await client.threads.create({
        graphId: 'agent',
        supersteps: [
        {
          updates: [
            {
              values: {},
              asNode: '__input__',
            },
          ],
        },
        {
          updates: [
            {
              values: {
                messages: [
                  {
                    type: 'human',
                    content: 'hello',
                  },
                ],
              },
              asNode: '__start__',
            },
          ],
        },
        {
          updates: [
            {
              values: {
                messages: [
                  {
                    content: 'Hello! How can I assist you today?',
                    type: 'ai',
                  },
                ],
              },
              asNode: 'call_model',
            },
          ],
        },
      ],
    });

    console.log(thread);
    ```

=== "CURL"

    ```bash
    curl --request POST \
        --url <DEPLOYMENT_URL>/threads \
        --header 'Content-Type: application/json' \
        --data '{"metadata":{"graph_id":"agent"},"supersteps":[{"updates":[{"values":{},"as_node":"__input__"}]},{"updates":[{"values":{"messages":[{"type":"human","content":"hello"}]},"as_node":"__start__"}]},{"updates":[{"values":{"messages":[{"content":"Hello\u0021 How can I assist you today?","type":"ai"}]},"as_node":"call_model"}]}]}'
    ```

Output:

    {
        "thread_id": "f15d70a1-27d4-4793-a897-de5609920b7d",
        "created_at": "2025-05-12T15:37:08.935038+00:00",
        "updated_at": "2025-05-12T15:37:08.935046+00:00",
        "metadata": {"graph_id": "agent"},
        "status": "idle",
        "config": {},
        "values": {
            "messages": [
                {
                    "content": "hello",
                    "additional_kwargs": {},
                    "response_metadata": {},
                    "type": "human",
                    "name": null,
                    "id": "8701f3be-959c-4b7c-852f-c2160699b4ab",
                    "example": false
                },
                {
                    "content": "Hello! How can I assist you today?",
                    "additional_kwargs": {},
                    "response_metadata": {},
                    "type": "ai",
                    "name": null,
                    "id": "4d8ea561-7ca1-409a-99f7-6b67af3e1aa3",
                    "example": false,
                    "tool_calls": [],
                    "invalid_tool_calls": [],
                    "usage_metadata": null
                }
            ]
        }
    }

## List threads

### LangGraph SDK

To list threads, use the [LangGraph SDK](../../concepts/sdk.md) `search` method. This will list the threads in the application that match the provided filters. See the [Python](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/#langgraph_sdk.client.ThreadsClient.search) and [JS](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/js_ts_sdk_ref/#search_2) SDK reference docs for more information.

#### Filter by thread status

Use the `status` field to filter threads based on their status. Supported values are `idle`, `busy`, `interrupted`, and `error`. See [here](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/?h=thread+status#langgraph_sdk.auth.types.ThreadStatus) for information on each status. For example, to view `idle` threads:

=== "Python"

    ```python
    print(await client.threads.search(status="idle",limit=1))
    ```

=== "Javascript"

    ```js
    console.log(await client.threads.search({ status: "idle", limit: 1 }));
    ```

=== "CURL"

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads/search \
    --header 'Content-Type: application/json' \
    --data '{"status": "idle", "limit": 1}'
    ```

Output:

    [
      {
        'thread_id': 'cacf79bb-4248-4d01-aabc-938dbd60ed2c',
        'created_at': '2024-08-14T17:36:38.921660+00:00',
        'updated_at': '2024-08-14T17:36:38.921660+00:00',
        'metadata': {'graph_id': 'agent'},
        'status': 'idle',
        'config': {'configurable': {}}
      }
    ]

#### Filter by metadata

The `search` method allows you to filter on metadata:

=== "Python"

    ```python
    print((await client.threads.search(metadata={"graph_id":"agent"},limit=1)))
    ```

=== "Javascript"

    ```js
    console.log((await client.threads.search({ metadata: { "graph_id": "agent" }, limit: 1 })));
    ```

=== "CURL"

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads/search \
    --header 'Content-Type: application/json' \
    --data '{"metadata": {"graph_id":"agent"}, "limit": 1}'
    ```

Output:

    [
      {
        'thread_id': 'cacf79bb-4248-4d01-aabc-938dbd60ed2c',
        'created_at': '2024-08-14T17:36:38.921660+00:00',
        'updated_at': '2024-08-14T17:36:38.921660+00:00',
        'metadata': {'graph_id': 'agent'},
        'status': 'idle',
        'config': {'configurable': {}}
      }
    ]

#### Sorting

The SDK also supports sorting threads by `thread_id`, `status`, `created_at`, and `updated_at` using the `sort_by` and `sort_order` params.

### LangGraph Platform UI

You can also view threads in a deployment via the LangGraph Platform UI.

Inside your deployment, select the "Threads" tab. This will load a table of all of the threads in your deployment.

To filter by thread status, select a status in the top bar. To sort by a supported property, click on the arrow icon for the desired column.

## Inspect threads

### LangGraph SDK

#### Get Thread

To view a specific thread given its `thread_id`, use the `get` method:

=== "Python"

    ```python
    print((await client.threads.get(<THREAD_ID>)))
    ```

=== "Javascript"

    ```js
    console.log((await client.threads.get(<THREAD_ID>)));
    ```

=== "CURL"

    ```bash
    curl --request GET \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID> \
    --header 'Content-Type: application/json'
    ```

Output:

    {
      'thread_id': 'cacf79bb-4248-4d01-aabc-938dbd60ed2c',
      'created_at': '2024-08-14T17:36:38.921660+00:00',
      'updated_at': '2024-08-14T17:36:38.921660+00:00',
      'metadata': {'graph_id': 'agent'},
      'status': 'idle',
      'config': {'configurable': {}}
    }

#### Inspect Thread State

To view the current state of a given thread, use the `get_state` method:

=== "Python"

    ```python
    print((await client.threads.get_state(<THREAD_ID>)))
    ```

=== "Javascript"

    ```js
    console.log((await client.threads.getState(<THREAD_ID>)));
    ```

=== "CURL"

    ```bash
    curl --request GET \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/state \
    --header 'Content-Type: application/json'
    ```

Output:

    {
        "values": {
            "messages": [
                {
                    "content": "hello",
                    "additional_kwargs": {},
                    "response_metadata": {},
                    "type": "human",
                    "name": null,
                    "id": "8701f3be-959c-4b7c-852f-c2160699b4ab",
                    "example": false
                },
                {
                    "content": "Hello! How can I assist you today?",
                    "additional_kwargs": {},
                    "response_metadata": {},
                    "type": "ai",
                    "name": null,
                    "id": "4d8ea561-7ca1-409a-99f7-6b67af3e1aa3",
                    "example": false,
                    "tool_calls": [],
                    "invalid_tool_calls": [],
                    "usage_metadata": null
                }
            ]
        },
        "next": [],
        "tasks": [],
        "metadata": {
            "thread_id": "f15d70a1-27d4-4793-a897-de5609920b7d",
            "checkpoint_id": "1f02f46f-7308-616c-8000-1b158a9a6955",
            "graph_id": "agent_with_quite_a_long_name",
            "source": "update",
            "step": 1,
            "writes": {
                "call_model": {
                    "messages": [
                        {
                            "content": "Hello! How can I assist you today?",
                            "type": "ai"
                        }
                    ]
                }
            },
            "parents": {}
        },
        "created_at": "2025-05-12T15:37:09.008055+00:00",
        "checkpoint": {
            "checkpoint_id": "1f02f46f-733f-6b58-8001-ea90dcabb1bd",
            "thread_id": "f15d70a1-27d4-4793-a897-de5609920b7d",
            "checkpoint_ns": ""
        },
        "parent_checkpoint": {
            "checkpoint_id": "1f02f46f-7308-616c-8000-1b158a9a6955",
            "thread_id": "f15d70a1-27d4-4793-a897-de5609920b7d",
            "checkpoint_ns": ""
        },
        "checkpoint_id": "1f02f46f-733f-6b58-8001-ea90dcabb1bd",
        "parent_checkpoint_id": "1f02f46f-7308-616c-8000-1b158a9a6955"
    }

Optionally, to view the state of a thread at a given checkpoint, simply pass in the checkpoint id (or the entire checkpoint object):

=== "Python"

    ```python
    thread_state = await client.threads.get_state(
      thread_id=<THREAD_ID>
      checkpoint_id=<CHECKPOINT_ID>
    )
    ```

=== "Javascript"

    ```js
    const threadState = await client.threads.getState(<THREAD_ID>, <CHECKPOINT_ID>);
    ```

=== "CURL"

    ```bash
    curl --request GET \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/state/<CHECKPOINT_ID> \
    --header 'Content-Type: application/json'
    ```

#### Inspect Full Thread History

To view a thread's history, use the `get_history` method. This returns a list of every state the thread experienced. For more information see the [Python](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/?h=thread+status#langgraph_sdk.client.ThreadsClient.get_history) and [JS](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/js_ts_sdk_ref/#gethistory) reference docs.

### LangGraph Platform UI

You can also view threads in a deployment via the LangGraph Platform UI.

Inside your deployment, select the "Threads" tab. This will load a table of all of the threads in your deployment.

Select a thread to inspect its current state. To view its full history and for further debugging, open the thread in [LangGraph Studio](../../concepts//langgraph_studio.md).
