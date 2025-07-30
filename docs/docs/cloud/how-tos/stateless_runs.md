# Stateless Runs

Most of the time, you provide a `thread_id` to your client when you run your graph in order to keep track of prior runs through the persistent state implemented in LangGraph Platform. However, if you don't need to persist the runs you don't need to use the built in persistent state and can create stateless runs.

## Setup

First, let's setup our client:

=== "Python"

    ```python
    from langgraph_sdk import get_client

    client = get_client(url=<DEPLOYMENT_URL>)
    # Using the graph deployed with the name "agent"
    assistant_id = "agent"
    # create thread
    thread = await client.threads.create()
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    // Using the graph deployed with the name "agent"
    const assistantId = "agent";
    // create thread
    const thread = await client.threads.create();
    ```

=== "CURL"

    ```bash
    curl --request POST \
        --url <DEPLOYMENT_URL>/assistants/search \
        --header 'Content-Type: application/json' \
        --data '{
            "limit": 10,
            "offset": 0
        }' | jq -c 'map(select(.config == null or .config == {})) | .[0].graph_id' && \
    curl --request POST \
        --url <DEPLOYMENT_URL>/threads \
        --header 'Content-Type: application/json' \
        --data '{}'
    ```

## Stateless streaming

We can stream the results of a stateless run in an almost identical fashion to how we stream from a run with the state attribute, but instead of passing a value to the `thread_id` parameter, we pass `None`:

=== "Python"

    ```python
    input = {
        "messages": [
            {"role": "user", "content": "Hello! My name is Bagatur and I am 26 years old."}
        ]
    }

    async for chunk in client.runs.stream(
        # Don't pass in a thread_id and the stream will be stateless
        None,
        assistant_id,
        input=input,
        stream_mode="updates",
    ):
        if chunk.data and "run_id" not in chunk.data:
            print(chunk.data)
    ```

=== "Javascript"

    ```js
    let input = {
      messages: [
        { role: "user", content: "Hello! My name is Bagatur and I am 26 years old." }
      ]
    };

    const streamResponse = client.runs.stream(
      // Don't pass in a thread_id and the stream will be stateless
      null,
      assistantId,
      {
        input,
        streamMode: "updates"
      }
    );
    for await (const chunk of streamResponse) {
      if (chunk.data && !("run_id" in chunk.data)) {
        console.log(chunk.data);
      }
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
        --url <DEPLOYMENT_URL>/runs/stream \
        --header 'Content-Type: application/json' \
        --data "{
            \"assistant_id\": \"agent\",
            \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"Hello! My name is Bagatur and I am 26 years old.\"}]},
            \"stream_mode\": [
                \"updates\"
            ]
        }" | jq -c 'select(.data and (.data | has("run_id") | not)) | .data'
    ```

Output:

    {'agent': {'messages': [{'content': "Hello Bagatur! It's nice to meet you. Thank you for introducing yourself and sharing your age. Is there anything specific you'd like to know or discuss? I'm here to help with any questions or topics you're interested in.", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-489ec573-1645-4ce2-a3b8-91b391d50a71', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}}

## Waiting for stateless results

In addition to streaming, you can also wait for a stateless result by using the `.wait` function like follows:

=== "Python"

    ```python
    stateless_run_result = await client.runs.wait(
        None,
        assistant_id,
        input=input,
    )
    print(stateless_run_result)
    ```

=== "Javascript"

    ```js
    let statelessRunResult = await client.runs.wait(
      null,
      assistantId,
      { input: input }
    );
    console.log(statelessRunResult);
    ```

=== "CURL"

    ```bash
    curl --request POST \
        --url <DEPLOYMENT_URL>/runs/wait \
        --header 'Content-Type: application/json' \
        --data '{
            "assistant_id": <ASSISTANT_IDD>,
        }'
    ```

Output:

    {
        'messages': [
            {
                'content': 'Hello! My name is Bagatur and I am 26 years old.',
                'additional_kwargs': {},
                'response_metadata': {},
                'type': 'human',
                'name': None,
                'id': '5e088543-62c2-43de-9d95-6086ad7f8b48',
                'example': False}
            ,
            {
                'content': "Hello Bagatur! It's nice to meet you. Thank you for introducing yourself and sharing your age. Is there anything specific you'd like to know or discuss? I'm here to help with any questions or topics you'd like to explore.",
                'additional_kwargs': {},
                'response_metadata': {},
                'type': 'ai',
                'name': None,
                'id': 'run-d6361e8d-4d4c-45bd-ba47-39520257f773',
                'example': False,
                'tool_calls': [],
                'invalid_tool_calls': [],
                'usage_metadata': None
            }
        ]
    }