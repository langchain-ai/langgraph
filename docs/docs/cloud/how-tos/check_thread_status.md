# Check the Status of your Threads

There are a variety of reasons you would want

## Setup

To start, we can setup our client with whatever URL you are hosting your graph from:

### SDK initialization

First, we need to setup our client so that we can communicate with our hosted graph:

=== "Python"

    ```python
    from langgraph_sdk import get_client
    client = get_client(url="<DEPLOYMENT_URL>")
    assistant_id = "agent"
    thread = await client.threads.create()
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl:"<DEPLOYMENT_URL>" });
    const assistantId = agent;
    const thread = await client.threads.create();
    ```

=== "CURL"

    ```bash
    curl --request POST \
      --url <DEPLOYMENT_URL>/threads \
      --header 'Content-Type: application/json' \
      --data '{
        "metadata": {}
      }'
    ```

## Find idle threads

We can use the following commands to find threads that are idle:

=== "Python"

    ```python
    print(await client.threads.search(status="idle",limit=1))
    ```

=== "Javascript"

    ```js
    console.log(await client.threads.search({status: "idle",limit:1}));
    ```

=== "CURL"

    ```bash
    curl --request POST \  
    --url <DEPLOYMENT_URL>/threads/search \
    --header 'Content-Type: application/json' \
    --data '{"status": "idle", "limit": 1}'
    ```

Output:

    [{'thread_id': 'cacf79bb-4248-4d01-aabc-938dbd60ed2c',
    'created_at': '2024-08-14T17:36:38.921660+00:00',
    'updated_at': '2024-08-14T17:36:38.921660+00:00',
    'metadata': {'graph_id': 'agent'},
    'status': 'idle',
    'config': {'configurable': {}}}]


## Find interrupted threads

We can use the following commands to find threads that have been interrupted in the middle of a run: 

=== "Python"

    ```python
    print(await client.threads.search(status="interrupted",limit=1))
    ```

=== "Javascript"

    ```js
    console.log(await client.threads.search({status: "interrupted",limit:1}));
    ```

=== "CURL"

    ```bash
    curl --request POST \  
    --url <DEPLOYMENT_URL>/threads/search \
    --header 'Content-Type: application/json' \
    --data '{"status": "interrupted", "limit": 1}'
    ```

Output:

    [{'thread_id': '0d282b22-bbd5-4d95-9c61-04dcc2e302a5',
    'created_at': '2024-08-14T17:41:50.235455+00:00',
    'updated_at': '2024-08-14T17:41:50.235455+00:00',
    'metadata': {'graph_id': 'agent'},
    'status': 'interrupted',
    'config': {'configurable': {}}}]
    
## Find busy threads

We can use the following commands to find threads that are busy, meaning they are currently handling the execution of a run:

=== "Python"

    ```python
    print(await client.threads.search(status="busy",limit=1))
    ```

=== "Javascript"

    ```js
    console.log(await client.threads.search({status: "busy",limit: 1}));
    ```

=== "CURL"

    ```bash
    curl --request POST \  
    --url <DEPLOYMENT_URL>/threads/search \
    --header 'Content-Type: application/json' \
    --data '{"status": "busy", "limit": 1}'
    ```

Output:

    [{'thread_id': '0d282b22-bbd5-4d95-9c61-04dcc2e302a5',
    'created_at': '2024-08-14T17:41:50.235455+00:00',
    'updated_at': '2024-08-14T17:41:50.235455+00:00',
    'metadata': {'graph_id': 'agent'},
    'status': 'busy',
    'config': {'configurable': {}}}]

## Find specific threads

You may also want to check the status of specific threads, which you can do in a few ways:

### Find by ID

You can use the `get` function to find the status of a specific thread, as long as you have the ID saved

=== "Python"

    ```python
    print((await client.threads.get(<THREAD_ID>))['status'])
    ```

=== "Javascript"

    ```js
    console.log((await client.threads.get(<THREAD_ID>)).status);
    ```

=== "CURL"

    ```bash
    curl --request GET \ 
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID> \
    --header 'Content-Type: application/json' | jq -r '.status'
    ```

Output:

    'idle'

### Find by metadata

The search endpoint for threads also allows you to filter on metadata, which can be helpful if you use metadata to tag threads in order to keep them organized:

=== "Python"

    ```python
    print(await client.threads.search(metadata={"foo":"bar"},limit=1))
    ```

=== "Javascript"

    ```js
    console.log(await client.threads.search({metadata: {"foo":"bar"},limit: 1}));
    ```

=== "CURL"

    ```bash
    curl --request POST \  
    --url <DEPLOYMENT_URL>/threads/search \
    --header 'Content-Type: application/json' \
    --data '{"metadata": {"foo":"bar"}, "limit": 1}'
    ```

Output:

    [{'thread_id': '323e1557-db89-4f78-8361-c2b4e58dd348',
    'created_at': '2024-08-14T18:11:54.788901+00:00',
    'updated_at': '2024-08-14T18:11:54.788901+00:00',
    'metadata': {'foo': 'bar'},
    'status': 'idle',
    'config': {}}]