# Copying Threads

You may wish to copy an existing thread in order to keep the existing history and experiment with different runs in the future.

## Setup

To copy a thread, we must first set up our client and also create a thread we wish to copy. 

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

### Thread creation

In order to copy a thread - you need to create one first. While it won't throw an error if you try to copy a thread before doing any runs, it doesn't really make sense to copy an empty thread, so in practice you need to create the thread and then execute some runs on it.

Here is how you can create a thread:

=== "Python"

    ```python
    thread = await client.threads.create()
    ```

=== "Javascript"

    ```js
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

There are a few ways to execute runs on a thread. You can [stream runs](https://langchain-ai.github.io/langgraph/cloud/how-tos/#streaming), execute them in the [background](https://langchain-ai.github.io/langgraph/cloud/how-tos/cloud_examples/background_run/), or even [setup cron jobs](https://langchain-ai.github.io/langgraph/cloud/how-tos/cloud_examples/cron_jobs/) to execute runs on a schedule. 

Whichever way you choose to execute runs on a thread is fine, as long as you ensure that you have properly executed at least one run on your thread before proceeding.

## Copying a thread

After you have created a thread and executed some runs on it, you may wish to copy the thread. This will create a new thread with the same history as the existing thread, and then allow you to continue executing runs. You may wish to create multiple copies in order to experiment with different runs.

=== "Python"

    ```python
    copied_thread = await client.threads.copy("a74af490-f5e4-4a32-bdcf-526ab8534594")
    ```

=== "Javascript"

    ```js
    let copiedThread = await client.threads.copy(<THREAD_ID>);
    ```