# Copying Threads

You may wish to copy an existing thread in order to keep the existing history and experiment with different runs in the future.

## Setup

To copy a thread, we must first set up our client and also create a thread we wish to copy. 

### SDK initialization

First, we need to setup our client so that we can communicate with our hosted graph:

=== "Python"

    ```python
    from langgraph_sdk import get_client
    client = get_client(url=<DEPLOYMENT_URL>)
    assistant_id = "agent"
    thread = await client.threads.create()
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl:<DEPLOYMENT_URL> });
    const assistantId = agent;
    const thread = await client.threads.create();
    ```

=== "CURL"

    ```bash
    curl --request POST \
      --url <DEPLOYMENT_URL>/threads \
      --header 'Content-Type: application/json'
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
      --header 'Content-Type: application/json'
    ```

There are a few ways to execute runs on a thread. You can [stream runs](https://langchain-ai.github.io/langgraph/cloud/how-tos/#streaming), execute them in the [background](https://langchain-ai.github.io/langgraph/cloud/how-tos/cloud_examples/background_run/), or even [setup cron jobs](https://langchain-ai.github.io/langgraph/cloud/how-tos/cloud_examples/cron_jobs/) to execute runs on a schedule. 

Whichever way you choose to execute runs on a thread is fine, as long as you ensure that you have properly executed at least one run on your thread before proceeding.

## Copying a thread

After you have created a thread and executed some runs on it, you may wish to copy the thread. This will create a new thread with the same history as the existing thread, and then allow you to continue executing runs.

### Create copy

=== "Python"

    ```python
    copied_thread = await client.threads.copy(<THREAD_ID>)
    ```

=== "Javascript"

    ```js
    let copiedThread = await client.threads.copy(<THREAD_ID>);
    ```

=== "CURL"

    ```bash
    curl --request POST --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/copy \
    --header 'Content-Type: application/json'
    ```

### Verify copy

We can verify that the history from the prior thread did indeed copy over correctly:

=== "Python"

    ```python
    def remove_thread_id(d):
      if 'metadata' in d and 'thread_id' in d['metadata']:
          del d['metadata']['thread_id']
      return d

    original_thread_history = list(map(remove_thread_id,await client.threads.get_history(<THREAD_ID>)))
    copied_thread_history = list(map(remove_thread_id,await client.threads.get_history(copied_thread['thread_id'])))

    if original_thread_history == copied_thread_history:
      print("The histories are the same.")
    else:
      print("The histories are different.")
    ```

=== "Javascript"

    ```js
    function removeThreadId(d) {
        if (d.metadata && d.metadata.thread_id) {
            delete d.metadata.thread_id;
        }
        return d;
    }

    // Assuming `client.threads.getHistory(threadId)` is an async function that returns a list of dicts
    async function compareThreadHistories(threadId, copiedThreadId) {
        const originalThreadHistory = (await client.threads.getHistory(threadId)).map(removeThreadId);
        const copiedThreadHistory = (await client.threads.getHistory(copiedThreadId)).map(removeThreadId);

        // Compare the two histories
        if (JSON.stringify(originalThreadHistory) === JSON.stringify(copiedThreadHistory)) {
            console.log("The histories are the same.");
        } else {
            console.log("The histories are different.");
        }
    }

    // Example usage
    compareThreadHistories(<THREAD_ID>, copiedThread.thread_id);
    ```

=== "CURL"

    ```bash
    if diff <(
        curl --request GET --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/history | jq -S 'map(del(.metadata.thread_id))'
    ) <(
        curl --request GET --url <DEPLOYMENT_URL>/threads/<COPIED_THREAD_ID>/history | jq -S 'map(del(.metadata.thread_id))'
    ) >/dev/null; then
        echo "The histories are the same."
    else
        echo "The histories are different."
    fi
    ```

Output:

    The histories are the same.