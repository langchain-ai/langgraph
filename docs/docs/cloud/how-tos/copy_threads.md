# Copying Threads

You may wish to copy (i.e. "fork") an existing thread in order to keep the existing thread's history and create independent runs that do not affect the original thread. This guide shows how you can do that.

## Setup

This code assumes you already have a thread to copy. You can read about what a thread is [here](https://langchain-ai.github.io/langgraph/cloud/concepts/api/#threads) and learn how to stream a run on a thread in [these how-to guides](https://langchain-ai.github.io/langgraph/cloud/how-tos/#streaming).

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

    const client = new Client({ apiUrl: "<DEPLOYMENT_URL>" });
    const assistantId = "agent";
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

## Copying a thread

The code below assumes that a thread you'd like to copy already exists.

Copying a thread will create a new thread with the same history as the existing thread, and then allow you to continue executing runs.

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

    # Compare the two histories
    assert original_thread_history == copied_thread_history
    # if we made it here the assertion passed!
    print("The histories are the same.")
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
      console.assert(JSON.stringify(originalThreadHistory) === JSON.stringify(copiedThreadHistory));
      // if we made it here the assertion passed!
      console.log("The histories are the same.");
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