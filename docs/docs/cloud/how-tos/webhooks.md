# Use Webhooks

You may wish to use webhooks in your client, especially when using async streams in case you want to update something in your service once the API call to LangGraph Cloud has finished running. To do so, you will need to expose an endpoint that can accept POST requests, and then pass it to your API request in the "webhook" parameter.

Currently, the SDK has not exposed this endpoint but you can access it through curl commands as follows.

The following endpoints accept `webhook` as a parameter: 

- Create Run -> POST /thread/{thread_id}/runs
- Create Thread Cron -> POST /thread/{thread_id}/runs/crons
- Stream Run -> POST /thread/{thread_id}/runs/stream
- Wait Run -> POST /thread/{thread_id}/runs/wait
- Create Cron -> POST /runs/crons
- Stream Run Stateless -> POST /runs/stream
- Wait Run Stateless -> POST /runs/wait

In this example, we will show calling a webhook after streaming a run. First, let's setup our assistant and thread:

=== "Python"

    ```python
    from langgraph_sdk import get_client

    client = get_client(url=<DEPLOYMENT_URL>)
    # get default assistant
    assistants = await client.assistants.search()
    assistant = [a for a in assistants if not a["config"]][0]
    # create thread
    thread = await client.threads.create()
    print(thread)
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    // get default assistant
    const assistants = await client.assistants.search();
    const assistant = assistants.find(a => !a.config);
    // create thread
    const thread = await client.threads.create();
    console.log(thread)
    ```

=== "CURL"

    ```bash
    curl --request POST \
        --url <DEPLOYMENT_URL>/assistants/search \
        --header 'Content-Type: application/json' \
        --data '{
            "limit": 10,
            "offset": 0
        }' | jq -c 'map(select(.config == null or .config == {})) | .[0]' && \
    curl --request POST \
        --url <DEPLOYMENT_URL>/threads \
        --header 'Content-Type: application/json' \
        --data '{}'
    ```

Output:

    {
        'thread_id': '9dde5490-2b67-47c8-aa14-4bfec88af217', 
        'created_at': '2024-08-30T23:07:38.242730+00:00', 
        'updated_at': '2024-08-30T23:07:38.242730+00:00', 
        'metadata': {}, 
        'status': 'idle', 
        'config': {}, 
        'values': None
    }

Now we can invoke a run with a webhook:

=== "Python"

    ```python
    # create input
    input = { "messages": [{ "role": "human", "content": "Hello!" }] }

    async for chunk in client.runs.stream(
        thread_id=thread["thread_id"],
        assistant_id="agent",
        input=input,
        stream_mode="events",
        webhook="your-webhook"
    ):
        # Do something with the stream output
        pass
    ```

=== "Javascript"

    ```js
    // create input
    const input = { "messages": [{ "role": "human", "content": "Hello!" }] }

    // stream events
    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistant["id"],
      {
        input: input,
        webhook: "your-webhook"
      }
    );
    for await (const chunk of streamResponse) {
      // Do something with the stream output
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
        --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
        --header 'Content-Type: application/json' \
        --data '{
            "assistant_id": <ASSISTANT_ID>,
            "input" : {"messages":[{"role": "user", "content": "Hello!"}]},
            "webhook": <YOUR_WEBHOOK_URL>
        }'
    ```

And that's it! Now you can trigger your custom webhooks whenever you want in your LangGraph applications!