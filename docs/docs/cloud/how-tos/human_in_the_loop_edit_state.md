# How to Edit State of a Deployed Graph

When creating LangGraph agents, it is often nice to add a human-in-the-loop component. This can be helpful when giving them access to tools. Often in these situations you may want to edit the graph state before continuing (for example, to edit what tool is being called, or how it is being called).

This can be in several ways, but the primary supported way is to add an "interrupt" before a node is executed. This interrupts execution at that node. You can then use update_state to update the state, and then resume from that spot to continue.

## Setup

We are not going to show the full code for the graph we are hosting, but you can see it [here](../../how-tos/human_in_the_loop/edit-graph-state.ipynb#build-the-agent) if you want to. Once this graph is hosted, we are ready to invoke it and wait for user input. 

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
    const assistantId = "agent";
    const thread = await client.threads.create();
    ```

=== "CURL"

    ```bash
    curl --request POST \
      --url <DEPLOYMENT_URL>/threads \
      --header 'Content-Type: application/json'
    ```

## Editing state

### Initial invocation

Now let's invoke our graph, making sure to interrupt before the `action` node.

=== "Python"

    ```python
    input = { 'messages':[{ "role":"user", "content":"search for weather in SF" }] }

    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=input,
        stream_mode="updates",
        interrupt_before=["action"],
    ):
        if chunk.data and chunk.event != "metadata": 
            print(chunk.data)
    ```

=== "Javascript"

    ```js
    const input = {"messages": [{ "role": "human", "content": "search for weather in SF"}] }

    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantId,
      {
        input: input,
        streamMode: "updates",
        interruptBefore: ["action"],
      }
    );
    for await (const chunk of streamResponse) {
      if (chunk.data && chunk.event !== "metadata") {
        console.log(chunk.data);
      }
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
     --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
     --header 'Content-Type: application/json' \
     --data "{
       \"assistant_id\": \"agent\",
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"search for weather in SF\"}]},
       \"interrupt_before\": [\"action\"],
       \"stream_mode\": [
         \"updates\"
       ]
     }" | \
     sed 's/\r$//' | \
     awk '
     /^event:/ {
         if (data_content != "" && event_type != "metadata") {
             print data_content "\n"
         }
         sub(/^event: /, "", $0)
         event_type = $0
         data_content = ""
     }
     /^data:/ {
         sub(/^data: /, "", $0)
         data_content = $0
     }
     END {
         if (data_content != "" && event_type != "metadata") {
             print data_content "\n"
         }
     }
     '
    ```

Output:

    {'agent': {'messages': [{'content': [{'text': "Certainly! I'll search for the current weather in San Francisco for you using the search function. Here's how I'll do that:", 'type': 'text'}, {'id': 'toolu_01KEJMBFozSiZoS4mAcPZeqQ', 'input': {'query': 'current weather in San Francisco'}, 'name': 'search', 'type': 'tool_use'}], 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-6dbb0167-f8f6-4e2a-ab68-229b2d1fbb64', 'example': False, 'tool_calls': [{'name': 'search', 'args': {'query': 'current weather in San Francisco'}, 'id': 'toolu_01KEJMBFozSiZoS4mAcPZeqQ'}], 'invalid_tool_calls': [], 'usage_metadata': None}]}}


### Edit the state

Now, let's assume we actually meant to search for the weather in Sidi Frej (another city with the initials SF). We can edit the state to properly reflect that:


=== "Python"

    ```python
    # First, lets get the current state
    current_state = await client.threads.get_state(thread['thread_id'])

    # Let's now get the last message in the state
    # This is the one with the tool calls that we want to update
    last_message = current_state['values']['messages'][-1]

    # Let's now update the args for that tool call
    last_message['tool_calls'][0]['args'] = {'query': 'current weather in Sidi Frej'}

    # Let's now call `update_state` to pass in this message in the `messages` key
    # This will get treated as any other update to the state
    # It will get passed to the reducer function for the `messages` key
    # That reducer function will use the ID of the message to update it
    # It's important that it has the right ID! Otherwise it would get appended
    # as a new message
    await client.threads.update_state(thread['thread_id'], {"messages": last_message})
    ```

=== "Javascript"

    ```js
    // First, lets get the current state
    const currentState = await client.threads.getState(thread['thread_id']);

    // Let's now get the last message in the state
    // This is the one with the tool calls that we want to update
    let lastMessage = currentState['values']['messages'][-1];

    // Let's now update the args for that tool call
    lastMessage['tool_calls'][0]['args'] = {'query': 'current weather in Sidi Frej'};

    // Let's now call `update_state` to pass in this message in the `messages` key
    // This will get treated as any other update to the state
    // It will get passed to the reducer function for the `messages` key
    // That reducer function will use the ID of the message to update it
    // It's important that it has the right ID! Otherwise it would get appended
    // as a new message
    await client.threads.updateState(thread['thread_id'], {values:{"messages": lastMessage}});
    ```

=== "CURL"

    ```bash
    curl --request GET --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/state | \                                                                                      
    jq '.values.messages[-1] | (.tool_calls[0].args = {"query": "current weather in Sidi Frej"})' | \
    curl --request POST \
      --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/state \
      --header 'Content-Type: application/json' \
      --data @-
    ```

Output:

    {'configurable': {'thread_id': '9c8f1a43-9dd8-4017-9271-2c53e57cf66a',
      'checkpoint_ns': '',
      'checkpoint_id': '1ef58e7e-3641-649f-8002-8b4305a64858'}}



### Resume invocation

Now we can resume our graph run but with the updated state:


=== "Python"

    ```python
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=None,
        stream_mode="updates",
    ):
        if chunk.data and chunk.event != "metadata": 
            print(chunk.data)
    ```
=== "Javascript"

    ```js
    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantId,
      {
        input: null,
        streamMode: "updates",
      }
    );
    for await (const chunk of streamResponse) {
      if (chunk.data && chunk.event !== "metadata") {
        console.log(chunk.data);
      }
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \                                                                             
     --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
     --header 'Content-Type: application/json' \
     --data "{
       \"assistant_id\": \"agent\",
       \"stream_mode\": [
         \"updates\"
       ]
     }"| \ 
     sed 's/\r$//' | \
     awk '
     /^event:/ {
         if (data_content != "" && event_type != "metadata") {
             print data_content "\n"
         }
         sub(/^event: /, "", $0)
         event_type = $0
         data_content = ""
     }
     /^data:/ {
         sub(/^data: /, "", $0)
         data_content = $0
     }
     END {
         if (data_content != "" && event_type != "metadata") {
             print data_content "\n"
         }
     }
     '
    ```

Output:

    {'action': {'messages': [{'content': '["I looked up: current weather in Sidi Frej. Result: It\'s sunny in San Francisco, but you better look out if you\'re a Gemini 😈."]', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'search', 'id': '1161b8d1-bee4-4188-9be8-698aecb69f10', 'tool_call_id': 'toolu_01KEJMBFozSiZoS4mAcPZeqQ'}]}}
    {'agent': {'messages': [{'content': [{'text': 'I apologize for the confusion in my search query. It seems the search function interpreted "SF" as "Sidi Frej" instead of "San Francisco" as we intended. Let me search again with the full city name to get the correct information:', 'type': 'text'}, {'id': 'toolu_0111rrwgfAcmurHZn55qjqTR', 'input': {'query': 'current weather in San Francisco'}, 'name': 'search', 'type': 'tool_use'}], 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-b8c25779-cfb4-46fc-a421-48553551242f', 'example': False, 'tool_calls': [{'name': 'search', 'args': {'query': 'current weather in San Francisco'}, 'id': 'toolu_0111rrwgfAcmurHZn55qjqTR'}], 'invalid_tool_calls': [], 'usage_metadata': None}]}}
    {'action': {'messages': [{'content': '["I looked up: current weather in San Francisco. Result: It\'s sunny in San Francisco, but you better look out if you\'re a Gemini 😈."]', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'search', 'id': '6bc632ae-5ee6-4d01-9532-79c524a2d443', 'tool_call_id': 'toolu_0111rrwgfAcmurHZn55qjqTR'}]}}
    {'agent': {'messages': [{'content': "Now, based on the search results, I can provide you with information about the current weather in San Francisco:\n\nThe weather in San Francisco is currently sunny. \n\nIt's worth noting that the search result included an unusual comment about Gemini, which doesn't seem directly related to the weather. This might be due to the search engine including some astrological information or a joke in its results. However, for the purpose of weather information, we can focus on the fact that it's sunny in San Francisco right now.\n\nIs there anything else you'd like to know about the weather in San Francisco or any other location?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-227a042b-dd97-476e-af32-76a3703af5d8', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}}


As you can see it now looks up the current weather in Sidi Frej (although our dummy search node still returns results for SF because we don't actually do a search in this example, we just return the same "It's sunny in San Francisco ..." result every time).
