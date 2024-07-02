# How to Replay and Branch from Prior States

With Langgraph Cloud you have the ability to return to any of your prior states and either re-run the graph to reproduce issues noticed during testing, or branch out in a different way from what was originally done in the prior states. In this guide we will show a quick example of how to rerun past states and how to branch off from previous states as well.

## Setup

We are not going to show the full code for the graph we are hosting, but you can see it [here](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/time-travel/#build-the-agent) if you want to. Once this graph is hosted, we are ready to invoke it and wait for user input. 

### SDK initialization

First, we need to setup our client so that we can communicate with our hosted graph:

=== "Python"

    ```python
    from langgraph_sdk import get_client
    client = get_client(url="whatever-your-deployment-url-is")
    assistant_id = "agent"
    thread = await client.threads.create()
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({apiUrl:"whatever-your-deployment-url-is"});
    const assistantId = agent;
    const thread = await client.threads.create();
    ```

## Replay a state

### Initial invocation

Before replaying a state - we need to create states to replay from! In order to do this, let's invoke our graph with a simple message:

=== "Python"

    ```python
    input = {'messages':[{"role":"user","content":"Please search the weather in SF"}]}

    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id, # graph_id
        input=input,
        stream_mode="updates",
    ):
        if chunk.data and chunk.event != "metadata": 
            print(chunk.data)
    ```

=== "Javascript"

    ```js
    const input = {
      "messages": [
        {
          "role": "human",
          "content": "Please search the weather in SF",
        }
      ]
    }

    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantId,
      {
        input: input,
        streamMode: "updates",
      }
    );
    for await (const chunk of streamResponse) {
      if (chunk.data && chunk.event !== "metadata") {
        console.log(chunk.data);
      }
    }
    ```
    
Output:

    {'agent': {'messages': [{'content': [{'text': "Certainly! I'll use the search function to look up the current weather in San Francisco for you. Let me do that now.", 'type': 'text'}, {'id': 'toolu_011vroKUtWU7SBdrngpgpFMn', 'input': {'query': 'current weather in San Francisco'}, 'name': 'search', 'type': 'tool_use'}], 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-ee639877-d97d-40f8-96dc-d0d1ae22d203', 'example': False, 'tool_calls': [{'name': 'search', 'args': {'query': 'current weather in San Francisco'}, 'id': 'toolu_011vroKUtWU7SBdrngpgpFMn'}], 'invalid_tool_calls': [], 'usage_metadata': None}]}}
    {'action': {'messages': [{'content': '["I looked up: current weather in San Francisco. Result: It\'s sunny in San Francisco, but you better look out if you\'re a Gemini 😈."]', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'search', 'id': '7bad0e72-5ebe-4b08-9b8a-b99b0fe22fb7', 'tool_call_id': 'toolu_011vroKUtWU7SBdrngpgpFMn'}]}}
    {'agent': {'messages': [{'content': "Based on the search results, I can provide you with information about the current weather in San Francisco:\n\nThe weather in San Francisco is currently sunny. This is great news for outdoor activities and enjoying the city's beautiful sights.\n\nIt's worth noting that the search result included an unusual comment about Geminis, which isn't typically part of a weather report. This might be due to the search engine including some astrological information or a joke in its results. However, for the purpose of answering your question about the weather, we can focus on the fact that it's sunny in San Francisco.\n\nIf you need any more specific information about the weather in San Francisco, such as temperature, wind speed, or forecast for the coming days, please let me know, and I'd be happy to search for that information for you.", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-dbac539a-33c8-4f0c-9e20-91f318371e7c', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}}


Now let's get our list of states, and invoke from the third state (right before the tool get called):


=== "Python"

    ```python
    states = await client.threads.get_history(thread['thread_id'])

    # We can confirm that this state is correct by checking the 'next' attribute and seeing that it is the tool call node
    state_to_replay = states[2]
    print(state_to_replay['next'])
    ```

=== "Javascript"

    ```js
    const states = await client.threads.getHistory(thread['thread_id']);

    // We can confirm that this state is correct by checking the 'next' attribute and seeing that it is the tool call node
    const stateToReplay = states[2];
    console.log(stateToReplay['next']);
    ```

Output:

    ['action']



To rerun from a state, we need to pass in the `checkpoint_id` into the config of the run like follows:

=== "Python"

    ```python
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id, # graph_id
        input=None,
        stream_mode="updates",
        config={"configurable": {"thread_ts": state_to_replay['checkpoint_id']}}
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
        config: {"configurable": {"thread_ts": stateToReplay['checkpoint_id']}},
      }
    );
    for await (const chunk of streamResponse) {
      if (chunk.data && chunk.event !== "metadata") {
        console.log(chunk.data);
      }
    }
    ```

Output:

    {'action': {'messages': [{'content': '["I looked up: current weather in San Francisco. Result: It\'s sunny in San Francisco, but you better look out if you\'re a Gemini 😈."]', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'search', 'id': 'eba650e5-400e-4938-8508-f878dcbcc532', 'tool_call_id': 'toolu_011vroKUtWU7SBdrngpgpFMn'}]}}
    {'agent': {'messages': [{'content': "Based on the search results, I can provide you with information about the current weather in San Francisco:\n\nThe weather in San Francisco is currently sunny. This is great news if you're planning any outdoor activities or simply want to enjoy a pleasant day in the city.\n\nIt's worth noting that the search result included an unusual comment about Geminis, which doesn't seem directly related to the weather. This appears to be a playful or humorous addition to the weather report, possibly from the source where this information was obtained.\n\nIs there anything else you'd like to know about the weather in San Francisco or any other information you need?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-bc6dca3f-a1e2-4f59-a69b-fe0515a348bb', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}}


As we can see, the graph restarted from the tool node with the same input as our original graph run.

## Branch off from previous state

Using LangGraph's checkpointing, you can do more than just replay past states. You can branch off previous locations to let the agent explore alternate trajectories or to let a user "version control" changes in a workflow.

Let's show how to do this to edit the state at a particular point in time. Let's update the state to change the input to the tool

=== "Python"

    ```python
    # Let's now get the last message in the state
    # This is the one with the tool calls that we want to update
    last_message = state_to_replay['values']['messages'][-1]

    # Let's now update the args for that tool call
    last_message['tool_calls'][0]['args'] = {'query': 'current weather in SF'}

    new_state = await client.threads.update_state(thread['thread_id'],{"messages":[last_message]},checkpoint_id=state_to_replay['checkpoint_id'])
    ```

=== "Javascript"

    ```js
    // Let's now get the last message in the state
    // This is the one with the tool calls that we want to update
    let lastMessage = stateToReplay['values']['messages'][-1];

    // Let's now update the args for that tool call
    lastMessage['tool_calls'][0]['args'] = {'query': 'current weather in SF'};

    const newState = await client.threads.updateState(thread['thread_id'],{values:{"messages":[lastMessage]},checkpointId:stateToReplay['checkpoint_id']});
    ```

Now we can rerun our graph with this new config, starting from the `new_state`, which is a branch of our `state_to_replay`:

=== "Python"

    ```python
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant["assistant_id"], # graph_id
        input=None,
        stream_mode="updates",
        config={"configurable": {"thread_ts": new_state['configurable']['thread_ts']}}
    ):
        if chunk.data and chunk.event != "metadata": 
            print(chunk.data)
    ```

=== "Javascript"

    ```js
    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistant["assistant_id"],
      {
        input: null,
        streamMode: "updates",
        config: {"configurable": {"thread_ts": newState['configurable']['thread_ts']}},
      }
    );
    for await (const chunk of streamResponse) {
      if (chunk.data && chunk.event !== "metadata") {
        console.log(chunk.data);
      }
    }
    ```

Output:


    {'action': {'messages': [{'content': '["I looked up: current weather in SF. Result: It\'s sunny in San Francisco, but you better look out if you\'re a Gemini 😈."]', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'search', 'id': '2baf9941-4fda-4081-9f87-d76795d289f1', 'tool_call_id': 'toolu_011vroKUtWU7SBdrngpgpFMn'}]}}
    {'agent': {'messages': [{'content': "Based on the search results, I can provide you with information about the current weather in San Francisco (SF):\n\nThe weather in San Francisco is currently sunny. This means it's a clear day with plenty of sunshine. \n\nIt's worth noting that the specific temperature wasn't provided in the search result, but sunny weather in San Francisco typically means comfortable temperatures. San Francisco is known for its mild climate, so even on sunny days, it's often not too hot.\n\nThe search result also included a playful reference to astrological signs, mentioning Gemini. However, this is likely just a joke or part of the search engine's presentation and not related to the actual weather conditions.\n\nIs there any specific information about the weather in San Francisco you'd like to know more about? I'd be happy to perform another search if you need details on temperature, wind conditions, or the forecast for the coming days.", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-a83de52d-ed18-4402-9384-75c462485743', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}}


As we can see, the search query changed from San Francisco to SF, just as we had hoped!
