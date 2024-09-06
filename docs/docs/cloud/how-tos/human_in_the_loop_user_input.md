# How to Wait for User Input

<div class="admonition tip">
    <p class="admonition-title">Setup <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
    <p style="padding-top: 5px;">
        Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how LangSmith can help you <a href="https://docs.smith.langchain.com
        ">here</a>. 
    </p>
</div>    

One of the main human-in-the-loop interaction patterns is waiting for human input. A key use case involves asking the user clarifying questions. One way to accomplish this is simply go to the `END` node and exit the graph. Then, any user response comes back in as fresh invocation of the graph. This is basically just creating a chatbot architecture.

The issue with this is it is tough to resume back in a particular point in the graph. Often times the agent is halfway through some process, and just needs a bit of a user input. Although it is possible to design your graph in such a way where you have a `conditional_entry_point` to route user messages back to the right place, that is not super scalable (as it essentially involves having a routing function that can end up almost anywhere).

A separate way to do this is to have a node explicitly for getting user input. This is easy to implement in a notebook setting - you just put an `input()` call in the node. But that isn't exactly production ready.

Luckily, LangGraph makes it possible to do similar things in a production way. The basic idea is:

- Set up a node that represents human input. This can have specific incoming/outgoing edges (as you desire). There shouldn't actually be any logic inside this node.
- Add a breakpoint before the node. This will stop the graph before this node executes (which is good, because there's no real logic in it anyways)
- Use `.update_state` to update the state of the graph. Pass in whatever human response you get. The key here is to use the `as_node` parameter to apply this update **as if you were that node**. This will have the effect of making it so that when you resume execution next it resumes as if that node just acted, and not from the beginning.

## Setup

We are not going to show the full code for the graph we are hosting, but you can see it [here](../../how-tos/human_in_the_loop/wait-user-input.ipynb#build-the-agent) if you want to. Once this graph is hosted, we are ready to invoke it and wait for user input. 

### SDK initialization

First, we need to setup our client so that we can communicate with our hosted graph:

=== "Python"

    ```python
    from langgraph_sdk import get_client
    client = get_client(url=<DEPLOYMENT_URL>)
    # Using the graph deployed with the name "agent"
    assistant_id = "agent"
    thread = await client.threads.create()
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    // Using the graph deployed with the name "agent"
    const assistantId = "agent";
    const thread = await client.threads.create();
    ```

=== "CURL"

    ```bash
    curl --request POST \
      --url <DEPLOYMENT_URL>/threads \
      --header 'Content-Type: application/json' \
      --data '{}'
    ```

## Waiting for user input

### Initial invocation

Now, let's invoke our graph by interrupting before `ask_human` node:

=== "Python"

    ```python
    input = {
        "messages": [
            {
                "role": "user",
                "content": "Use the search tool to ask the user where they are, then look up the weather there",
            }
        ]
    }

    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=input,
        stream_mode="updates",
        interrupt_before=["ask_human"],
    ):
        if chunk.data and chunk.event != "metadata": 
            print(chunk.data)
    ```
=== "Javascript"

    ```js
    const input = {
      messages: [
        {
          role: "human",
          content: "Use the search tool to ask the user where they are, then look up the weather there"
        }
      ]
    };

    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantId,
      {
        input: input,
        streamMode: "updates",
        interruptBefore: ["ask_human"]
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
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"Use the search tool to ask the user where they are, then look up the weather there\"}]},
       \"interrupt_before\": [\"ask_human\"],
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

    {'agent': {'messages': [{'content': [{'text': "Certainly! I'll use the AskHuman function to ask the user about their location, and then I'll use the search function to look up the weather for that location. Let's start by asking the user where they are.", 'type': 'text'}, {'id': 'toolu_01RFahzYPvnPWTb2USk2RdKR', 'input': {'question': 'Where are you currently located?'}, 'name': 'AskHuman', 'type': 'tool_use'}], 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-a8422215-71d3-4093-afb4-9db141c94ddb', 'example': False, 'tool_calls': [{'name': 'AskHuman', 'args': {'question': 'Where are you currently located?'}, 'id': 'toolu_01RFahzYPvnPWTb2USk2RdKR'}], 'invalid_tool_calls': [], 'usage_metadata': None}]}}


### Adding user input to state

We now want to update this thread with a response from the user. We then can kick off another run.

Because we are treating this as a tool call, we will need to update the state as if it is a response from a tool call. In order to do this, we will need to check the state to get the ID of the tool call.


=== "Python"

    ```python
    state = await client.threads.get_state(thread['thread_id'])
    tool_call_id = state['values']['messages'][-1]['tool_calls'][0]['id']

    # We now create the tool call with the id and the response we want
    tool_message = [{"tool_call_id": tool_call_id, "type": "tool", "content": "san francisco"}]

    await client.threads.update_state(thread['thread_id'], {"messages": tool_message}, as_node="ask_human")
    ```

=== "Javascript"

    ```js
    const state = await client.threads.getState(thread["thread_id"]);
    const toolCallId = state.values.messages[state.values.messages.length - 1].tool_calls[0].id;

    // We now create the tool call with the id and the response we want
    const toolMessage = [
      {
        tool_call_id: toolCallId,
        type: "tool",
        content: "san francisco"
      }
    ];

    await client.threads.updateState(
      thread["thread_id"],
      { values: { messages: toolMessage } },
      { asNode: "ask_human" }
    );
    ```

=== "CURL"

    ```bash
    curl --request GET \
     --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/state \
     | jq -r '.values.messages[-1].tool_calls[0].id' \
     | sh -c '
         TOOL_CALL_ID="$1"
         
         # Construct the JSON payload
         JSON_PAYLOAD=$(printf "{\"messages\": [{\"tool_call_id\": \"%s\", \"type\": \"tool\", \"content\": \"san francisco\"}], \"as_node\": \"ask_human\"}" "$TOOL_CALL_ID")
         
         # Send the updated state
         curl --request POST \
              --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/state \
              --header "Content-Type: application/json" \
              --data "${JSON_PAYLOAD}"
     ' _ 
    ```

Output:

    {'configurable': {'thread_id': 'a9f322ae-4ed1-41ec-942b-38cb3d342c3a',
    'checkpoint_ns': '',
    'checkpoint_id': '1ef58e97-a623-63dd-8002-39a9a9b20be3'}}


### Invoking after receiving human input

We can now tell the agent to continue. We can just pass in None as the input to the graph, since no additional input is needed:

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
        streamMode: "updates"
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

    {'agent': {'messages': [{'content': [{'text': "Thank you for letting me know that you're in San Francisco. Now, I'll use the search function to look up the weather in San Francisco.", 'type': 'text'}, {'id': 'toolu_01K57ofmgG2wyJ8tYJjbq5k7', 'input': {'query': 'current weather in San Francisco'}, 'name': 'search', 'type': 'tool_use'}], 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-241baed7-db5e-44ce-ac3c-56431705c22b', 'example': False, 'tool_calls': [{'name': 'search', 'args': {'query': 'current weather in San Francisco'}, 'id': 'toolu_01K57ofmgG2wyJ8tYJjbq5k7'}], 'invalid_tool_calls': [], 'usage_metadata': None}]}}
    {'action': {'messages': [{'content': '["I looked up: current weather in San Francisco. Result: It\'s sunny in San Francisco, but you better look out if you\'re a Gemini 😈."]', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'search', 'id': '8b699b95-8546-4557-8e66-14ea71a15ed8', 'tool_call_id': 'toolu_01K57ofmgG2wyJ8tYJjbq5k7'}]}}
    {'agent': {'messages': [{'content': "Based on the search results, I can provide you with information about the current weather in San Francisco:\n\nThe weather in San Francisco is currently sunny. It's a beautiful day in the city! \n\nHowever, I should note that the search result included an unusual comment about Gemini zodiac signs. This appears to be either a joke or potentially irrelevant information added by the search engine. For accurate and detailed weather information, you might want to check a reliable weather service or app for San Francisco.\n\nIs there anything else you'd like to know about the weather or San Francisco?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-b4d7309f-f849-46aa-b6ef-475bcabd2be9', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}}

