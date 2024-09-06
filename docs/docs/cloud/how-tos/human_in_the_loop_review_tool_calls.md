# Review Tool Calls

Human-in-the-loop (HIL) interactions are crucial for [agentic systems](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#human-in-the-loop). A common pattern is to add some human in the loop step after certain tool calls. These tool calls often lead to either a function call or saving of some information. Examples include:

- A tool call to execute SQL, which will then be run by the tool
- A tool call to generate a summary, which will then be saved to the State of the graph

Note that using tool calls is common **whether actually calling tools or not**.

There are typically a few different interactions you may want to do here:

1. Approve the tool call and continue
2. Modify the tool call manually and then continue
3. Give natural language feedback, and then pass that back to the agent instead of continuing

We can implement this in LangGraph using a [breakpoint](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/breakpoints/): breakpoints allow us to interrupt graph execution before a specific step. At this breakpoint, we can manually update the graph state taking one of the three options above

## Setup

We are not going to show the full code for the graph we are hosting, but you can see it [here](../../how-tos/human_in_the_loop/review-tool-calls.ipynb#simple-usage) if you want to. Once this graph is hosted, we are ready to invoke it and wait for user input. 

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

## Example with no review

Let's look at an example when no review is required (because no tools are called)

=== "Python"

    ```python
    input = { 'messages':[{ "role":"user", "content":"hi!" }] }

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
    const input = { "messages": [{ "role": "user", "content": "hi!" }] };

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
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"hi!\"}]},
       \"stream_mode\": [
         \"updates\"
       ],
       \"interrupt_before\": [\"action\"]
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

    {'messages': [{'content': 'hi!', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '39c51f14-2d5c-4690-883a-d940854b1845', 'example': False}]}
    {'messages': [{'content': 'hi!', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '39c51f14-2d5c-4690-883a-d940854b1845', 'example': False}, {'content': [{'text': "Hello! Welcome. How can I assist you today? Is there anything specific you'd like to know or any information you're looking for?", 'type': 'text', 'index': 0}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-d65e07fb-43ff-4d98-ab6b-6316191b9c8b', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 355, 'output_tokens': 31, 'total_tokens': 386}}]}


If we check the state, we can see that it is finished

=== "Python"

    ```python
    state = await client.threads.get_state(thread["thread_id"])

    print(state['next'])
    ```

=== "Javascript"

    ```js
    const state = await client.threads.getState(thread["thread_id"]);

    console.log(state.next);
    ```

=== "CURL"

    ```bash
    curl --request GET \
        --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/state | jq -c '.next'
    ```

Output:

    []

## Example of approving tool

Let's now look at what it looks like to approve a tool call. Note that we don't need to pass an interrupt to our streaming calls because the graph (defined [here](../../how-tos/human_in_the_loop/review-tool-calls.ipynb#simple-usage)) was already compiled with an interrupt before the `human_review_node`.

=== "Python"

    ```python
    input = {"messages": [{"role": "user", "content": "what's the weather in sf?"}]}

    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=input,
    ):
        if chunk.data and chunk.event != "metadata": 
            print(chunk.data)
    ```

=== "Javascript"

    ```js
    const input = { "messages": [{ "role": "user", "content": "what's the weather in sf?" }] };

    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantId,
      {
        input: input,
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
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"what's the weather in sf?\"}]}
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

    {'messages': [{'content': "what's the weather in sf?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '54e19d6e-89fa-44fb-b92c-12e7dd4ddf08', 'example': False}]}
    {'messages': [{'content': "what's the weather in sf?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '54e19d6e-89fa-44fb-b92c-12e7dd4ddf08', 'example': False}, {'content': [{'text': "Certainly! I can help you check the weather in San Francisco. To get this information, I'll use the weather search function. Let me do that for you right away.", 'type': 'text', 'index': 0}, {'id': 'toolu_015yrR3GMDXe6X8m2p9CsEDN', 'input': {}, 'name': 'weather_search', 'type': 'tool_use', 'index': 1, 'partial_json': '{"city": "San Francisco"}'}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'tool_use', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-45a6b6c3-ac69-42a4-8957-d982203d6392', 'example': False, 'tool_calls': [{'name': 'weather_search', 'args': {'city': 'San Francisco'}, 'id': 'toolu_015yrR3GMDXe6X8m2p9CsEDN', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 360, 'output_tokens': 90, 'total_tokens': 450}}]}


If we now check, we can see that it is waiting on human review:

=== "Python"

    ```python
    state = await client.threads.get_state(thread["thread_id"])

    print(state['next'])
    ```

=== "Javascript"

    ```js
    const state = await client.threads.getState(thread["thread_id"]);

    console.log(state.next);
    ```

=== "CURL"

    ```bash
    curl --request GET \
        --url <DELPOYMENT_URL>/threads/<THREAD_ID>/state | jq -c '.next'
    ```

Output:

    ['human_review_node']

To approve the tool call, we can just continue the thread with no edits. To do this, we just create a new run with no inputs.

=== "Python"

    ```python
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=None,
        stream_mode="values",
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
        streamMode: "values",
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
       \"assistant_id\": \"agent\"
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

    {'messages': [{'content': "what's the weather in sf?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '54e19d6e-89fa-44fb-b92c-12e7dd4ddf08', 'example': False}, {'content': [{'text': "Certainly! I can help you check the weather in San Francisco. To get this information, I'll use the weather search function. Let me do that for you right away.", 'type': 'text', 'index': 0}, {'id': 'toolu_015yrR3GMDXe6X8m2p9CsEDN', 'input': {}, 'name': 'weather_search', 'type': 'tool_use', 'index': 1, 'partial_json': '{"city": "San Francisco"}'}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'tool_use', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-45a6b6c3-ac69-42a4-8957-d982203d6392', 'example': False, 'tool_calls': [{'name': 'weather_search', 'args': {'city': 'San Francisco'}, 'id': 'toolu_015yrR3GMDXe6X8m2p9CsEDN', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 360, 'output_tokens': 90, 'total_tokens': 450}}, {'content': 'Sunny!', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'weather_search', 'id': '826cd0f2-9cc6-46f0-b7df-daa6a05d13d2', 'tool_call_id': 'toolu_015yrR3GMDXe6X8m2p9CsEDN', 'artifact': None, 'status': 'success'}]}
    {'messages': [{'content': "what's the weather in sf?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '54e19d6e-89fa-44fb-b92c-12e7dd4ddf08', 'example': False}, {'content': [{'text': "Certainly! I can help you check the weather in San Francisco. To get this information, I'll use the weather search function. Let me do that for you right away.", 'type': 'text', 'index': 0}, {'id': 'toolu_015yrR3GMDXe6X8m2p9CsEDN', 'input': {}, 'name': 'weather_search', 'type': 'tool_use', 'index': 1, 'partial_json': '{"city": "San Francisco"}'}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'tool_use', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-45a6b6c3-ac69-42a4-8957-d982203d6392', 'example': False, 'tool_calls': [{'name': 'weather_search', 'args': {'city': 'San Francisco'}, 'id': 'toolu_015yrR3GMDXe6X8m2p9CsEDN', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 360, 'output_tokens': 90, 'total_tokens': 450}}, {'content': 'Sunny!', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'weather_search', 'id': '826cd0f2-9cc6-46f0-b7df-daa6a05d13d2', 'tool_call_id': 'toolu_015yrR3GMDXe6X8m2p9CsEDN', 'artifact': None, 'status': 'success'}, {'content': [{'text': "\n\nGreat news! The weather in San Francisco is sunny today. It's a beautiful day in the city by the bay. Is there anything else you'd like to know about the weather or any other information I can help you with?", 'type': 'text', 'index': 0}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-5d5fd0f1-a939-447e-801a-9aaa812322d3', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 464, 'output_tokens': 50, 'total_tokens': 514}}]}

## Edit Tool Call

Let's now say we want to edit the tool call. E.g. change some of the parameters (or even the tool called!) but then execute that tool.

=== "Python"

    ```python
    input = {"messages": [{"role": "user", "content": "what's the weather in sf?"}]}

    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=input,
        stream_mode="values",
    ):
        if chunk.data and chunk.event != "metadata": 
            print(chunk.data)
    ```

=== "Javascript"

    ```js
    const input = { "messages": [{ "role": "user", "content": "what's the weather in sf?" }] };

    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantId,
      {
        input: input,
        streamMode: "values",
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
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"what's the weather in sf?\"}]}
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

    {'messages': [{'content': "what's the weather in sf?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'cec11391-84da-464b-bd2a-bd4f0d93b9ee', 'example': False}]}
    {'messages': [{'content': "what's the weather in sf?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'cec11391-84da-464b-bd2a-bd4f0d93b9ee', 'example': False}, {'content': [{'text': 'To get the weather information for San Francisco, I can use the weather_search function. Let me do that for you.', 'type': 'text', 'index': 0}, {'id': 'toolu_01SunSpDurNfcnXppWLPrtjC', 'input': {}, 'name': 'weather_search', 'type': 'tool_use', 'index': 1, 'partial_json': '{"city": "San Francisco"}'}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'tool_use', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-6326da9f-6061-4e12-8586-482e32ab4cab', 'example': False, 'tool_calls': [{'name': 'weather_search', 'args': {'city': 'San Francisco'}, 'id': 'toolu_01SunSpDurNfcnXppWLPrtjC', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 360, 'output_tokens': 80, 'total_tokens': 440}}]}


To do this, we first need to update the state. We can do this by passing a message in with the **same** id of the message we want to overwrite. This will have the effect of **replacing** that old message. Note that this is only possible because of the **reducer** we are using that replaces messages with the same ID - read more about that [here](https://langchain-ai.github.io/langgraph/concepts/low_level/#working-with-messages-in-graph-state).


=== "Python"

    ```python
    # To get the ID of the message we want to replace, we need to fetch the current state and find it there.
    state = await client.threads.get_state(thread['thread_id'])
    print("Current State:")
    print(state['values'])
    print("\nCurrent Tool Call ID:")
    current_content = state['values']['messages'][-1]['content']
    current_id = state['values']['messages'][-1]['id']
    tool_call_id = state['values']['messages'][-1]['tool_calls'][0]['id']
    print(tool_call_id)

    # We now need to construct a replacement tool call.
    # We will change the argument to be `San Francisco, USA`
    # Note that we could change any number of arguments or tool names - it just has to be a valid one
    new_message = {
        "role": "assistant", 
        "content": current_content,
        "tool_calls": [
            {
                "id": tool_call_id,
                "name": "weather_search",
                "args": {"city": "San Francisco, USA"}
            }
        ],
        # This is important - this needs to be the same as the message you replacing!
        # Otherwise, it will show up as a separate message
        "id": current_id
    }
    await client.threads.update_state(
        # This is the config which represents this thread
        thread['thread_id'], 
        # This is the updated value we want to push
        {"messages": [new_message]}, 
        # We push this update acting as our human_review_node
        as_node="human_review_node"
    )

    print("\nResuming Execution")
    # Let's now continue executing from here
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=None,
    ):
        if chunk.data and chunk.event != "metadata": 
            print(chunk.data)
    ```

=== "Javascript"

    ```js
    const state = await client.threads.getState(thread.thread_id);
    console.log("Current State:");
    console.log(state.values);

    console.log("\nCurrent Tool Call ID:");
    const lastMessage = state.values.messages[state.values.messages.length - 1];
    const currentContent = lastMessage.content;
    const currentId = lastMessage.id;
    const toolCallId = lastMessage.tool_calls[0].id;
    console.log(toolCallId);

    // Construct a replacement tool call
    const newMessage = {
      role: "assistant",
      content: currentContent,
      tool_calls: [
        {
          id: toolCallId,
          name: "weather_search",
          args: { city: "San Francisco, USA" }
        }
      ],
      // Ensure the ID is the same as the message you're replacing
      id: currentId
    };

    await client.threads.updateState(
      thread.thread_id,  // Thread ID
      {
        values: { "messages": [newMessage] },  // Updated message
        asNode: "human_review_node"
      }  // Acting as human_review_node
    );

    console.log("\nResuming Execution");
    // Continue executing from here
    const streamResponseResumed = client.runs.stream(
      thread["thread_id"],
      assistantId,
      {
        input: null,
      }
    );

    for await (const chunk of streamResponseResumed) {
      if (chunk.data && chunk.event !== "metadata") {
        console.log(chunk.data);
      }
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/state \
    --header 'Content-Type: application/json' \
    --data "{
        \"values\": { \"messages\": [$(curl --request GET \
            --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/state |
            jq -c '{
            role: "assistant",
            content: .values.messages[-1].content,
            tool_calls: [
                {
                id: .values.messages[-1].tool_calls[0].id,
                name: "weather_search",
                args: { city: "San Francisco, USA" }
                }
            ],
            id: .values.messages[-1].id
            }')
        ]},
        \"as_node\": \"human_review_node\"
    }" && echo "Resuming Execution" && curl --request POST \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
    --header 'Content-Type: application/json' \
    --data '{
    "assistant_id": "agent"
    }' | \
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

    Current State:
    {'messages': [{'content': "what's the weather in sf?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '8713d1fa-9b26-4eab-b768-dafdaac70590', 'example': False}, {'content': [{'text': 'To get the weather information for San Francisco, I can use the weather_search function. Let me do that for you.', 'type': 'text', 'index': 0}, {'id': 'toolu_01VzagzsUGZsNMwW1wHkcw7h', 'input': {}, 'name': 'weather_search', 'type': 'tool_use', 'index': 1, 'partial_json': '{"city": "San Francisco"}'}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'tool_use', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-ede13f26-daf5-4d8f-817a-7611075bbcf1', 'example': False, 'tool_calls': [{'name': 'weather_search', 'args': {'city': 'San Francisco'}, 'id': 'toolu_01VzagzsUGZsNMwW1wHkcw7h', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 360, 'output_tokens': 80, 'total_tokens': 440}}]}

    Current Tool Call ID:
    toolu_01VzagzsUGZsNMwW1wHkcw7h

    Resuming Execution
    {'messages': [{'content': "what's the weather in sf?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '8713d1fa-9b26-4eab-b768-dafdaac70590', 'example': False}, {'content': [{'text': 'To get the weather information for San Francisco, I can use the weather_search function. Let me do that for you.', 'type': 'text', 'index': 0}, {'id': 'toolu_01VzagzsUGZsNMwW1wHkcw7h', 'input': {}, 'name': 'weather_search', 'type': 'tool_use', 'index': 1, 'partial_json': '{"city": "San Francisco"}'}], 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-ede13f26-daf5-4d8f-817a-7611075bbcf1', 'example': False, 'tool_calls': [{'name': 'weather_search', 'args': {'city': 'San Francisco, USA'}, 'id': 'toolu_01VzagzsUGZsNMwW1wHkcw7h', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'Sunny!', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'weather_search', 'id': '7fc7d463-66bf-4555-9929-6af483de169b', 'tool_call_id': 'toolu_01VzagzsUGZsNMwW1wHkcw7h', 'artifact': None, 'status': 'success'}]}
    {'messages': [{'content': "what's the weather in sf?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '8713d1fa-9b26-4eab-b768-dafdaac70590', 'example': False}, {'content': [{'text': 'To get the weather information for San Francisco, I can use the weather_search function. Let me do that for you.', 'type': 'text', 'index': 0}, {'id': 'toolu_01VzagzsUGZsNMwW1wHkcw7h', 'input': {}, 'name': 'weather_search', 'type': 'tool_use', 'index': 1, 'partial_json': '{"city": "San Francisco"}'}], 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-ede13f26-daf5-4d8f-817a-7611075bbcf1', 'example': False, 'tool_calls': [{'name': 'weather_search', 'args': {'city': 'San Francisco, USA'}, 'id': 'toolu_01VzagzsUGZsNMwW1wHkcw7h', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'Sunny!', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'weather_search', 'id': '7fc7d463-66bf-4555-9929-6af483de169b', 'tool_call_id': 'toolu_01VzagzsUGZsNMwW1wHkcw7h', 'artifact': None, 'status': 'success'}, {'content': [{'text': "\n\nBased on the search result, the weather in San Francisco is sunny! It's a beautiful day in the city by the bay. Is there anything else you'd like to know about the weather or any other information I can help you with?", 'type': 'text', 'index': 0}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-d90ce97a-39f9-4330-985e-67c5f351a0c5', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 455, 'output_tokens': 52, 'total_tokens': 507}}]}

## Give feedback to a tool call

Sometimes, you may not want to execute a tool call, but you also may not want to ask the user to manually modify the tool call. In that case it may be better to get natural language feedback from the user. You can then insert these feedback as a mock **RESULT** of the tool call.

There are multiple ways to do this:

You could add a new message to the state (representing the "result" of a tool call)
You could add TWO new messages to the state - one representing an "error" from the tool call, other HumanMessage representing the feedback
Both are similar in that they involve adding messages to the state. The main difference lies in the logic AFTER the `human_node` and how it handles different types of messages.

For this example we will just add a single tool call representing the feedback. Let's see this in action!

=== "Python"

    ```python
    input = {"messages": [{"role": "user", "content": "what's the weather in sf?"}]}

    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=input,
    ):
        if chunk.data and chunk.event != "metadata": 
            print(chunk.data)
    ```

=== "Javascript"

    ```js
    const input = { "messages": [{ "role": "user", "content": "what's the weather in sf?" }] };

    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantId,
      {
        input: input,
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
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"what's the weather in sf?\"}]}
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

    {'messages': [{'content': "what's the weather in sf?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'c80f13d0-674d-4233-b6a0-3940509d3cf3', 'example': False}]}
    {'messages': [{'content': "what's the weather in sf?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'c80f13d0-674d-4233-b6a0-3940509d3cf3', 'example': False}, {'content': [{'text': 'To get the weather information for San Francisco, I can use the weather_search function. Let me do that for you.', 'type': 'text', 'index': 0}, {'id': 'toolu_016XyTdFA8NuPWeLyZPSzoM3', 'input': {}, 'name': 'weather_search', 'type': 'tool_use', 'index': 1, 'partial_json': '{"city": "San Francisco"}'}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'tool_use', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-4911ac27-3d7c-4edf-a3ca-c2908e3922eb', 'example': False, 'tool_calls': [{'name': 'weather_search', 'args': {'city': 'San Francisco'}, 'id': 'toolu_016XyTdFA8NuPWeLyZPSzoM3', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 360, 'output_tokens': 80, 'total_tokens': 440}}]}

To do this, we first need to update the state. We can do this by passing a message in with the same **tool call id** of the tool call we want to respond to. Note that this is a **different*** ID from above

=== "Python"

    ```python
    # To get the ID of the message we want to replace, we need to fetch the current state and find it there.
    state = await client.threads.get_state(thread['thread_id'])
    print("Current State:")
    print(state['values'])
    print("\nCurrent Tool Call ID:")
    tool_call_id = state['values']['messages'][-1]['tool_calls'][0]['id']
    print(tool_call_id)

    # We now need to construct a replacement tool call.
    # We will change the argument to be `San Francisco, USA`
    # Note that we could change any number of arguments or tool names - it just has to be a valid one
    new_message = {
        "role": "tool", 
        # This is our natural language feedback
        "content": "User requested changes: pass in the country as well",
        "name": "weather_search",
        "tool_call_id": tool_call_id
    }
    await client.threads.update_state(
        # This is the config which represents this thread
        thread['thread_id'], 
        # This is the updated value we want to push
        {"messages": [new_message]}, 
        # We push this update acting as our human_review_node
        as_node="human_review_node"
    )

    print("\nResuming execution")
    # Let's now continue executing from here
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=None,
        stream_mode="values",
    ):
        if chunk.data and chunk.event != "metadata": 
            print(chunk.data)
    ```

=== "Javascript"

    ```js
    const state = await client.threads.getState(thread.thread_id);
    console.log("Current State:");
    console.log(state.values);

    console.log("\nCurrent Tool Call ID:");
    const lastMessage = state.values.messages[state.values.messages.length - 1];
    const toolCallId = lastMessage.tool_calls[0].id;
    console.log(toolCallId);

    // Construct a replacement tool call
    const newMessage = {
      role: "tool",
      content: "User requested changes: pass in the country as well",
      name: "weather_search",
      tool_call_id: toolCallId,
    };

    await client.threads.updateState(
      thread.thread_id,  // Thread ID
      {
        values: { "messages": [newMessage] },  // Updated message
        asNode: "human_review_node"
      }  // Acting as human_review_node
    );

    console.log("\nResuming Execution");
    // Continue executing from here
    const streamResponseEdited = client.runs.stream(
      thread["thread_id"],
      assistantId,
      {
        input: null,
        streamMode: "values",
        interruptBefore: ["action"],
      }
    );

    for await (const chunk of streamResponseEdited) {
      if (chunk.data && chunk.event !== "metadata") {
        console.log(chunk.data);
      }
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/state \
    --header 'Content-Type: application/json' \
    --data "{
        \"values\": { \"messages\": [$(curl --request GET \
            --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/state |
            jq -c '{
            role: "tool",
            content: "User requested changes: pass in the country as well",
            name: "get_weather",
            tool_call_id: .values.messages[-1].id.tool_calls[0].id
            }')
        ]},
        \"as_node\": \"human_review_node\"
    }" && echo "Resuming Execution" && curl --request POST \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
    --header 'Content-Type: application/json' \
    --data '{
    "assistant_id": "agent"
    }' | \
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

    Current State:
    {'messages': [{'content': "what's the weather in sf?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '3b2bbc38-d11b-49eb-80c0-c24a40dab5a8', 'example': False}, {'content': [{'text': 'To get the weather information for San Francisco, I can use the weather_search function. Let me do that for you.', 'type': 'text', 'index': 0}, {'id': 'toolu_01NNw18j57GEGPZvsa9f1wvX', 'input': {}, 'name': 'weather_search', 'type': 'tool_use', 'index': 1, 'partial_json': '{"city": "San Francisco"}'}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'tool_use', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-c5a50900-abf5-4885-9cdb-da2bf0d892ac', 'example': False, 'tool_calls': [{'name': 'weather_search', 'args': {'city': 'San Francisco'}, 'id': 'toolu_01NNw18j57GEGPZvsa9f1wvX', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 360, 'output_tokens': 80, 'total_tokens': 440}}]}

    Current Tool Call ID:
    toolu_01NNw18j57GEGPZvsa9f1wvX

    Resuming execution
    {'messages': [{'content': "what's the weather in sf?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '3b2bbc38-d11b-49eb-80c0-c24a40dab5a8', 'example': False}, {'content': [{'text': 'To get the weather information for San Francisco, I can use the weather_search function. Let me do that for you.', 'type': 'text', 'index': 0}, {'id': 'toolu_01NNw18j57GEGPZvsa9f1wvX', 'input': {}, 'name': 'weather_search', 'type': 'tool_use', 'index': 1, 'partial_json': '{"city": "San Francisco"}'}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'tool_use', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-c5a50900-abf5-4885-9cdb-da2bf0d892ac', 'example': False, 'tool_calls': [{'name': 'weather_search', 'args': {'city': 'San Francisco'}, 'id': 'toolu_01NNw18j57GEGPZvsa9f1wvX', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 360, 'output_tokens': 80, 'total_tokens': 440}}, {'content': 'User requested changes: pass in the country as well', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'weather_search', 'id': '787288be-213c-4fd3-8503-4a009bdb1b00', 'tool_call_id': 'toolu_01NNw18j57GEGPZvsa9f1wvX', 'artifact': None, 'status': 'success'}, {'content': [{'text': '\n\nI apologize for the oversight. It seems the function requires additional information. Let me try again with a more specific request.', 'type': 'text', 'index': 0}, {'id': 'toolu_01YAbLBoKozJyRQnB8LUMpXC', 'input': {}, 'name': 'weather_search', 'type': 'tool_use', 'index': 1, 'partial_json': '{"city": "San Francisco, USA"}'}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'tool_use', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-5c355a56-cfe3-4046-b49f-f5b09fc397ef', 'example': False, 'tool_calls': [{'name': 'weather_search', 'args': {'city': 'San Francisco, USA'}, 'id': 'toolu_01YAbLBoKozJyRQnB8LUMpXC', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 461, 'output_tokens': 83, 'total_tokens': 544}}]}

We can see that we now get to another breakpoint - because it went back to the model and got an entirely new prediction of what to call. Let's now approve this one and continue

=== "Python"

    ```python
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=None,
    ):
        if chunk.data and chunk.event != "metadata": 
            print(chunk.data)
    ```

=== "Javascript"

    ```js
    const streamResponseResumed = client.runs.stream(
      thread["thread_id"],
      assistantId,
      {
        input: null,
      }
    );

    for await (const chunk of streamResponseResumed) {
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
       \"assistant_id\": \"agent\"
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

    {'messages': [{'content': "what's the weather in sf?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '3b2bbc38-d11b-49eb-80c0-c24a40dab5a8', 'example': False}, {'content': [{'text': 'To get the weather information for San Francisco, I can use the weather_search function. Let me do that for you.', 'type': 'text', 'index': 0}, {'id': 'toolu_01NNw18j57GEGPZvsa9f1wvX', 'input': {}, 'name': 'weather_search', 'type': 'tool_use', 'index': 1, 'partial_json': '{"city": "San Francisco"}'}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'tool_use', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-c5a50900-abf5-4885-9cdb-da2bf0d892ac', 'example': False, 'tool_calls': [{'name': 'weather_search', 'args': {'city': 'San Francisco'}, 'id': 'toolu_01NNw18j57GEGPZvsa9f1wvX', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 360, 'output_tokens': 80, 'total_tokens': 440}}, {'content': 'User requested changes: pass in the country as well', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'weather_search', 'id': '787288be-213c-4fd3-8503-4a009bdb1b00', 'tool_call_id': 'toolu_01NNw18j57GEGPZvsa9f1wvX', 'artifact': None, 'status': 'success'}, {'content': [{'text': '\n\nI apologize for the oversight. It seems the function requires additional information. Let me try again with a more specific request.', 'type': 'text', 'index': 0}, {'id': 'toolu_01YAbLBoKozJyRQnB8LUMpXC', 'input': {}, 'name': 'weather_search', 'type': 'tool_use', 'index': 1, 'partial_json': '{"city": "San Francisco, USA"}'}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'tool_use', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-5c355a56-cfe3-4046-b49f-f5b09fc397ef', 'example': False, 'tool_calls': [{'name': 'weather_search', 'args': {'city': 'San Francisco, USA'}, 'id': 'toolu_01YAbLBoKozJyRQnB8LUMpXC', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 461, 'output_tokens': 83, 'total_tokens': 544}}, {'content': 'Sunny!', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'weather_search', 'id': '3b857482-bca2-4a73-a9ab-1f35a3e43e5f', 'tool_call_id': 'toolu_01YAbLBoKozJyRQnB8LUMpXC', 'artifact': None, 'status': 'success'}]}
    {'messages': [{'content': "what's the weather in sf?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '3b2bbc38-d11b-49eb-80c0-c24a40dab5a8', 'example': False}, {'content': [{'text': 'To get the weather information for San Francisco, I can use the weather_search function. Let me do that for you.', 'type': 'text', 'index': 0}, {'id': 'toolu_01NNw18j57GEGPZvsa9f1wvX', 'input': {}, 'name': 'weather_search', 'type': 'tool_use', 'index': 1, 'partial_json': '{"city": "San Francisco"}'}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'tool_use', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-c5a50900-abf5-4885-9cdb-da2bf0d892ac', 'example': False, 'tool_calls': [{'name': 'weather_search', 'args': {'city': 'San Francisco'}, 'id': 'toolu_01NNw18j57GEGPZvsa9f1wvX', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 360, 'output_tokens': 80, 'total_tokens': 440}}, {'content': 'User requested changes: pass in the country as well', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'weather_search', 'id': '787288be-213c-4fd3-8503-4a009bdb1b00', 'tool_call_id': 'toolu_01NNw18j57GEGPZvsa9f1wvX', 'artifact': None, 'status': 'success'}, {'content': [{'text': '\n\nI apologize for the oversight. It seems the function requires additional information. Let me try again with a more specific request.', 'type': 'text', 'index': 0}, {'id': 'toolu_01YAbLBoKozJyRQnB8LUMpXC', 'input': {}, 'name': 'weather_search', 'type': 'tool_use', 'index': 1, 'partial_json': '{"city": "San Francisco, USA"}'}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'tool_use', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-5c355a56-cfe3-4046-b49f-f5b09fc397ef', 'example': False, 'tool_calls': [{'name': 'weather_search', 'args': {'city': 'San Francisco, USA'}, 'id': 'toolu_01YAbLBoKozJyRQnB8LUMpXC', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 461, 'output_tokens': 83, 'total_tokens': 544}}, {'content': 'Sunny!', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'weather_search', 'id': '3b857482-bca2-4a73-a9ab-1f35a3e43e5f', 'tool_call_id': 'toolu_01YAbLBoKozJyRQnB8LUMpXC', 'artifact': None, 'status': 'success'}, {'content': [{'text': "\n\nGreat news! The weather in San Francisco is sunny today. Is there anything else you'd like to know about the weather or any other information I can help you with?", 'type': 'text', 'index': 0}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-6a857bb1-f65b-4b86-93d6-c025e003c777', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 557, 'output_tokens': 38, 'total_tokens': 595}}]}
