# How to review tool calls

!!! tip "Prerequisites"

    This guide assumes familiarity with the following concepts:

    * [Tool calling](https://python.langchain.com/docs/concepts/tool_calling/)
    * [Human-in-the-loop](../../concepts/human_in_the_loop.md)
    * [LangGraph Glossary](../../concepts/low_level.md)

Human-in-the-loop (HIL) interactions are crucial for [agentic systems](../../concepts/agentic_concepts.md). A common pattern is to add some human in the loop step after certain tool calls. These tool calls often lead to either a function call or saving of some information. Examples include:

- A tool call to execute SQL, which will then be run by the tool
- A tool call to generate a summary, which will then be saved to the State of the graph

Note that using tool calls is common **whether actually calling tools or not**.

There are typically a few different interactions you may want to do here:

1. Approve the tool call and continue
2. Modify the tool call manually and then continue
3. Give natural language feedback, and then pass that back to the agent


We can implement these in LangGraph using the [`interrupt()`][langgraph.types.interrupt] function. `interrupt` allows us to stop graph execution to collect input from a user and continue execution with collected input:


```python
def human_review_node(state) -> Command[Literal["call_llm", "run_tool"]]:
    # this is the value we'll be providing via Command(resume=<human_review>)
    human_review = interrupt(
        {
            "question": "Is this correct?",
            # Surface tool calls for review
            "tool_call": tool_call
        }
    )

    review_action, review_data = human_review

    # Approve the tool call and continue
    if review_action == "continue":
        return Command(goto="run_tool")

    # Modify the tool call manually and then continue
    elif review_action == "update":
        ...
        updated_msg = get_updated_msg(review_data)
        return Command(goto="run_tool", update={"messages": [updated_message]})

    # Give natural language feedback, and then pass that back to the agent
    elif review_action == "feedback":
        ...
        feedback_msg = get_feedback_msg(review_data)
        return Command(goto="call_llm", update={"messages": [feedback_msg]})

```

## Setup

We are not going to show the full code for the graph we are hosting, but you can see it [here](../../how-tos/human_in_the_loop/review-tool-calls.ipynb). Once this graph is hosted, we are ready to invoke it and wait for user input. 

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

## Example of approving tool

First, let's run the agent with an input that requires tool calls with approval:

=== "Python"

    ```python
    input = {"messages": [{"role": "user", "content": "what's the weather in sf?"}]}

    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=input,
        stream_mode="updates",
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
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"what's the weather in sf?\"}]},
       \"stream_mode\": [
         \"updates\"
       ]
     }"
    ```

Output:

    {'call_llm': {'messages': [{'content': [{'text': "I'll help you check the weather in San Francisco.", 'type': 'text'}, {'id': 'toolu_01142G3woscA8JjFTLdqymtn', 'input': {'city': 'San Francisco'}, 'name': 'weather_search', 'type': 'tool_use'}], 'additional_kwargs': {}, 'response_metadata': {'id': 'msg_01Tdfufy4nZYXMbVZvgyNbhc', 'model': 'claude-3-5-sonnet-20241022', 'stop_reason': 'tool_use', 'stop_sequence': None, 'usage': {'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0, 'input_tokens': 379, 'output_tokens': 66}, 'model_name': 'claude-3-5-sonnet-20241022'}, 'type': 'ai', 'name': None, 'id': 'run-a33434b2-f5ca-40c6-98e2-6288d349d4ce-0', 'example': False, 'tool_calls': [{'name': 'weather_search', 'args': {'city': 'San Francisco'}, 'id': 'toolu_01142G3woscA8JjFTLdqymtn', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 379, 'output_tokens': 66, 'total_tokens': 445, 'input_token_details': {'cache_read': 0, 'cache_creation': 0}}}]}}
    {'__interrupt__': [{'value': {'question': 'Is this correct?', 'tool_call': {'name': 'weather_search', 'args': {'city': 'San Francisco'}, 'id': 'toolu_01142G3woscA8JjFTLdqymtn', 'type': 'tool_call'}}, 'resumable': True, 'ns': ['human_review_node:9caf42cf-1371-7213-a331-e6fe5d026be8'], 'when': 'during'}]}

To approve the tool call, we need to let `human_review_node` know what value to use for the `human_review` variable we defined inside the node. We can provide this value by invoking the graph with a `Command(resume=<human_review>)` input.  Since we're approving the tool call, we'll provide `resume` value of `{"action": "continue"}` to navigate to `run_tool` node:

=== "Python"

    ```python
    # highlight-next-line
    from langgraph_sdk.schema import Command

    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        # highlight-next-line
        command=Command(resume={"action": "continue"}),
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
        // highlight-next-line
        command: { resume: { "action": "continue" } },
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
       \"command\": {
         \"resume\": { \"action\": \"continue\"}
       },
       \"stream_mode\": [
         \"updates\"
       ]
     }"
    ```

Output:

    {'human_review_node': None}
    {'run_tool': {'messages': [{'role': 'tool', 'name': 'weather_search', 'content': 'Sunny!', 'tool_call_id': 'toolu_01142G3woscA8JjFTLdqymtn'}]}}
    {'call_llm': {'messages': [{'content': "According to the search, it's sunny in San Francisco right now!", 'additional_kwargs': {}, 'response_metadata': {'id': 'msg_01JJE9AtT4a9Lob91RRiW9rU', 'model': 'claude-3-5-sonnet-20241022', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0, 'input_tokens': 458, 'output_tokens': 18}, 'model_name': 'claude-3-5-sonnet-20241022'}, 'type': 'ai', 'name': None, 'id': 'run-5e8d80b5-c46a-4aad-af37-b01f8bb15963-0', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 458, 'output_tokens': 18, 'total_tokens': 476, 'input_token_details': {'cache_read': 0, 'cache_creation': 0}}}]}}

## Edit Tool Call

Let's now say we want to edit the tool call. E.g. change some of the parameters (or even the tool called!) but then execute that tool.

=== "Python"

    ```python
    input = {"messages": [{"role": "user", "content": "what's the weather in sf?"}]}

    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=input,
        stream_mode="updates",
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
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"what's the weather in sf?\"}]},
       \"stream_mode\": [
         \"updates\"
       ]
     }"
    ```

To do this, we will use `Command` with a different resume value of `{"action": "update", "data": <tool call args>}`. This will do the following:

* combine existing tool call with user-provided tool call arguments and update the existing AI message with the new tool call
* navigate to `run_tool` node with the updated AI message and continue execution


=== "Python"

    ```python
    # highlight-next-line
    from langgraph_sdk.schema import Command

    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        # highlight-next-line
        command=Command(
            # highlight-next-line
            resume={"action": "update", "data": {"city": "San Francisco, USA"}}
            # highlight-next-line
        ),
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
        // highlight-next-line
        command: {
          // highlight-next-line
          resume: { "action": "update", "data": { "city": "San Francisco, USA" } }
          // highlight-next-line
        },
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
       \"command\": {
         \"resume\": { \"action\": \"update\", \"data\": { \"city\": \"San Francisco, USA\" } }
       },
       \"stream_mode\": [
         \"updates\"
       ]
     }"
    ```

Output:

    {'human_review_node': {'messages': [{'role': 'ai', 'content': [{'text': "I'll help you check the weather in San Francisco.", 'type': 'text'}, {'id': 'toolu_016L4EDPcaQRzzZxiB4Wq2wa', 'input': {'city': 'San Francisco'}, 'name': 'weather_search', 'type': 'tool_use'}], 'tool_calls': [{'id': 'toolu_016L4EDPcaQRzzZxiB4Wq2wa', 'name': 'weather_search', 'args': {'city': 'San Francisco, USA'}}], 'id': 'run-b07f0c35-4e93-43a5-9b48-363767ada3ca-0'}]}}
    {'run_tool': {'messages': [{'role': 'tool', 'name': 'weather_search', 'content': 'Sunny!', 'tool_call_id': 'toolu_016L4EDPcaQRzzZxiB4Wq2wa'}]}}
    {'call_llm': {'messages': [{'content': "According to the search, it's sunny in San Francisco right now!", 'additional_kwargs': {}, 'response_metadata': {'id': 'msg_01De5HurjNUMwMUpfRtMLbX1', 'model': 'claude-3-5-sonnet-20241022', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0, 'input_tokens': 460, 'output_tokens': 18}, 'model_name': 'claude-3-5-sonnet-20241022'}, 'type': 'ai', 'name': None, 'id': 'run-85e2aaaa-6f61-4fa0-b594-b6e57129d7e7-0', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 460, 'output_tokens': 18, 'total_tokens': 478, 'input_token_details': {'cache_read': 0, 'cache_creation': 0}}}]}}

## Give feedback to a tool call

Sometimes, you may not want to execute a tool call, but you also may not want to ask the user to manually modify the tool call. In that case it may be better to get natural language feedback from the user. You can then insert this feedback as a mock **RESULT** of the tool call.

There are multiple ways to do this:

1. You could add a new message to the state (representing the "result" of a tool call)
2. You could add TWO new messages to the state - one representing an "error" from the tool call, other HumanMessage representing the feedback

Both are similar in that they involve adding messages to the state. The main difference lies in the logic AFTER the `human_review_node` and how it handles different types of messages.

For this example we will just add a single tool call representing the feedback (see `human_review_node` implementation). Let's see this in action!

=== "Python"

    ```python
    input = {"messages": [{"role": "user", "content": "what's the weather in sf?"}]}

    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=input,
        stream_mode="updates",
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
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"what's the weather in sf?\"}]},
       \"stream_mode\": [
         \"updates\"
       ]
     }"
    ```

To do this, we will use `Command` with a different resume value of `{"action": "feedback", "data": <feedback string>}`. This will do the following:

* create a new tool message that combines existing tool call from LLM with the with user-provided feedback as content
* navigate to `call_llm` node with the updated tool message and continue execution

=== "Python"

    ```python
    # highlight-next-line
    from langgraph_sdk.schema import Command

    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        # highlight-next-line
        command=Command(
            resume={
                "action": "feedback", 
                "data": "User requested changes: use <city, country> format for location"
            }
        ),
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
        // highlight-next-line
        command: {
          resume: {
            "action": "feedback", 
            "data": "User requested changes: use <city, country> format for location"
          }
        },
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
       \"command\": {
         \"resume\": { \"action\": \"feedback\", \"data\": \"User requested changes: use <city, country> format for location\" }
       },
       \"stream_mode\": [
         \"updates\"
       ]
     }"
    ```


Output:

    {'human_review_node': {'messages': [{'role': 'tool', 'content': 'User requested changes: use <city, country> format for location', 'name': 'weather_search', 'tool_call_id': 'toolu_01RkPHCjpfoUvPAktaq4Cqhm'}]}}
    {'call_llm': {'messages': [{'content': [{'text': 'Let me try that again with the correct format:', 'type': 'text'}, {'id': 'toolu_01Rdrag6cVufHZG26BwVaiE7', 'input': {'city': 'San Francisco, USA'}, 'name': 'weather_search', 'type': 'tool_use'}], 'additional_kwargs': {}, 'response_metadata': {'id': 'msg_01EBan969yY5f6iGk6sPgKcj', 'model': 'claude-3-5-sonnet-20241022', 'stop_reason': 'tool_use', 'stop_sequence': None, 'usage': {'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0, 'input_tokens': 469, 'output_tokens': 68}, 'model_name': 'claude-3-5-sonnet-20241022'}, 'type': 'ai', 'name': None, 'id': 'run-64bbc255-d126-4db0-8ae5-3197cf29bed1-0', 'example': False, 'tool_calls': [{'name': 'weather_search', 'args': {'city': 'San Francisco, USA'}, 'id': 'toolu_01Rdrag6cVufHZG26BwVaiE7', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 469, 'output_tokens': 68, 'total_tokens': 537, 'input_token_details': {'cache_read': 0, 'cache_creation': 0}}}]}}
    {'__interrupt__': [{'value': {'question': 'Is this correct?', 'tool_call': {'name': 'weather_search', 'args': {'city': 'San Francisco, USA'}, 'id': 'toolu_01Rdrag6cVufHZG26BwVaiE7', 'type': 'tool_call'}}, 'resumable': True, 'ns': ['human_review_node:e9856878-e28c-5dd1-d353-4d83aa1a3a2b'], 'when': 'during'}]}

We can see that we now get to another interrupt - because it went back to the model and got an entirely new prediction of what to call. Let's now approve this one and continue.

=== "Python"

    ```python
    # highlight-next-line
    from langgraph_sdk.schema import Command

    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        # highlight-next-line
        command=Command(resume={"action": "continue"}),
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
        // highlight-next-line
        command: { resume: { "action": "continue" } },
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
       \"command\": {
         \"resume\": { \"action\": \"continue\"}
       },
       \"stream_mode\": [
         \"updates\"
       ]
     }"
    ```

Output:

    {'human_review_node': None}
    {'run_tool': {'messages': [{'role': 'tool', 'name': 'weather_search', 'content': 'Sunny!', 'tool_call_id': 'toolu_01Rdrag6cVufHZG26BwVaiE7'}]}}
    {'call_llm': {'messages': [{'content': 'The weather in San Francisco is sunny!', 'additional_kwargs': {}, 'response_metadata': {'id': 'msg_013WTDHhbg8WiYLiQ9n2CaTk', 'model': 'claude-3-5-sonnet-20241022', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0, 'input_tokens': 550, 'output_tokens': 12}, 'model_name': 'claude-3-5-sonnet-20241022'}, 'type': 'ai', 'name': None, 'id': 'run-b6c815f0-989a-47cf-b150-33e3bbc4eab7-0', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 550, 'output_tokens': 12, 'total_tokens': 562, 'input_token_details': {'cache_read': 0, 'cache_creation': 0}}}]}}