# How to wait for user input using `interrupt`

!!! tip "Prerequisites"

    This guide assumes familiarity with the following concepts:

    * [Human-in-the-loop](../../concepts/human_in_the_loop.md)
    * [LangGraph Glossary](../../concepts/low_level.md)
    

**Human-in-the-loop (HIL)** interactions are crucial for [agentic systems](../../concepts/agentic_concepts.md#human-in-the-loop). Waiting for human input is a common HIL interaction pattern, allowing the agent to ask the user clarifying questions and await input before proceeding.

We can implement this in LangGraph using the [`interrupt()`][langgraph.types.interrupt] function. `interrupt` allows us to stop graph execution to collect input from a user and continue execution with collected input.

## Setup

We are not going to show the full code for the graph we are hosting, but you can see it [here](../../how-tos/human_in_the_loop/wait-user-input.ipynb#agent) if you want to. Once this graph is hosted, we are ready to invoke it and wait for user input.

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

Now, let's invoke our graph.

=== "Python"

    ```python
    input = {
        "messages": [
            {
                "role": "user",
                "content": "Ask the user where they are, then look up the weather there",
            }
        ]
    }

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
    const input = {
      messages: [
        {
          role: "human",
          content: "Ask the user where they are, then look up the weather there" }
      ]
    };

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
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"Ask the user where they are, then look up the weather there\"}]},
       \"stream_mode\": [
         \"updates\"
       ]
     }"
    ```
    
Output:

    {'agent': {'messages': [{'content': [{'text': "I'll help you ask the user about their location and then search for weather information.", 'type': 'text'}, {'id': 'toolu_012JeNEvyePZFWK39d52Wdwi', 'input': {'question': 'Where are you located?'}, 'name': 'AskHuman', 'type': 'tool_use'}], 'additional_kwargs': {}, 'response_metadata': {'id': 'msg_01UBEdS6UvuFMetdokNsykVG', 'model': 'claude-3-5-sonnet-20241022', 'stop_reason': 'tool_use', 'stop_sequence': None, 'usage': {'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0, 'input_tokens': 438, 'output_tokens': 76}, 'model_name': 'claude-3-5-sonnet-20241022'}, 'type': 'ai', 'name': None, 'id': 'run-1b1210d8-39e0-4607-9f0e-0ea932d28d5c-0', 'example': False, 'tool_calls': [{'name': 'AskHuman', 'args': {'question': 'Where are you located?'}, 'id': 'toolu_012JeNEvyePZFWK39d52Wdwi', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 438, 'output_tokens': 76, 'total_tokens': 514, 'input_token_details': {'cache_read': 0, 'cache_creation': 0}}}]}}
    {'__interrupt__': [{'value': 'Where are you located?', 'resumable': True, 'ns': ['ask_human:2d41f894-f297-211e-9bfe-1d162ecba54a'], 'when': 'during'}]}

You can see that our graph got interrupted inside the `ask_human` node, which is now waiting for a `location` to be provided.

### Providing human input

We can provide human input (`location`) by invoking the graph with a `Command(resume="<location>")`:

=== "Python"

    ```python
    # highlight-next-line
    from langgraph_sdk.schema import Command

    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        # highlight-next-line
        command=Command(resume="san francisco"),
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
        command: { resume: "san francisco" },
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
         \"resume\": \"san francisco\"
       },
       \"stream_mode\": [
         \"updates\"
       ]
     }"
    ```

Output:

    {'ask_human': {'messages': [{'tool_call_id': 'toolu_012JeNEvyePZFWK39d52Wdwi', 'type': 'tool', 'content': 'san francisco'}]}}
    {'agent': {'messages': [{'content': [{'text': 'Let me search for the weather in San Francisco.', 'type': 'text'}, {'id': 'toolu_019f9Y7ST6rNeDQkDjFCHk6C', 'input': {'query': 'current weather in san francisco'}, 'name': 'search', 'type': 'tool_use'}], 'additional_kwargs': {}, 'response_metadata': {'id': 'msg_0152YFm7DtnzfZQuiMUzaSsw', 'model': 'claude-3-5-sonnet-20241022', 'stop_reason': 'tool_use', 'stop_sequence': None, 'usage': {'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0, 'input_tokens': 527, 'output_tokens': 67}, 'model_name': 'claude-3-5-sonnet-20241022'}, 'type': 'ai', 'name': None, 'id': 'run-f509b5b2-eb30-4200-a8da-fa79ed68812a-0', 'example': False, 'tool_calls': [{'name': 'search', 'args': {'query': 'current weather in san francisco'}, 'id': 'toolu_019f9Y7ST6rNeDQkDjFCHk6C', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 527, 'output_tokens': 67, 'total_tokens': 594, 'input_token_details': {'cache_read': 0, 'cache_creation': 0}}}]}}
    {'action': {'messages': [{'content': "I looked up: current weather in san francisco. Result: It's sunny in San Francisco, but you better look out if you're a Gemini 😈.", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'search', 'id': 'cbd0f623-cc12-48a2-8c18-3cbb943e46e0', 'tool_call_id': 'toolu_019f9Y7ST6rNeDQkDjFCHk6C', 'artifact': None, 'status': 'success'}]}}
    {'agent': {'messages': [{'content': "Based on the search results, it's currently sunny in San Francisco. Would you like any specific details about the weather forecast?", 'additional_kwargs': {}, 'response_metadata': {'id': 'msg_01FhzXj72CehBYkJGX69vsBc', 'model': 'claude-3-5-sonnet-20241022', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0, 'input_tokens': 639, 'output_tokens': 29}, 'model_name': 'claude-3-5-sonnet-20241022'}, 'type': 'ai', 'name': None, 'id': 'run-f48e818e-dd88-415e-9a0b-4a958498b553-0', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 639, 'output_tokens': 29, 'total_tokens': 668, 'input_token_details': {'cache_read': 0, 'cache_creation': 0}}}]}}